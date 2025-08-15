import logging
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.decomposition import PCA
from torchvision.transforms.functional import pil_to_tensor

import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

logger = logging.getLogger(__name__)

class DINOV3ModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`DINOV3Model`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        model_name: the DINOV3 model name to load from Hugging Face (default: "facebook/dinov3-vits16-pretrain-lvd1689m")
        model_path: optional path to the saved model file on disk
        output_type: what to return - "cls", "register", "patch", or "all"
        return_attention_maps: whether to return attention maps
        use_mixed_precision: whether to use mixed precision for inference
    """

    def __init__(self, d):
        super().__init__(d)

        self.model_name = self.parse_string(d, "model_name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
        self.model_path = self.parse_string(d, "model_path")
        self.output_type = self.parse_string(d, "output_type", default="cls")
        self.return_attention_maps = self.parse_bool(d, "return_attention_maps", default=False)
        self.use_external_preprocessor = self.parse_bool(d, "use_external_preprocessor", default=False)
        self.use_mixed_precision = self.parse_bool(d, "use_mixed_precision", default=True)
        self.apply_smoothing = self.parse_bool(d, "apply_smoothing", default=True)
        self.smoothing_sigma = self.parse_number(d, "smoothing_sigma", default=1.51)

class DINOV3Model(fout.TorchImageModel):
    """Wrapper for DINOV3 models from .

    Args:
        config: a :class:`DINOV3ModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Load the DINOV3 model and setup preprocessor
        self._dinov3_model = self._load_dinov3_model()
        if config.use_external_preprocessor:
            self._conditioner = self._dinov3_model.make_preprocessor_external()
        else:
            self._conditioner = None
            
    def _check_mixed_precision_support(self):
        """Check if the current GPU supports mixed precision with bfloat16."""
        if not self._using_gpu:
            return False
            
        try:
            # Check GPU capability
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(self._device)
                # bfloat16 is supported on Ampere (8.0+) and newer architectures
                return device_capability[0] >= 8
            return False
        except Exception as e:
            logger.warning(f"Could not determine mixed precision support: {e}")
            return False

    def _load_model(self, config):
        """Load the DINOV3 model from Hugging Face or disk."""
        from transformers import AutoImageProcessor, AutoModel
        import os
        
        # Load from local path if provided
        if config.model_path and os.path.exists(config.model_path):
            logger.info(f"Loading DINOV3 model from local path: {config.model_path}")
            model = AutoModel.from_pretrained(config.model_path)
            self._processor = AutoImageProcessor.from_pretrained(config.model_path)
        else:
            # Load from Hugging Face hub
            logger.info(f"Loading DINOV3 model from Hugging Face: {config.model_name}")
            model = AutoModel.from_pretrained(config.model_name)
            self._processor = AutoImageProcessor.from_pretrained(config.model_name)
        
        # Store model info
        self._patch_size = model.config.patch_size
        self._hidden_size = model.config.hidden_size
        self._num_register_tokens = getattr(model.config, "num_register_tokens", 4)
        logger.info(f"DINOV3 model loaded: patch_size={self._patch_size}, num_register_tokens={self._num_register_tokens}")
        
        return model

    def _load_dinov3_model(self):
        """Load and setup the DINOV3 model."""
        model = self._load_model(self.config)
        # Ensure the model is on the correct device and in eval mode
        model = model.to(self._device)
        model.eval()
        return model

    def _preprocess_image(self, img):
        """Preprocess a single image for DINOV3 model.
        
        Args:
            img: PIL Image, numpy array, or torch tensor
            
        Returns:
            preprocessed tensor ready for DINOV3 model
        """
        # Convert to PIL if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            # Convert tensor back to PIL for consistent preprocessing
            if img.dim() == 3:  # CHW
                img = img.permute(1, 2, 0)  # HWC
            img = img.cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use the DINOV3 processor from HuggingFace for proper preprocessing
        if hasattr(self, '_processor'):
            inputs = self._processor(images=img, return_tensors="pt")
            # Move tensor to correct device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            return inputs
        else:
            # Fallback to simple normalization if processor is not available
            x = pil_to_tensor(img).to(dtype=torch.float32)
            x.div_(255.0)  # normalize to [0, 1]
            x = x.to(self._device)  # Move to device after preprocessing
            return {"pixel_values": x}

    def _predict_all(self, imgs):
        """Apply DINOV3 model to batch of images."""
        
        # Debug: check input types and devices
        logger.debug(f"Input imgs type: {type(imgs)}, length: {len(imgs) if hasattr(imgs, '__len__') else 'unknown'}")
        if len(imgs) > 0:
            logger.debug(f"First img type: {type(imgs[0])}")
            if isinstance(imgs[0], torch.Tensor):
                logger.debug(f"First img device: {imgs[0].device}")
        
        # Preprocess images if needed
        if self._preprocess and self._transforms is not None:
            imgs = [self._transforms(img) for img in imgs]
            # Ensure all tensors are on the correct device
            imgs = [img.to(self._device) if isinstance(img, torch.Tensor) else img for img in imgs]
        else:
            # Apply our custom preprocessing (already handles device placement)
            imgs = [self._preprocess_image(img) for img in imgs]

        # Debug: check preprocessed images
        logger.debug(f"After preprocessing, first img device: {imgs[0].device if isinstance(imgs[0], torch.Tensor) else 'not tensor'}")

        # Process each image individually
        cls_tokens = []
        register_tokens = []
        patch_tokens = []
        
        for i, img in enumerate(imgs):
            try:
                # Ensure tensor is on the correct device
                if isinstance(img, torch.Tensor):
                    img = img.to(self._device)
                    logger.debug(f"Image {i} device after .to(): {img.device}")
                
                # Add batch dimension if needed
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                
                # Prepare inputs with the DINOV3 processor
                # Convert PIL image if needed
                if not isinstance(img, torch.Tensor):
                    processed_inputs = self._processor(images=img, return_tensors="pt")
                    processed_inputs = {k: v.to(self._device) for k, v in processed_inputs.items()}
                else:
                    # Handle tensor inputs - resize to model's expected input size if needed
                    if img.shape[-1] != 224 or img.shape[-2] != 224:
                        img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
                    processed_inputs = {"pixel_values": img}
                
                # Forward pass with optional mixed precision
                use_mixed_precision = (
                    getattr(self.config, 'use_mixed_precision', False) and 
                    getattr(self, '_mixed_precision_supported', False) and 
                    self._using_gpu
                )
                
                if use_mixed_precision:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        with torch.no_grad():
                            outputs = self._dinov3_model(**processed_inputs)
                else:
                    with torch.no_grad():
                        outputs = self._dinov3_model(**processed_inputs)
                
                # Extract different token types from the output
                last_hidden_states = outputs.last_hidden_state
                
                # Extract CLS token [batch_size, 1, hidden_size]
                cls_token = last_hidden_states[:, 0:1, :]
                
                # Extract register tokens [batch_size, num_register_tokens, hidden_size]
                reg_tokens = last_hidden_states[:, 1:1+self._num_register_tokens, :]
                
                # Extract patch tokens [batch_size, num_patches, hidden_size]
                patch_token = last_hidden_states[:, 1+self._num_register_tokens:, :]
                
                # Calculate grid dimensions
                batch_size, num_patches, hidden_dim = patch_token.shape
                h = w = int(math.sqrt(num_patches))
                
                # Reshape patch tokens to spatial grid [batch_size, h, w, hidden_dim]
                patch_token_spatial = patch_token.reshape(batch_size, h, w, hidden_dim)
                
                # Add results to lists
                cls_tokens.append(cls_token)
                register_tokens.append(reg_tokens)
                patch_tokens.append(patch_token)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                logger.error(f"Image {i} shape: {img.shape if hasattr(img, 'shape') else 'no shape'}")
                logger.error(f"Image {i} device: {img.device if isinstance(img, torch.Tensor) else 'not tensor'}")
                logger.error(f"Model device: {next(self._dinov3_model.parameters()).device}")
                raise
        
        # Stack results
        try:
            batch_cls = torch.cat(cls_tokens, dim=0)
            batch_register = torch.cat(register_tokens, dim=0)
            batch_patch = torch.cat(patch_tokens, dim=0)
        except Exception as e:
            logger.error(f"Error stacking results: {e}")
            raise
        
        # Return based on output type
        if self.config.output_type == "cls":
            output = batch_cls.squeeze(1)  # Remove the extra dimension for CLS token
        elif self.config.output_type == "register":
            output = batch_register
        elif self.config.output_type == "patch":
            output = batch_patch
        elif self.config.output_type == "all":
            # Return a dictionary with all token types
            return {
                "cls": [t.squeeze(1).detach().cpu().numpy() for t in cls_tokens],
                "register": [t.detach().cpu().numpy() for t in register_tokens],
                "patch": [t.detach().cpu().numpy() for t in patch_tokens]
            }
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")
        
        # Process output if we have an output processor
        if self._output_processor is not None:
            # Collect original frame sizes for batch
            frame_sizes = []
            for img in imgs:
                if hasattr(img, 'shape') and len(img.shape) >= 2:
                    h, w = img.shape[-2:]
                    frame_sizes.append((w, h))  # (width, height) format
                else:
                    # Fallback - this shouldn't happen but just in case
                    frame_sizes.append((224, 224))
            
            return self._output_processor(
                output, frame_sizes, confidence_thresh=self.config.confidence_thresh
            )
        
        # Return raw features as numpy arrays for embeddings
        return [output[i].detach().cpu().numpy() for i in range(len(imgs))]

    def _check_mixed_precision_support(self):
        """Check if the current GPU supports mixed precision with bfloat16."""
        if not self._using_gpu:
            return False
            
        try:
            # Check GPU capability
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability(self._device)
                # bfloat16 is supported on Ampere (8.0+) and newer architectures
                return device_capability[0] >= 8
            return False
        except Exception as e:
            logger.warning(f"Could not determine mixed precision support: {e}")
            return False


class DINOV3OutputProcessor(fout.OutputProcessor):
    """Output processor for DINOV3 models that handles embeddings output.
    
    This processor can handle different types of embeddings from DINOV3 models:
    - CLS token embeddings (global image representation)
    - Register token embeddings (global information memory slots)
    - Patch token embeddings (local patch representations)
    """
    
    def __init__(self, output_type="cls", **kwargs):
        super().__init__(**kwargs)
        self.output_type = output_type
        
    def __call__(self, output, frame_size, confidence_thresh=None):
        """Process DINOV3 model output into embeddings.
        
        Args:
            output: tensor from DINOV3 model (depends on output_type)
            frame_size: (width, height) - not used for embeddings
            confidence_thresh: not used for embeddings
            
        Returns:
            list of numpy arrays containing embeddings
        """
        if isinstance(output, dict):
            # Handle dictionary output with all embedding types
            return output
            
        batch_size = output.shape[0]
        return [output[i].detach().cpu().numpy() for i in range(batch_size)]

class DINOV3HeatmapOutputProcessor(fout.OutputProcessor):
    """Spatial heatmap processor for DINOV3 patch embeddings.
    
    This processor visualizes DINOV3 patch embeddings as colorful attention heatmaps using PCA,
    similar to the official Meta DINOV3 visualization techniques. The visualizations help
    understand what parts of the image the model is focusing on and different semantic regions.
    """

    def __init__(self, apply_smoothing=True, smoothing_sigma=1.0, pca_components=3, 
                 use_color=True, foreground_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma
        self.pca_components = pca_components if use_color else 1
        self.use_color = use_color
        self.foreground_threshold = foreground_threshold

    def __call__(self, output, frame_sizes, confidence_thresh=None):
        """
        Args:
            output: torch.Tensor of shape [B, num_patches, hidden_dim] or [B, H, W, hidden_dim]
            frame_sizes: list of (width, height) for each image
            confidence_thresh: used as foreground threshold if provided

        Returns:
            List of fol.Heatmap instances
        """
        batch_size = output.shape[0]
        heatmaps = []
        
        # Use confidence_thresh if provided, otherwise use the default
        foreground_threshold = confidence_thresh if confidence_thresh is not None else self.foreground_threshold

        for i in range(batch_size):
            # Get patch tokens for this image
            patch_embedding = output[i].detach().cpu().numpy()
            
            # Handle different input formats
            if len(patch_embedding.shape) == 2:  # [num_patches, hidden_dim]
                num_patches, hidden_dim = patch_embedding.shape
                h = w = int(math.sqrt(num_patches))  # Assume square grid
                patch_embedding = patch_embedding.reshape(h, w, hidden_dim)
            elif len(patch_embedding.shape) == 3:  # [h, w, hidden_dim]
                h, w, hidden_dim = patch_embedding.shape
            else:
                raise ValueError(f"Unexpected patch embedding shape: {patch_embedding.shape}")

            # Flatten spatial grid to [H*W, hidden_dim] for PCA
            reshaped = patch_embedding.reshape(-1, hidden_dim)
            
            # Calculate L2 norm for each patch embedding (used for foreground masking)
            feature_norms = np.linalg.norm(reshaped, axis=1).reshape(h, w)
            
            # Normalize the norms to [0, 1] for foreground masking
            norm_min, norm_max = feature_norms.min(), feature_norms.max()
            if norm_max > norm_min:
                normalized_norms = (feature_norms - norm_min) / (norm_max - norm_min)
            else:
                normalized_norms = np.zeros_like(feature_norms)
                
            # Create foreground mask based on feature norms
            foreground_mask = normalized_norms > foreground_threshold

            try:
                # Apply PCA to get color or grayscale visualization
                pca = PCA(n_components=self.pca_components, whiten=True)
                pca_result = pca.fit_transform(reshaped)
                
                if self.use_color and self.pca_components >= 3:
                    # Reshape to spatial grid with 3 channels [h, w, 3]
                    pca_image = pca_result[:, :3].reshape(h, w, 3)
                    
                    # Convert to torch tensor for processing
                    pca_tensor = torch.from_numpy(pca_image)
                    
                    # Apply sigmoid scaling for vibrant colors (from Meta's approach)
                    pca_tensor = torch.sigmoid(pca_tensor.mul(2.0))
                    
                    # Apply foreground masking
                    mask_tensor = torch.from_numpy(foreground_mask).unsqueeze(-1).float()
                    pca_tensor = pca_tensor * mask_tensor
                    
                    # Convert to numpy and prepare for final output
                    pca_image = pca_tensor.numpy()
                    
                    # Optional smoothing
                    if self.apply_smoothing:
                        for c in range(pca_image.shape[2]):
                            pca_image[:, :, c] = gaussian_filter(pca_image[:, :, c], sigma=self.smoothing_sigma)
                    
                    # Resize to match original image dimensions
                    orig_w, orig_h = frame_sizes[i]
                    color_resized = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
                    
                    for c in range(3):
                        color_resized[:, :, c] = resize(
                            pca_image[:, :, c],
                            (orig_h, orig_w),
                            preserve_range=True,
                            anti_aliasing=True
                        )
                    
                    # Convert to uint8 [0, 255]
                    color_uint8 = (color_resized * 255).astype(np.uint8)
                    
                    # Create RGB heatmap
                    heatmap = fol.Heatmap(
                        map=color_uint8,
                        range=[0, 255]
                    )
                else:
                    # Single component PCA for grayscale visualization
                    attention_1d = pca_result[:, 0].reshape(h, w)
                    
                    # Apply foreground masking
                    attention_1d = attention_1d * foreground_mask
                    
                    # Optional smoothing
                    if self.apply_smoothing:
                        attention_1d = gaussian_filter(attention_1d, sigma=self.smoothing_sigma)
                    
                    # Resize to match original image dimensions
                    orig_w, orig_h = frame_sizes[i]
                    attention_resized = resize(
                        attention_1d,
                        (orig_h, orig_w),
                        preserve_range=True,
                        anti_aliasing=True
                    )
                    
                    # Normalize to uint8 [0, 255]
                    att_min, att_max = attention_resized.min(), attention_resized.max()
                    if att_max > att_min:
                        attention_uint8 = ((attention_resized - att_min) / (att_max - att_min) * 255).astype(np.uint8)
                    else:
                        attention_uint8 = np.zeros_like(attention_resized, dtype=np.uint8)
                    
                    # Create grayscale heatmap
                    heatmap = fol.Heatmap(
                        map=attention_uint8,
                        range=[0, 255]
                    )
            except Exception as e:
                # Fallback to simple feature norm if PCA fails
                warnings.warn(f"PCA failed on image {i}: {e}. Falling back to feature norm.")
                
                # Use the already computed feature norms
                attention_1d = normalized_norms * foreground_mask
                
                # Optional smoothing
                if self.apply_smoothing:
                    attention_1d = gaussian_filter(attention_1d, sigma=self.smoothing_sigma)
                
                # Resize to match original image dimensions
                orig_w, orig_h = frame_sizes[i]
                attention_resized = resize(
                    attention_1d,
                    (orig_h, orig_w),
                    preserve_range=True,
                    anti_aliasing=True
                )
                
                # Normalize to uint8 [0, 255]
                attention_uint8 = (attention_resized * 255).astype(np.uint8)
                
                heatmap = fol.Heatmap(
                    map=attention_uint8,
                    range=[0, 255]
                )
                
            heatmaps.append(heatmap)

        return heatmaps