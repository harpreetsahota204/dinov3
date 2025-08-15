import os
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

from transformers import AutoImageProcessor, AutoModel

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
        output_type: what to return - "cls", "attention_map", or "patch"
        use_mixed_precision: whether to use mixed precision for inference
    """

    def __init__(self, d):
        super().__init__(d)

        self.model_name = self.parse_string(d, "model_name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
        self.model_path = self.parse_string(d, "model_path")
        self.output_type = self.parse_string(d, "output_type", default="cls")
        self.apply_smoothing = self.parse_bool(d, "apply_smoothing", default=True)
        self.aggregration = self.parse_bool(d, "aggregration", default=True)
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

        logger.info(f"Loading DINOV3 model from local path: {config.model_path}")
        model = AutoModel.from_pretrained(config.model_path)
        self._processor = AutoImageProcessor.from_pretrained(config.model_path)

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
        elif self.config.output_type == "attention_map":
            output = batch_register
        elif self.config.output_type == "patch":
            output = batch_patch
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
        batch_size = output.shape[0]
        return [output[i].detach().cpu().numpy() for i in range(batch_size)]



class DINOV3PatchOutputProcessor(fout.OutputProcessor):
    """Spatial heatmap processor for DINOV3 patch embeddings.
    
    This processor visualizes DINOV3 patch embeddings as colorful attention heatmaps using PCA,
    similar to the official Meta DINOV3 visualization techniques. The visualizations help
    understand what parts of the image the model is focusing on and different semantic regions.
    
    Expected input: Patch tokens of shape [batch_size, 196, 384] for 14x14 spatial grid
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
            output: torch.Tensor of shape [B, num_patches, hidden_dim] where num_patches=196
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
            
            # Validate we have patch tokens (196 for 14x14 grid)
            if len(patch_embedding.shape) == 2:  # [num_patches, hidden_dim]
                num_patches, hidden_dim = patch_embedding.shape
                if num_patches != 196:
                    warnings.warn(f"Expected 196 patches, got {num_patches}. This may not be patch tokens.")
                h = w = int(np.sqrt(num_patches))  # Should be 14x14
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


class DINOV3RegisterOutputProcessor(fout.OutputProcessor):
    """Visualization processor for DINOV3 register tokens.
    
    Register tokens are global memory slots that capture different aspects of the image.
    This processor creates visualizations by computing similarity between register tokens
    and patch tokens, showing which image regions each register token focuses on.
    
    Expected input: Register tokens of shape [batch_size, 4, 384] 
    Requires access to patch tokens for similarity computation.
    """

    def __init__(self, patch_tokens=None, aggregation='mean', apply_smoothing=True, 
                 smoothing_sigma=1.5, temperature=0.1, **kwargs):
        """
        Args:
            patch_tokens: Optional pre-computed patch tokens for similarity computation
            aggregation: How to combine multiple register tokens - 'mean', 'max', or 'separate'
            apply_smoothing: Whether to apply Gaussian smoothing
            smoothing_sigma: Sigma for Gaussian smoothing
            temperature: Temperature for softmax scaling of similarities
        """
        super().__init__(**kwargs)
        self.patch_tokens = patch_tokens
        self.aggregation = aggregation
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma
        self.temperature = temperature

    def set_patch_tokens(self, patch_tokens):
        """Set patch tokens for similarity computation.
        
        Args:
            patch_tokens: torch.Tensor of shape [batch_size, 196, 384]
        """
        self.patch_tokens = patch_tokens

    def __call__(self, output, frame_sizes, confidence_thresh=None, patch_tokens=None):
        """
        Args:
            output: torch.Tensor of shape [B, 4, hidden_dim] (register tokens)
            frame_sizes: list of (width, height) for each image
            confidence_thresh: not used for register tokens
            patch_tokens: Optional torch.Tensor of shape [B, 196, hidden_dim] for similarity computation

        Returns:
            List of fol.Heatmap instances
        """
        # Use provided patch_tokens if available, otherwise fall back to stored ones
        patch_tokens_to_use = patch_tokens if patch_tokens is not None else self.patch_tokens
        
        if patch_tokens_to_use is None:
            # If no patch tokens provided, create a simple visualization based on register token norms
            return self._create_norm_based_visualization(output, frame_sizes)
        
        batch_size = output.shape[0]
        heatmaps = []
        
        for i in range(batch_size):
            # Get register tokens for this image [4, hidden_dim]
            register_tokens = output[i].detach().cpu().numpy()
            
            # Validate shape
            if register_tokens.shape[0] != 4:
                warnings.warn(f"Expected 4 register tokens, got {register_tokens.shape[0]}")
            
            # Get corresponding patch tokens [196, hidden_dim]
            if isinstance(patch_tokens_to_use, torch.Tensor):
                patch_tokens = patch_tokens_to_use[i].detach().cpu().numpy()
            else:
                patch_tokens = patch_tokens_to_use[i]
            
            # Reshape patch tokens if needed
            if len(patch_tokens.shape) == 3:  # [h, w, hidden_dim]
                h, w, hidden_dim = patch_tokens.shape
                patch_tokens = patch_tokens.reshape(-1, hidden_dim)
            else:
                num_patches = patch_tokens.shape[0]
                h = w = int(np.sqrt(num_patches))
            
            # Compute similarity between register tokens and patch tokens
            # Normalize vectors for cosine similarity
            register_norm = register_tokens / (np.linalg.norm(register_tokens, axis=1, keepdims=True) + 1e-8)
            patch_norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8)
            
            # Compute similarities [4, 196]
            similarities = register_norm @ patch_norm.T
            
            # Apply temperature scaling and softmax for sharper focus
            similarities = similarities / self.temperature
            similarities = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
            similarities = similarities / np.sum(similarities, axis=1, keepdims=True)
            
            # Reshape to spatial grid [4, 14, 14]
            similarity_maps = similarities.reshape(4, h, w)
            
            # Aggregate multiple register tokens based on strategy
            if self.aggregation == 'mean':
                # Average across all register tokens
                attention_map = np.mean(similarity_maps, axis=0)
            elif self.aggregation == 'max':
                # Take maximum activation across register tokens
                attention_map = np.max(similarity_maps, axis=0)
            elif self.aggregation == 'separate':
                # Create 4 separate channels (will handle differently)
                # For now, just use the first register token
                attention_map = similarity_maps[0]
                # TODO: Could return multiple heatmaps or combine into RGB channels
            else:
                attention_map = np.mean(similarity_maps, axis=0)
            
            # Optional smoothing
            if self.apply_smoothing:
                attention_map = gaussian_filter(attention_map, sigma=self.smoothing_sigma)
            
            # Resize to match original image dimensions
            orig_w, orig_h = frame_sizes[i]
            attention_resized = resize(
                attention_map,
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
            
            # Create heatmap
            heatmap = fol.Heatmap(
                map=attention_uint8,
                range=[0, 255]
            )
            
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def _create_norm_based_visualization(self, output, frame_sizes):
        """Fallback visualization using register token norms when patch tokens aren't available.
        
        Creates a simple uniform heatmap weighted by register token magnitudes.
        """
        batch_size = output.shape[0]
        heatmaps = []
        
        for i in range(batch_size):
            register_tokens = output[i].detach().cpu().numpy()
            
            # Compute L2 norms of register tokens
            register_norms = np.linalg.norm(register_tokens, axis=1)  # [4]
            
            # Normalize to [0, 1]
            if register_norms.max() > register_norms.min():
                register_norms = (register_norms - register_norms.min()) / (register_norms.max() - register_norms.min())
            
            # Create a simple 2x2 grid visualization of register token strengths
            norm_grid = register_norms.reshape(2, 2)
            
            # Resize to match original image dimensions
            orig_w, orig_h = frame_sizes[i]
            norm_resized = resize(
                norm_grid,
                (orig_h, orig_w),
                preserve_range=True,
                anti_aliasing=True,
                order=3  # Cubic interpolation for smoother result
            )
            
            # Apply smoothing for better visualization
            if self.apply_smoothing:
                norm_resized = gaussian_filter(norm_resized, sigma=self.smoothing_sigma * 10)  # More smoothing for 2x2 grid
            
            # Convert to uint8
            norm_uint8 = (norm_resized * 255).astype(np.uint8)
            
            heatmap = fol.Heatmap(
                map=norm_uint8,
                range=[0, 255]
            )
            
            heatmaps.append(heatmap)
        
        warnings.warn("No patch tokens provided for register token visualization. "
                     "Using simple norm-based visualization. For better results, "
                     "provide patch tokens using set_patch_tokens() method.")
        
        return heatmaps