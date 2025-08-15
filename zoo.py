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

class TorchDinoV3ModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`TorchDinoV3Model`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        model_name: the DINOv3 model name to load from Hugging Face
        output_type: what to return - "dense_features", "attention_maps", "segmentation", 
                    "depth", "correspondence", or "retrieval"
        feature_format: "NCHW" or "NLC" for feature format
        feature_layers: "last", "intermediate", or "all" transformer layers
        use_mixed_precision: whether to use mixed precision for inference
        apply_smoothing: whether to apply smoothing to outputs
        smoothing_sigma: sigma for Gaussian smoothing
        use_dpt_decoder: whether to use DPT decoder for segmentation
        discretize_depth: whether to discretize depth into bins
        depth_min: minimum depth value in meters
        depth_max: maximum depth value in meters
        matching_method: feature matching method for correspondence
        matching_threshold: similarity threshold for matching
    """

    def __init__(self, d):
        super().__init__(d)

        self.model_name = self.parse_string(d, "model_name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.output_type = self.parse_string(d, "output_type", default="dense_features")
        self.feature_format = self.parse_string(d, "feature_format", default="NCHW")
        self.feature_layers = self.parse_string(d, "feature_layers", default="intermediate")
        self.use_mixed_precision = self.parse_bool(d, "use_mixed_precision", default=True)
        self.apply_smoothing = self.parse_bool(d, "apply_smoothing", default=True)
        self.smoothing_sigma = self.parse_number(d, "smoothing_sigma", default=1.0)
        
        # Segmentation-specific parameters
        self.use_dpt_decoder = self.parse_bool(d, "use_dpt_decoder", default=True)
        
        # Depth-specific parameters
        self.discretize_depth = self.parse_bool(d, "discretize_depth", default=True)
        self.depth_min = self.parse_number(d, "depth_min", default=0.001)
        self.depth_max = self.parse_number(d, "depth_max", default=100.0)
        
        # Correspondence-specific parameters
        self.matching_method = self.parse_string(d, "matching_method", default="dense")
        self.matching_threshold = self.parse_number(d, "matching_threshold", default=0.7)

class TorchDinoV3Model(fout.TorchImageModel):
    """Wrapper for DINOv3 models from Hugging Face.

    Args:
        config: a :class:`TorchDinoV3ModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)
        
        # Load the DINOv3 model and processor
        self._dino_model, self._processor = self._load_dino_model()
        
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
        """Load the DINOv3 model from Hugging Face."""
        logger.info(f"Loading DINOv3 model: {config.model_name}")
        
        # Load model and processor from Hugging Face
        model = AutoModel.from_pretrained(config.model_name)
        processor = AutoImageProcessor.from_pretrained(config.model_name)
        
        return model, processor

    def _load_dino_model(self):
        """Load and setup the DINOv3 model."""
        model, processor = self._load_model(self.config)
        
        # Ensure the model is on the correct device and in eval mode
        model = model.to(self._device)
        model.eval()
        
        return model, processor

    def _preprocess_image(self, img):
        """Preprocess a single image for DINOv3 model.
        
        Args:
            img: PIL Image, numpy array, or torch tensor
            
        Returns:
            preprocessed tensor ready for DINOv3 model
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
        
        # Use DINOv3 processor for preprocessing
        inputs = self._processor(images=img, return_tensors="pt")
        
        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self._device)
        
        return inputs

    def _predict_all(self, imgs):
        """Apply DINOv3 model to batch of images."""
        
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
        logger.debug(f"After preprocessing, first img device: {imgs[0]['pixel_values'].device if isinstance(imgs[0], dict) else 'not dict'}")

        # Process each image individually
        summaries = []
        spatial_features_list = []
        
        for i, img_inputs in enumerate(imgs):
            try:
                # Ensure inputs are on the correct device
                if isinstance(img_inputs, dict):
                    for key in img_inputs:
                        if isinstance(img_inputs[key], torch.Tensor):
                            img_inputs[key] = img_inputs[key].to(self._device)
                    logger.debug(f"Image {i} inputs device: {img_inputs['pixel_values'].device}")
                
                # Forward pass with optional mixed precision
                use_mixed_precision = (
                    getattr(self.config, 'use_mixed_precision', False) and 
                    getattr(self, '_mixed_precision_supported', False) and 
                    self._using_gpu
                )
                
                if use_mixed_precision:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        with torch.no_grad():
                            outputs = self._dino_model(**img_inputs, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        outputs = self._dino_model(**img_inputs, output_hidden_states=True)
                
                # Extract features based on output type
                output_type = getattr(self.config, 'output_type', 'dense_features')
                
                if output_type == "dense_features":
                    # For dense features, return both summary and spatial
                    # Use the last hidden state [CLS] token for global features
                    summary = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
                    
                    # Use the last hidden state without [CLS] token for spatial features
                    # Remove [CLS] token and reshape to spatial format
                    hidden_states = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                    
                elif output_type == "attention_maps":
                    # For attention maps, focus on spatial features
                    summary = None
                    hidden_states = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                    
                elif output_type == "segmentation":
                    # For segmentation, focus on spatial features
                    summary = None
                    hidden_states = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                    
                elif output_type == "depth":
                    # For depth estimation, focus on spatial features
                    summary = None
                    hidden_states = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                    
                elif output_type == "correspondence":
                    # For correspondence, return both summary and spatial
                    summary = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
                    hidden_states = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                    
                elif output_type == "retrieval":
                    # For retrieval, focus on summary features
                    summary = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
                    spatial = None
                    
                else:
                    # Default: both summary and spatial
                    hidden_states = outputs.last_hidden_state
                    summary = hidden_states[:, 0, :]  # [CLS] token
                    hidden_states_spatial = hidden_states[:, 1:, :]  # [1, num_patches, hidden_dim]
                    
                    # Get image dimensions from processor
                    if hasattr(self._processor, 'size'):
                        img_size = self._processor.size
                        if isinstance(img_size, dict):
                            img_size = img_size.get('height', 224)
                        else:
                            img_size = img_size
                    else:
                        img_size = 224
                    
                    # Calculate patch size (assuming square patches)
                    patch_size = int(math.sqrt(hidden_states_spatial.shape[1]))
                    
                    # Reshape to spatial format [1, hidden_dim, H, W]
                    spatial = hidden_states_spatial.permute(0, 2, 1).reshape(1, -1, patch_size, patch_size)
                
                logger.debug(f"Image {i} output devices - summary: {summary.device if summary is not None else 'None'}, spatial: {spatial.device if spatial is not None else 'None'}")
                
                summaries.append(summary)
                spatial_features_list.append(spatial)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                logger.error(f"Image {i} inputs: {img_inputs.keys() if isinstance(img_inputs, dict) else 'not dict'}")
                logger.error(f"Model device: {next(self._dino_model.parameters()).device}")
                raise
        
        # Stack results
        try:
            if summaries[0] is not None:
                batch_summary = torch.cat(summaries, dim=0)
            else:
                batch_summary = None
                
            if spatial_features_list[0] is not None:
                batch_spatial = torch.cat(spatial_features_list, dim=0)
            else:
                batch_spatial = None
                
        except Exception as e:
            logger.error(f"Error stacking results: {e}")
            logger.error(f"Summary devices: {[s.device if s is not None else 'None' for s in summaries]}")
            logger.error(f"Spatial devices: {[s.device if s is not None else 'None' for s in spatial_features_list]}")
            raise
        
        # Return based on output type
        output_type = getattr(self.config, 'output_type', 'dense_features')
        
        if output_type == "retrieval":
            output = batch_summary
        elif output_type in ["attention_maps", "segmentation", "depth"]:
            output = batch_spatial
        else:
            # dense_features, correspondence, or default - return as tuple
            output = (batch_summary, batch_spatial)
        
        # Process output if we have an output processor
        if self._output_processor is not None:
            # Collect original frame sizes for batch
            frame_sizes = []
            for img in imgs:
                if isinstance(img, dict) and 'pixel_values' in img:
                    # Get dimensions from processed tensor
                    h, w = img['pixel_values'].shape[-2:]
                    frame_sizes.append((w, h))  # (width, height) format
                else:
                    # Fallback
                    frame_sizes.append((224, 224))
            
            return self._output_processor(
                output, frame_sizes, confidence_thresh=self.config.confidence_thresh
            )
        
        # Return raw features as numpy arrays for embeddings
        if isinstance(output, tuple):
            # Both summary and spatial
            summary_np = [output[0][i].detach().cpu().numpy() for i in range(len(imgs))]
            spatial_np = [output[1][i].detach().cpu().numpy() for i in range(len(imgs))]
            return summary_np, spatial_np
        else:
            return [output[i].detach().cpu().numpy() for i in range(len(imgs))]

class DinoV3OutputProcessor(fout.OutputProcessor):
    """Output processor for DINOv3 models that handles embeddings output."""
    
    def __init__(self, output_type="dense_features", **kwargs):
        super().__init__(**kwargs)
        self.output_type = output_type
        
    def __call__(self, output, frame_size, confidence_thresh=None):
        """Process DINOv3 model output into embeddings.
        
        Args:
            output: tensor from DINOv3 model
            frame_size: (width, height) - not used for embeddings
            confidence_thresh: not used for embeddings
            
        Returns:
            list of numpy arrays containing embeddings
        """
        if isinstance(output, tuple):
            # Both summary and spatial
            batch_size = output[0].shape[0]
            summary_embeddings = [output[0][i].detach().cpu().numpy() for i in range(batch_size)]
            spatial_embeddings = [output[1][i].detach().cpu().numpy() for i in range(batch_size)]
            return summary_embeddings, spatial_embeddings
        else:
            batch_size = output.shape[0]
            return [output[i].detach().cpu().numpy() for i in range(batch_size)]

class SpatialHeatmapOutputProcessor(fout.OutputProcessor):
    """Improved spatial heatmap processor for DINOv3 with NCHW features and smoothing."""

    def __init__(self, apply_smoothing=True, smoothing_sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.apply_smoothing = apply_smoothing
        self.smoothing_sigma = smoothing_sigma

    def __call__(self, output, frame_sizes, confidence_thresh=None):
        """
        Args:
            output: torch.Tensor of shape [B, C, H, W] or tuple (summary, spatial)
            frame_sizes: list of (width, height) for each image
            confidence_thresh: unused

        Returns:
            List of fol.Heatmap instances
        """
        # Handle both single output and tuple output
        if isinstance(output, tuple):
            spatial = output[1]  # Use spatial features from tuple
        else:
            spatial = output
            
        batch_size = spatial.shape[0]
        heatmaps = []

        for i in range(batch_size):
            spatial_feat = spatial[i].detach().cpu().numpy()  # [C, H, W]
            C, H, W = spatial_feat.shape

            # Flatten spatial grid to [H*W, C] for PCA
            reshaped = spatial_feat.reshape(C, -1).T  # [H*W, C]

            try:
                # PCA to reduce channels to 1D attention per pixel
                pca = PCA(n_components=1)
                attention_1d = pca.fit_transform(reshaped).reshape(H, W)
            except Exception as e:
                # Fallback to simple mean over channels
                warnings.warn(f"PCA failed on image {i}: {e}. Falling back to channel mean.")
                attention_1d = spatial_feat.mean(axis=0)  # [H, W]

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

            heatmap = fol.Heatmap(
                map=attention_uint8,
                range=[0, 255]
            )
            heatmaps.append(heatmap)

        return heatmaps
