import logging
import os

from huggingface_hub import snapshot_download

from fiftyone.operators import types

# Import constants from zoo.py to ensure consistency
from .zoo import TorchDinoV3ModelConfig, TorchDinoV3Model, DinoV3OutputProcessor, SpatialHeatmapOutputProcessor

logger = logging.getLogger(__name__)

# Model variants and their configurations
MODEL_VARIANTS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "description": "DINOv3 ViT-S/16 model (21M parameters) - Small vision transformer for fast inference",
        "hidden_dim": 384,
        "architecture": "ViT-S/16",
        "params": "21M"
    },
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "description": "DINOv3 ViT-S+/16 model (29M parameters) - Small+ vision transformer with SwiGLU FFN",
        "hidden_dim": 384,
        "architecture": "ViT-S+/16",
        "params": "29M"
    },
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "description": "DINOv3 ViT-B/16 model (86M parameters) - Base vision transformer, balanced performance",
        "hidden_dim": 768,
        "architecture": "ViT-B/16",
        "params": "86M"
    },
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "description": "DINOv3 ViT-L/16 model (300M parameters) - Large vision transformer for high accuracy",
        "hidden_dim": 1024,
        "architecture": "ViT-L/16",
        "params": "300M"
    },
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "description": "DINOv3 ViT-H+/16 model (840M parameters) - Huge+ vision transformer with SwiGLU FFN",
        "hidden_dim": 1280,
        "architecture": "ViT-H+/16",
        "params": "840M"
    },
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "description": "DINOv3 ViT-7B/16 model (7.16B parameters) - Giant vision transformer for maximum performance",
        "hidden_dim": 4096,
        "architecture": "ViT-7B/16",
        "params": "7.16B"
    },
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "description": "DINOv3 ConvNeXt-Tiny model (29M parameters) - Tiny ConvNeXt for fast inference",
        "hidden_dim": 768,
        "architecture": "ConvNeXt-Tiny",
        "params": "29M"
    },
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "description": "DINOv3 ConvNeXt-Small model (50M parameters) - Small ConvNeXt for balanced performance",
        "hidden_dim": 768,
        "architecture": "ConvNeXt-Small",
        "params": "50M"
    },
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "description": "DINOv3 ConvNeXt-Base model (89M parameters) - Base ConvNeXt for good performance",
        "hidden_dim": 1024,
        "architecture": "ConvNeXt-Base",
        "params": "89M"
    },
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "description": "DINOv3 ConvNeXt-Large model (198M parameters) - Large ConvNeXt for high accuracy",
        "hidden_dim": 1536,
        "architecture": "ConvNeXt-Large",
        "params": "198M"
    },
    "facebook/dinov3-vitl16-pretrain-sat493m": {
        "model_name": "facebook/dinov3-vitl16-pretrain-sat493m",
        "description": "DINOv3 ViT-L/16 model (300M parameters) - Large vision transformer trained on satellite data",
        "hidden_dim": 1024,
        "architecture": "ViT-L/16-Satellite",
        "params": "300M"
    },
    "facebook/dinov3-vit7b16-pretrain-sat493m": {
        "model_name": "facebook/dinov3-vit7b16-pretrain-sat493m",
        "description": "DINOv3 ViT-7B/16 model (7.16B parameters) - Giant vision transformer trained on satellite data",
        "hidden_dim": 4096,
        "architecture": "ViT-7B/16-Satellite",
        "params": "7.16B"
    }
}

def download_model(model_name, model_path, **kwargs):
    """Downloads the model.
    
    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    logger.info(f"Downloading DINOv3 model {model_name} from Hugging Face...")
    
    try:
        snapshot_download(repo_id=model_name, local_dir=model_path)
        logger.info(f"✅ DINOv3 model {model_name} downloaded to {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to download DINOv3 model {model_name}: {e}")
        raise


def load_model(model_name, model_path, **kwargs):
    """Loads the model.
    
    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            downloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded
            
    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model(...)"
        )
    
    print(f"Loading DINOv3 model {model_name} from {model_path}")
    
    # Set default parameters if not provided
    default_params = {
        "output_type": "dense_features",
        "feature_format": "NCHW",
        "feature_layers": "intermediate",
        "use_mixed_precision": True,
        "raw_inputs": True,  # We handle preprocessing ourselves
    }
    
    # Update with provided kwargs
    default_params.update(kwargs)
    
    # Set up output processor based on output type
    output_type = default_params["output_type"]
    
    if output_type == "dense_features":
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DinoV3OutputProcessor
        default_params["output_processor_args"] = {"output_type": output_type}
        
    elif output_type == "attention_maps":
        default_params["output_processor_cls"] = SpatialHeatmapOutputProcessor
        default_params["output_processor_args"] = {
            "apply_smoothing": kwargs.get("apply_smoothing", True),
            "smoothing_sigma": kwargs.get("smoothing_sigma", 1.0),
        }
        
    elif output_type == "segmentation":
        default_params["output_processor_cls"] = SpatialHeatmapOutputProcessor
        default_params["output_processor_args"] = {
            "apply_smoothing": kwargs.get("apply_smoothing", True),
            "smoothing_sigma": kwargs.get("smoothing_sigma", 1.0),
            "use_dpt_decoder": kwargs.get("use_dpt_decoder", True),
        }
        
    elif output_type == "depth":
        default_params["output_processor_cls"] = SpatialHeatmapOutputProcessor
        default_params["output_processor_args"] = {
            "discretize_depth": kwargs.get("discretize_depth", True),
            "depth_min": kwargs.get("depth_min", 0.001),
            "depth_max": kwargs.get("depth_max", 100.0),
        }
        
    elif output_type == "correspondence":
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DinoV3OutputProcessor
        default_params["output_processor_args"] = {
            "output_type": output_type,
            "matching_method": kwargs.get("matching_method", "dense"),
            "matching_threshold": kwargs.get("matching_threshold", 0.7),
        }
        
    elif output_type == "retrieval":
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DinoV3OutputProcessor
        default_params["output_processor_args"] = {"output_type": output_type}
        
    else:
        raise ValueError(f"Unsupported output_type: {output_type}. Use 'dense_features', 'attention_maps', 'segmentation', 'depth', 'correspondence', or 'retrieval'")
    
    # Create and return the model
    config = TorchDinoV3ModelConfig(default_params)
    return TorchDinoV3Model(config)


def resolve_input(model_name, ctx):
    """Defines any necessary properties to collect the model's custom
    parameters from a user during prompting.
    
    Args:
        model_name: the name of the model, as declared by the ``base_name`` and
            optional ``version`` fields of the manifest
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        
    Returns:
        a :class:`fiftyone.operators.types.Property`, or None
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    inputs = types.Object()
    
    # Core DINOv3 output types (what the model actually produces)
    inputs.enum(
        "output_type",
        ["dense_features", "attention_maps", "segmentation", "depth", "correspondence", "retrieval"],
        default="dense_features",
        label="Output Type",
        description="Type of DINOv3 output: dense_features (multi-scale), attention_maps (self-attention), segmentation (object discovery), depth (3D), correspondence (matching), retrieval (similarity)"
    )
    
    # Feature extraction options
    inputs.enum(
        "feature_layers", 
        ["last", "intermediate", "all"],
        default="intermediate",
        label="Feature Layers",
        description="Which transformer layers to extract features from: last (final), intermediate [10,20,30,40], or all layers"
    )
    
    inputs.enum(
        "feature_format", 
        ["NCHW", "NLC"],
        default="NCHW",
        label="Feature Format",
        description="Format for features: NCHW (computer vision) or NLC (transformer sequence)"
    )
    
    # Performance options
    inputs.bool(
        "use_mixed_precision",
        default=True,
        label="Use Mixed Precision",
        description="Whether to use mixed precision for faster inference on supported GPUs"
    )

    # Segmentation options (when output_type="segmentation")
    inputs.bool(
        "use_dpt_decoder",
        default=True,
        label="Use DPT Decoder",
        description="Whether to use DPT decoder for segmentation (recommended)"
    )
    
    inputs.bool(
        "apply_smoothing",
        default=True,
        label="Apply Smoothing",
        description="Whether to smooth segmentation masks for better visualization"
    )

    inputs.float(
        "smoothing_sigma",
        default=1.0,
        label="Smoothing Sigma",
        description="The standard deviation (sigma) for Gaussian smoothing of masks"
    )

    # Depth estimation options (when output_type="depth")
    inputs.bool(
        "discretize_depth",
        default=True,
        label="Discretize Depth",
        description="Whether to discretize depth into 256 bins (0.001m to 100m)"
    )
    
    inputs.float(
        "depth_min",
        default=0.001,
        label="Minimum Depth",
        description="Minimum depth value in meters"
    )
    
    inputs.float(
        "depth_max",
        default=100.0,
        label="Maximum Depth",
        description="Maximum depth value in meters"
    )

    # Correspondence options (when output_type="correspondence")
    inputs.enum(
        "matching_method",
        ["dense", "sparse", "hybrid"],
        default="dense",
        label="Matching Method",
        description="Feature matching method: dense (all patches), sparse (keypoints), or hybrid"
    )
    
    inputs.float(
        "matching_threshold",
        default=0.7,
        label="Matching Threshold",
        description="Similarity threshold for feature matching (0.0 to 1.0)"
    )

    # Advanced options
    inputs.bool(
        "use_pipeline",
        default=False,
        label="Use Hugging Face Pipeline",
        description="Whether to use Hugging Face pipeline for simpler inference (may be slower)"
    )

    inputs.bool(
        "device_map_auto",
        default=True,
        label="Auto Device Mapping",
        description="Whether to automatically map model to available devices (recommended)"
    )

    return types.Property(inputs)


def parse_parameters(model_name, ctx, params):
    """Performs any execution-time formatting to the model's custom parameters.
    
    Args:
        model_name: the name of the model, as declared by the ``base_name`` and
            optional ``version`` fields of the manifest
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        params: a params dict
    """
    # Ensure output_type is valid
    output_type = params.get("output_type", "dense_features")
    if output_type not in ["dense_features", "attention_maps", "segmentation", "depth", "correspondence", "retrieval"]:
        params["output_type"] = "dense_features"
    
    # Ensure feature_format is valid
    feature_format = params.get("feature_format", "NCHW")
    if feature_format not in ["NCHW", "NLC"]:
        params["feature_format"] = "NCHW"
    
    # Ensure feature_layers is valid
    feature_layers = params.get("feature_layers", "intermediate")
    if feature_layers not in ["last", "intermediate", "all"]:
        params["feature_layers"] = "intermediate"
    
    # Ensure smoothing_sigma is positive
    smoothing_sigma = params.get("smoothing_sigma", 1.0)
    if smoothing_sigma <= 0:
        params["smoothing_sigma"] = 1.0
    
    # Ensure use_mixed_precision is boolean
    use_mixed_precision = params.get("use_mixed_precision", True)
    params["use_mixed_precision"] = bool(use_mixed_precision)
    
    # Ensure apply_smoothing is boolean
    apply_smoothing = params.get("apply_smoothing", True)
    params["apply_smoothing"] = bool(apply_smoothing)
    
    # Ensure use_pipeline is boolean
    use_pipeline = params.get("use_pipeline", False)
    params["use_pipeline"] = bool(use_pipeline)
    
    # Ensure device_map_auto is boolean
    device_map_auto = params.get("device_map_auto", True)
    params["device_map_auto"] = bool(device_map_auto)
    
    # Segmentation-specific parameters
    if output_type == "segmentation":
        use_dpt_decoder = params.get("use_dpt_decoder", True)
        params["use_dpt_decoder"] = bool(use_dpt_decoder)
    
    # Depth-specific parameters
    if output_type == "depth":
        discretize_depth = params.get("discretize_depth", True)
        params["discretize_depth"] = bool(discretize_depth)
        
        depth_min = params.get("depth_min", 0.001)
        if depth_min <= 0:
            params["depth_min"] = 0.001
        
        depth_max = params.get("depth_max", 100.0)
        if depth_max <= depth_min:
            params["depth_max"] = 100.0
    
    # Correspondence-specific parameters
    if output_type == "correspondence":
        matching_method = params.get("matching_method", "dense")
        if matching_method not in ["dense", "sparse", "hybrid"]:
            params["matching_method"] = "dense"
        
        matching_threshold = params.get("matching_threshold", 0.7)
        if matching_threshold < 0.0 or matching_threshold > 1.0:
            params["matching_threshold"] = 0.7
    
    # Log the final parameters for debugging
    logger.debug(f"Parsed parameters for {model_name}: {params}")


def get_model_info(model_name):
    """Get information about a specific model.
    
    Args:
        model_name: the name of the model
        
    Returns:
        dict: model information
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'. "
                        f"Supported models: {list(MODEL_VARIANTS.keys())}")
    
    return MODEL_VARIANTS[model_name]


def list_available_models():
    """List all available DINOv3 models.
    
    Returns:
        list: list of available model names
    """
    return list(MODEL_VARIANTS.keys())


def get_model_architecture_info(model_name):
    """Get detailed architecture information for a model.
    
    Args:
        model_name: the name of the model
        
    Returns:
        dict: architecture information
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model name '{model_name}'")
    
    model_info = MODEL_VARIANTS[model_name]
    
    # Add additional architecture details
    arch_info = {
        "model_name": model_info["model_name"],
        "architecture": model_info["architecture"],
        "parameters": model_info["params"],
        "hidden_dimensions": model_info["hidden_dim"],
        "description": model_info["description"],
        "supported_outputs": ["summary", "spatial", "both"],
        "supported_formats": ["NCHW", "NLC"],
        "patch_size": 16,  # All DINOv3 models use 16x16 patches
        "input_resolution": "224x224 (minimum), multiples of 16",
        "license": "DINOv3 License",
        "source": "https://github.com/facebookresearch/dinov3"
    }
    
    return arch_info