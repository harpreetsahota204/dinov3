import logging
import os
import torch

from huggingface_hub import snapshot_download

from fiftyone.operators import types

from .zoo import DINOV3ModelConfig, DINOV3Model, DINOV3OutputProcessor, DINOV3PatchOutputProcessor, DINOV3RegisterOutputProcessor

logger = logging.getLogger(__name__)

# Model variants and their configurations
MODEL_VARIANTS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
    },
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    },
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    },
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    },
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    },
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    },
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    },
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    },
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    },
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": {
        "model_name": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    },
    "facebook/dinov3-vitl16-pretrain-sat493m": {
        "model_name": "facebook/dinov3-vitl16-pretrain-sat493m",
    },
    "facebook/dinov3-vit7b16-pretrain-sat493m": {
        "model_name": "facebook/dinov3-vit7b16-pretrain-sat493m",
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


def load_model(
        model_name, 
        model_path, 
        output_type="cls",
        **kwargs
        ):
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
        "model_version": model_name,
        "model_path": model_path,  # Add the model path for loading from disk
        "output_type": output_type,
        **kwargs
    }
    
    # Update with provided kwargs
    default_params.update(kwargs)
    
    if output_type == "cls":
        # CLS token - global image representation (position 0)
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DINOV3OutputProcessor
        default_params["output_processor_args"] = {"output_type": output_type}
        
    elif output_type == "attention_map":
        # Register tokens - memory slots (positions 1 to 1+num_register_tokens)
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DINOV3RegisterOutputProcessor
        default_params["output_processor_args"] = {
            "apply_smoothing": kwargs.get("apply_smoothing", True),
            "smoothing_sigma": kwargs.get("smoothing_sigma", 1.51),
             "aggregation":'mean',
        }
        
    elif output_type == "patch":
        # Patch tokens - local embeddings for each 16x16 patch (remaining positions)
        default_params["as_feature_extractor"] = True
        default_params["output_processor_cls"] = DINOV3PatchOutputProcessor
        default_params["output_processor_args"] = {
            "apply_smoothing": kwargs.get("apply_smoothing", True),
            "smoothing_sigma": kwargs.get("smoothing_sigma", 1.51),
        }
        
    else:
        raise ValueError(f"Unsupported output_type: {output_type}. Use 'cls', 'register', or 'patch' (the three token types).")

    
    # Create and return the model
    config = DINOV3ModelConfig(default_params)
    return DINOV3Model(config)


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
    pass

def parse_parameters(model_name, ctx, params):
    """Performs any execution-time formatting to the model's custom parameters.
    
    Args:
        model_name: the name of the model, as declared by the ``base_name`` and
            optional ``version`` fields of the manifest
        ctx: an :class:`fiftyone.operators.ExecutionContext`
        params: a params dict
    """
    pass

