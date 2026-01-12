import os
import torch
import clip
from PIL import Image
from torchvision import transforms


def get_model(model_name, device, method='base', root_dir='data'):
    """
    Helper function that returns a model and a potential image preprocessing function.
    
    Args:
        model_name: Name of the model ("llava1.5", "llava1.6", or CLIP variants)
        device: Device to load the model on
        method: Experimental method (determines which model variant to use)
        root_dir: Root directory for model cache
    
    Returns:
        model: The model wrapper
        image_preprocess: Image preprocessing function (None for LLaVA models)
    """
    
    # ========================================================================
    # CLIP Models
    # ========================================================================
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(
            variant, 
            device=device, 
            download_root=root_dir
        )
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess
    
    elif 'spatial_ft' in model_name:
        from .clip_models import CLIPWrapper
        variant = 'ViT-B/32'
        model, image_preprocess = clip.load(
            variant, 
            device=device, 
            download_root=root_dir
        )
        model_path = f'data/{model_name}.pt'
        print(f'Loading fine-tuned weights from {model_path}')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                f"Please download the model weights first."
            )
        
        state = torch.load(model_path)
        state['model_state_dict'] = {
            k.replace('module.clip_model.', ''): v 
            for k, v in state['model_state_dict'].items()
        }
        model.load_state_dict(state['model_state_dict'])
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess
    
    # ========================================================================
    # LLaVA Models
    # ========================================================================
    elif model_name == "llava1.5":
        # Import the unified wrapper
        try:
            from .llava_wrapper_unified import LlavaWrapper
        except ImportError:
            # Fallback to original import structure
            try:
                from .llava15 import LlavaWrapper
            except ImportError:
                raise ImportError(
                    "Could not import LlavaWrapper. Please ensure either:\n"
                    "1. llava_wrapper_unified.py is in the model_zoo directory, or\n"
                    "2. The original llava15.py file exists in the model_zoo directory"
                )
        
        print(f"Loading LLaVA 1.5 model with method: {method}")
        llava_model = LlavaWrapper(
            root_dir=root_dir, 
            device=device, 
            method=method
        )
        image_preprocess = None
        
        # Print model type for debugging
        model_type = "LlavaForConditionalGenerationScal" if "Scal" in str(type(llava_model.model)) else "LlavaForConditionalGeneration"
        print(f"Loaded model type: {model_type}")
        
        return llava_model, image_preprocess
    
    elif model_name == "llava1.6":
        # Import the unified wrapper
        try:
            from .llava_wrapper_unified import LlavaWrapper
        except ImportError:
            # Fallback to original import structure
            try:
                from .llava16 import LlavaWrapper
            except ImportError:
                raise ImportError(
                    "Could not import LlavaWrapper. Please ensure either:\n"
                    "1. llava_wrapper_unified.py is in the model_zoo directory, or\n"
                    "2. The original llava16.py file exists in the model_zoo directory"
                )
        
        print(f"Loading LLaVA 1.6 model with method: {method}")
        llava_model = LlavaWrapper(
            root_dir=root_dir, 
            device=device, 
            method=method
        )
        image_preprocess = None
        
        # Print model type for debugging
        model_type = "LlavaForConditionalGenerationScal" if "Scal" in str(type(llava_model.model)) else "LlavaForConditionalGeneration"
        print(f"Loaded model type: {model_type}")
        
        return llava_model, image_preprocess
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: llava1.5, llava1.6, openai-clip:<variant>, spatial_ft"
        )


def print_model_info(model_name, method):
    """Print information about the model and method being used"""
    
    methods_need_scal = [
        'scaling_vis', 'adapt_vis', 'adapt_vis_jsd',
        'adapt_vis_entropy', 'adapt_vis_obj', 'adapt_vis_for_oracle_research'
    ]
    
    print("\n" + "="*60)
    print("Model Configuration")
    print("="*60)
    print(f"Model:  {model_name}")
    print(f"Method: {method}")
    
    if method in methods_need_scal:
        print(f"Type:   Requires visual scaling (Scal model)")
    else:
        print(f"Type:   Standard generation")
    
    print("="*60 + "\n")
