import os
import numpy as np
import torch
from PIL import Image
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_MODEL = None
_PROCESSOR = None

def init_model():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    print("Loading SAM3 Model (This happens only once)...")
    sam3_root = os.path.join(os.getcwd(), 'sam3')
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    _MODEL = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cuda",
        eval_mode=True,
        checkpoint_path="./sam3/models/sam3.pt",
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        compile=False,
    )
    _PROCESSOR = Sam3Processor(_MODEL)
    print("SAM3 Model Loaded Successfully.")
    return _MODEL, _PROCESSOR


def keep_largest_component(mask_bool):
    """
        Keep only the largest connected component, and filter out other noise.
        Internal holes will be preserved and not filled.
    """
    # 1. uint8 for OpenCV
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    
    # 2. Connected Component Analysis
    # connectivity=8 means the connectivity check includes diagonal directions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels < 2:
        return mask_bool 

    # 3. Find the largest connected component
    # stats[:, 4] is the area column.
    # We need to ignore index 0 (background) and find the index of the largest value starting from index 1
    # np.argmax returns the index relative to the slice stats[1:], so we need to add 1 at the end
    largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 4. new Mask
    filtered_mask = (labels == largest_label_idx)
    
    return filtered_mask

def seg_single_scene(image_pil, input_point, input_label, prompt=''):
    model, processor = init_model()

    inference_state = processor.set_image(image_pil)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mask_output, _, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

    mask_tensor = mask_output[0] 
    
    if isinstance(mask_tensor, torch.Tensor):
        mask_numpy = mask_tensor.detach().cpu().numpy()
    else:
        mask_numpy = mask_tensor
    
    mask_bool = mask_numpy.astype(bool)

    final_mask = keep_largest_component(mask_bool)

    return final_mask