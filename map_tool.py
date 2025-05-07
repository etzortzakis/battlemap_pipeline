# map_tool.py

from typing import Dict
import replicate
import numpy as np
from pycocotools import mask as mask_utils
import requests
from collections import OrderedDict
import os
import tempfile
from PIL import Image

def generate_map_image(prompt: str) -> str:
    """
    Generates a map image from a text prompt using Replicate (Flux Schnell) and saves it locally.

    Args:
        prompt (str): The text description of the map.
        api_token (str): Replicate API token.

    Returns:
        str: Path to the saved image file.
    """
    
    output_url = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt,
               "megapixels":"0.25",
               "output_format": "jpg",
               }
    )[0]
    image_url = output_url.url  # Extract the URL
    response = requests.get(image_url)
    output_path = "generated_map.jpg"
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return {"output_path":output_path, "image_url":image_url}


def segment_map_image(image_path: str,segmenter="ssa") -> Dict:
    """
    Segments the generated map image using Replicate's semantic-segment-anything model.

    Args:
        image_path (str): Path to the image file.
        api_key (str): Replicate API key.

    Returns:
        Dict: Dictionary containing segmentation results with binary masks.
    """
    if segmenter=="ssa":
        model_identifier="cjwbw/semantic-segment-anything:b2691db53f2d96add0051a4a98e7a3861bd21bf5972031119d344d956d2f8256"
    elif segmenter=="sam2":
        model_identifier="meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
    with open(image_path, "rb") as image_file:
        output = replicate.run(
            model_identifier,
            input={"image": image_file, "text_prompt": "obstacles"}
        )
    
    if segmenter=="ssa":
        url = output['json_out']
        out_json = requests.get(url).json()
        transformed_output = []
        for item in out_json:
            rle = item["segmentation"]
            binary_mask = mask_utils.decode(rle)
            new_item = OrderedDict()
            new_item["class_name"] = item["class_name"]
            new_item.update({
                "area": item["area"],
                "bbox": item["bbox"],
                "predicted_iou": item["predicted_iou"],
                "point_coords": item["point_coords"],
                "stability_score": item["stability_score"],
                "crop_box": item["crop_box"],
                "class_proposals": item["class_proposals"],
                "binary_mask": binary_mask,
            })

            transformed_output.append(new_item)

        return {"segments": transformed_output}
    elif segmenter=="sam2":
        masks = output['individual_masks']
        binary_masks = []
        for mask in masks:
            url = mask.url
            response = requests.get(url)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            img = Image.open(tmp_path).convert("L")
            mask = np.array(img)
            binary_mask = (mask > 127).astype(np.uint8)
            binary_masks.append(binary_mask)
        return {"segments":binary_masks}

def map_tool(prompt: str, api_key: str, segmenter: str = "ssa") -> Dict:
    """
    Main tool function to generate and segment a map based on a prompt.

    Args:
        prompt (str): Description of the game map.
        api_key (str): API key for both generation and segmentation.

    Returns:
        Dict: Combined output from generation and segmentation.
    """
    
    os.environ["REPLICATE_API_TOKEN"] = api_key

    gen_image = generate_map_image(prompt)
    segmentation = segment_map_image(gen_image["output_path"], segmenter=segmenter)
    
    return {
        "image": gen_image["image_url"],
        "segmentation": segmentation
    }