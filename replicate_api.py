import json
import numpy as np
import os
import replicate
import requests
import torch

from io import BytesIO
from PIL import Image

dir_node = os.path.dirname(__file__)
config_path = os.path.join(os.path.abspath(dir_node), "config.json")

with open(config_path, "r") as f:
    CONFIG = json.load(f)

os.environ["REPLICATE_API_TOKEN"] = CONFIG.get("REPLICATE_KEY")

class ReplicateAPI:
    def __init__(self): 
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                        "multiline": True,
                        "default": "",
                        "lazy": True
                    }
                ),
                "lora_model": ("STRING", {
                        "multiline": False,
                        "default": "fofr/flux-80s-cyberpunk",
                        "lazy": True
                    }
                ),
            },
             "optional": {
                "steps": ("INT", {
                    "default": 20, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "guidance": ("FLOAT", {
                    "default": 5.5, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": .5, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0, #Minimum value
                    "max": 3, #Maximum value
                    "step": .5, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "go_fast": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Enable", 
                    "label_off": "Disable"}),
                "aspect_ratio": (["1:1", "16:9","21:9","3:2","4:5","5:4","2:3","9:21","9:16",],),

            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "send_api_request"

    OUTPUT_NODE = True

    CATEGORY = "Replicate API"

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
    def download_image(self, url):
        img_req = requests.get(url, stream=True)
        
        if img_req.status_code == 200:
            pil_image = Image.open(BytesIO(img_req.content)).convert('RGB')
            torch_tensor = self.pil2tensor(pil_image)
            return torch_tensor


    def send_api_request(self, prompt, lora_model, lora_scale=1.0, steps=20,guidance=.5, go_fast=False,aspect_ratio="1:1"):

        output = replicate.run(
            "black-forest-labs/flux-dev-lora",
            input={
                "prompt": prompt,
                "go_fast": go_fast,
                "guidance": guidance,
                "lora_scale": lora_scale,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": aspect_ratio,
                "lora_weights": lora_model,
                "output_format": "jpg",
                "output_quality": 80,
                "prompt_strength": 0.8,
                "num_inference_steps": steps
            }
        )

        try:
            output_img = self.download_image(output[0])
        except:
            return
        
        return (output_img,)




