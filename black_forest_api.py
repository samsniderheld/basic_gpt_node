import base64
import json
import numpy as np
import os
import requests
import time
import torch

from io import BytesIO
from PIL import Image

dir_node = os.path.dirname(__file__)
config_path = os.path.join(os.path.abspath(dir_node), "config.json")

print(config_path)

with open(config_path, "r") as f:
    CONFIG = json.load(f)

class BlackForestAPI:
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
                "model": (["txt2img", "redux", "inpainting", "canny", "depth"],),
                "width": ("INT", {
                    "default": 1024, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 10000000, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                
                
            },
             "optional": {
  
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "image_prompt_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0, #Minimum value
                    "max": 1, #Maximum value
                    "step": .1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
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
                    "step": 1, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "prompt_upsampling": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "send_api_request"

    OUTPUT_NODE = True

    CATEGORY = "Flux API"

    def encode_image(self,images):

        for image in images:
            i = 255. * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
    def download_image(self, id):
        while True:
            time.sleep(0.5)
            result = requests.get(
                'https://api.bfl.ml/v1/get_result',
                headers={
                    'accept': 'application/json',
                    'x-key': CONFIG.get("BLACKFOREST_KEY"),
                },
                params={
                    'id': id,
                },
            ).json()

            print(result)

            if result["status"] == "Ready":
                img_req = requests.get(result['result']['sample'], stream=True)
                if img_req.status_code == 200:
                    pil_image = Image.open(BytesIO(img_req.content)).convert('RGB')
                    torch_tensor = self.pil2tensor(pil_image)
                    return torch_tensor
                break
            else:
                print(f"Status: {result['status']}")

    def send_api_request(self, prompt, model, width=1024, height=1024,seed=0, image=None, mask=None, image_prompt_strength=1.0,steps=20,guidance=.5,prompt_upsampling=False):

        model_urls = {
            "txt2img": "https://api.bfl.ml/v1/flux-pro-1.1",
            "redux": "https://api.bfl.ml/v1/flux-pro-1.1",
            "inpainting": "https://api.bfl.ml/v1/flux-pro-1.0-fill",
            "canny": "https://api.bfl.ml/v1/flux-pro-1.0-canny",
            "depth": "https://api.bfl.ml/v1/flux-pro-1.0-depth",
        }

        url = model_urls.get(model)

        headers = {
            "Content-Type": "application/json",
            "X-Key": CONFIG.get("BLACKFOREST_KEY")
        }

        data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "safety_tolerance": 2,
            "output_format": "jpeg",
            "guidance": guidance,
            "steps": steps
        }
        
        if image != None:
            base64_image = self.encode_image(image)

            match model:
                case "txt2img":
                    data["image_prompt"] = base64_image
                    # data["image_prompt_strength"] = image_prompt_strength
                case "inpainting":
                    data["image"] = base64_image
                    data["image_prompt_strength"] = image_prompt_strength

                case "canny":
                    data["preprocessed_image"] = base64_image

                case "depth":
                    data["preprocessed_image"] = base64_image
        
        if mask != None:
            
            base64_mask = self.encode_image(mask)
            data["mask"] = base64_mask


        response = requests.post(url, headers=headers, json=data)

        request_json = response.json()

        try:
            request_id = request_json["id"]
            output_img = self.download_image(request_id)
        except:
            print(request_json["detail"][0]["msg"])
            print(request_json["detail"][0]["loc"])
            return [request_json["detail"][0]["msg"],request_json["detail"][0]["loc"]]
        

        return (output_img,)




