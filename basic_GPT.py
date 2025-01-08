import base64
import numpy as np
import os
import json

from io import BytesIO
from openai import OpenAI
from PIL import Image


dir_node = os.path.dirname(__file__)
config_path = os.path.join(os.path.abspath(dir_node), "config.json")

print(config_path)

with open(config_path, "r") as f:
    CONFIG = json.load(f)

    os.environ["OPENAI_API_KEY"] = CONFIG.get("OPENAI_KEY")

client = OpenAI()


class BasicGPT:
    def __init__(self): 
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_system_prompt": ("STRING", {
                        "multiline": True,
                        "default": "",
                        "lazy": True
                    }
                ),
                "input_prompt": ("STRING", {
                        "multiline": True,
                        "default": "",
                        "lazy": True
                    }
                ),
            },
             "optional": {
  
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "send_gpt_request"

    OUTPUT_NODE = True

    CATEGORY = "GPT"

    def encode_image(self,images):

        for image in images:
            i = 255. * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def send_gpt_request(self, input_system_prompt, input_prompt, image=None):
        # System prompt for the bot
        system_prompt = input_system_prompt

        # User prompt for image analysis and crafting a FLUX dev format prompt
        prompt = input_prompt

        if image != None:
            base64_image = self.encode_image(image)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "system",
                    "content": system_prompt,
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt,
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{base64_image}"
                        },
                        },
                    ],
                    }
                ],
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "system",
                    "content": system_prompt,
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt,
                        }
                    ],
                    }
                ],
            )

        # generated_prompt = response.choices[0].message['content']
        generated_prompt = response.choices[0].message.content
        return (generated_prompt,)






