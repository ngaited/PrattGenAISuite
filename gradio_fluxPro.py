import time
import os, io
import requests
from datetime import datetime
import json
import gradio as gr
from gradio.themes import Ocean
from PIL import Image
from io import BytesIO
import base64
import yaml
import random
from google import genai
from google.genai import types
from typing import List, Union
from openai import OpenAI

# Content loader function
def load_content(filename):
    """Load markdown content from files"""
    try:
        content_path = os.path.join("content", filename)
        with open(content_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Content file {filename} not found")
        return "Content not available."
    except Exception as e:
        print(f"Error loading content from {filename}: {e}")
        return "Error loading content."

# Load configuration
with io.open('.config.yml','r') as stream:
    db = yaml.safe_load(stream)

BFL_API_KEY = db['BFL_API_KEY']
GOOGLE_API_KEY = db.get('GOOGLE_API_KEY')
OPENAI_API_KEY = db.get('OPENAI_API_KEY')

# Initialize Google Gemini client
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
# Logging Functionality
def create_log_directory():
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join('./logs', current_date)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_generation(prompt, image, seed, model_type="flux"):
    log_dir = create_log_directory()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save image
    image_filename = f"{timestamp}_{model_type}_seed_{seed}.png"
    image_path = os.path.join(log_dir, image_filename)
    image.save(image_path)
    
    # Log prompt and metadata
    log_data = {
        "timestamp": timestamp,
        "prompt": prompt,
        "seed": seed,
        "model_type": model_type,
        "image_filename": image_filename
    }
    
    log_filename = f"{timestamp}_{model_type}_metadata.json"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

def _coerce_to_pil_list(input_images: Union[None, Image.Image, List]) -> List[Image.Image]:
    """Normalize various gradio inputs (single image, list, file paths/objects) to a list of PIL Images."""
    if input_images is None:
        return []
    if isinstance(input_images, Image.Image):
        return [input_images]

    pil_list = []
    for item in input_images:
        try:
            if isinstance(item, Image.Image):
                img = item
            elif isinstance(item, bytes):
                img = Image.open(BytesIO(item))
            elif isinstance(item, str):
                img = Image.open(item)  # path
            elif hasattr(item, "name"):  # gr.Files file object
                img = Image.open(item.name)
            else:
                # Try numpy array or dict with 'name'/'path'
                try:
                    from PIL import Image as PILImage
                    import numpy as np
                    if "name" in getattr(item, "__dict__", {}) or (isinstance(item, dict) and item.get("name")):
                        path = item.get("name") if isinstance(item, dict) else item.name
                        img = Image.open(path)
                    elif "path" in getattr(item, "__dict__", {}) or (isinstance(item, dict) and item.get("path")):
                        path = item.get("path") if isinstance(item, dict) else item.path
                        img = Image.open(path)
                    elif isinstance(item, np.ndarray):
                        img = Image.fromarray(item)
                    else:
                        continue
                except Exception:
                    continue

            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_list.append(img)
        except Exception:
            continue
    return pil_list

# Google Gemini Image Generation Function (Updated)
def generate_image_gemini(prompt, seed, input_images=None):
    try:
        if not GOOGLE_API_KEY:
            print("Google API key not found in config")
            return None

        seed = int(seed)
        images = _coerce_to_pil_list(input_images)
        print(f"Gemini request: prompt only" if not images else f"Gemini request: prompt + {len(images)} image(s)")

        contents = [prompt, *images]  # multiple images supported

        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
        )

        # Find first image in response parts
        for part in getattr(response.candidates[0].content, "parts", []):
            if getattr(part, "inline_data", None) is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                gen_type = "gemini-txt2img" if not images else ("gemini-img2img-multi" if len(images) > 1 else "gemini-img2img")
                log_generation(prompt, generated_image, seed, gen_type)
                return generated_image
            if getattr(part, "text", None):
                print(f"Gemini text: {part.text}")

        print("No image data found in Gemini response")
        return None

    except Exception as e:
        print(f"Error in generate_image_gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_image_openai(prompt, seed, input_images=None):
    try:
        if not OPENAI_API_KEY:
            print("OpenAI API key not found in config")
            return None

        seed = int(seed)  # Not used by OpenAI, but kept for consistency
        images = _coerce_to_pil_list(input_images)
        print(f"OpenAI request: prompt only" if not images else f"OpenAI request: prompt + {len(images)} image(s)")

        if not images:
            # Text-to-image generation
            response = openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt
                # Note: No size, quality, style, or response_format parameters for gpt-image-1
            )
            
            # The response contains b64_json directly
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            generated_image = Image.open(BytesIO(image_bytes))
            
            gen_type = "openai-txt2img"
            
        else:
            # Image editing with reference images
            # Convert PIL images to file-like objects for OpenAI
            image_files = []
            for i, img in enumerate(images):
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                image_files.append(buffered)

            response = openai_client.images.edit(
                model="gpt-image-1",
                image=image_files,  # List of file-like objects
                prompt=prompt
                # Note: No additional parameters for gpt-image-1
            )
            
            # The response contains b64_json directly
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            generated_image = Image.open(BytesIO(image_bytes))
            
            gen_type = "openai-img2img-multi" if len(images) > 1 else "openai-img2img"

        log_generation(prompt, generated_image, seed, gen_type)
        return generated_image

    except Exception as e:
        print(f"Error in generate_image_openai: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# Generate Function via API
def generate_image(prompt, seed):
    seed = int(seed)

    post_endpoint = 'https://api.us1.bfl.ai/v1/flux-pro-1.1-ultra'
    headers = {
        'accept': 'application/json',
        'x-key': BFL_API_KEY,
        'Content-Type': 'application/json',
    }
    request_data = {
        'prompt': prompt,
        'seed':seed,
        "aspect_ratio": "16:9",
        
        # 'width': 1024,
        # 'height': 768,
    }
    
    try:
        response = requests.post(post_endpoint, headers=headers, json=request_data).json()
        request_id = response["id"]
    except requests.exceptions.RequestException as e:
        print(f"Error making initial API request: {e}")
        return None

    get_endpoint = 'https://api.us1.bfl.ai/v1/get_result'
    
    while True:
        time.sleep(0.8)
        
        try:
            result_response = requests.get(
                get_endpoint,
                headers={
                    'accept': 'application/json',
                    'x-key': BFL_API_KEY,
                },
                params={
                    'id': request_id,
                },
            ).json()
            
            if result_response["status"] == "Ready":
                image_url = result_response['result']['sample']
                
                # Download the image from the URL
                try:
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()  # Raises an HTTPError for bad responses
                    image_data = Image.open(BytesIO(image_response.content))
                    image = image_data
                    break
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading image from URL: {e}")
                    continue
                
        except requests.exceptions.RequestException as e:
            print(f"Error polling API for result: {e}")
            time.sleep(1)  # Wait a bit before retrying

    log_generation(prompt, image, seed, "flux")
    
    return image

# Generate Function via API
# First, let's modify the generate_image_fill function to properly handle the image and mask

def generate_image_fill(prompt, seed, image_with_mask):
    try:
        seed = int(seed)
        
        print(f"Received image_with_mask type: {type(image_with_mask)}")
        print(f"Image_with_mask keys: {image_with_mask.keys() if isinstance(image_with_mask, dict) else 'Not a dict'}")
        
        # Handle the format we're seeing: {'background', 'layers', 'composite'}
        if isinstance(image_with_mask, dict):
            # Get the original image
            if "background" in image_with_mask and image_with_mask["background"] is not None:
                image = image_with_mask["background"]
                print(f"Background image size: {image.size}")
            else:
                print("No background image found or background is None")
                return None
            
            # Convert image to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get the composite image (with the user's drawing)
            if "composite" in image_with_mask and image_with_mask["composite"] is not None:
                composite = image_with_mask["composite"]
                print(f"Composite image size: {composite.size}")
                if composite.mode != 'RGB':
                    composite = composite.convert('RGB')
            else:
                print("No composite image found or composite is None")
                return None
            
            # Resize images if they are smaller than 256x256 (minimum required by API)
            min_size = 256
            if image.width < min_size or image.height < min_size:
                # Calculate new size to preserve aspect ratio while meeting minimum
                aspect_ratio = image.width / image.height
                if aspect_ratio > 1:  # Wider than tall
                    new_width = max(min_size, image.width)
                    new_height = int(new_width / aspect_ratio)
                    if new_height < min_size:
                        new_height = min_size
                        new_width = int(new_height * aspect_ratio)
                else:  # Taller than wide
                    new_height = max(min_size, image.height)
                    new_width = int(new_height * aspect_ratio)
                    if new_width < min_size:
                        new_width = min_size
                        new_height = int(new_width / aspect_ratio)
                
                new_size = (new_width, new_height)
                print(f"Resizing images from {image.size} to {new_size}")
                image = image.resize(new_size, Image.LANCZOS)
                composite = composite.resize(new_size, Image.LANCZOS)
                print(f"Resized image size: {image.size}")
            
            # Create a mask by comparing the original and composite images
            mask = Image.new("L", image.size, 0)  # Use "L" mode for grayscale mask
            
            # Convert both images to RGB for comparison (already done above)
            image_rgb = image
            composite_rgb = composite
            
            # Compare the images pixel by pixel
            # If they're different, the user has drawn there
            mask_pixels_found = 0
            for y in range(image.height):
                for x in range(image.width):
                    bg_pixel = image_rgb.getpixel((x, y))
                    comp_pixel = composite_rgb.getpixel((x, y))
                    
                    # If pixels are different, mark as part of the mask
                    if bg_pixel != comp_pixel:
                        mask.putpixel((x, y), 255)  # White for areas to modify
                        mask_pixels_found += 1
            
            print(f"Mask pixels found: {mask_pixels_found}")
            
            # Check if any mask was created
            if mask_pixels_found == 0:
                print("No mask detected - user may not have drawn anything")
                return None
            
            # Alternative approach: use the layers directly if available
            if "layers" in image_with_mask and image_with_mask["layers"]:
                print(f"Found {len(image_with_mask['layers'])} layers")
                for i, layer in enumerate(image_with_mask["layers"]):
                    if layer is not None:
                        print(f"Layer {i} size: {layer.size}, mode: {layer.mode}")
                        # Resize layer to match the new image size if necessary
                        if layer.size != image.size:
                            layer = layer.resize(image.size, Image.LANCZOS)
                        # Convert layer to RGBA if it's not already
                        if layer.mode != 'RGBA':
                            layer = layer.convert('RGBA')
                        
                        # Get the alpha channel
                        alpha = layer.split()[3]
                        
                        # For each pixel where alpha is not zero, set the mask to white
                        for y in range(image.height):
                            for x in range(image.width):
                                if alpha.getpixel((x, y)) > 0:
                                    mask.putpixel((x, y), 255)
            
            # Save the mask for debugging
            debug_dir = "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            mask.save(os.path.join(debug_dir, "mask_debug.png"))
            image.save(os.path.join(debug_dir, "original_debug.png"))
            composite.save(os.path.join(debug_dir, "composite_debug.png"))
            print("Debug images saved to ./debug/")
            
        else:
            print(f"Unexpected image editor output type: {type(image_with_mask)}")
            return None
        
        print("Mask created successfully")
        
        # Convert the image and mask to base64 strings
        buffered_image = BytesIO()
        image.save(buffered_image, format="PNG")
        buffered_image.seek(0)
        image_base64 = base64.b64encode(buffered_image.getvalue()).decode('utf-8')
        
        buffered_mask = BytesIO()
        mask.save(buffered_mask, format="PNG")
        buffered_mask.seek(0)
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode('utf-8')
        
        print("Images converted to base64")
        
        post_endpoint = 'https://api.bfl.ai/v1/flux-pro-1.0-fill'
        headers = {
            'accept': 'application/json',
            'x-key': BFL_API_KEY,
            'Content-Type': 'application/json',
        }
        request_data = {
            'prompt': prompt,
            'seed': seed,
            'image': image_base64,
            'mask': mask_base64,
        }
        
        print("Sending request to API...")
        
        try:
            response = requests.post(post_endpoint, headers=headers, json=request_data)
            print(f"API Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            response_json = response.json()
            request_id = response_json["id"]
            polling_url = response_json.get("polling_url")  # Extract the dynamic polling URL
            if not polling_url:
                print("No polling_url found in response. Cannot poll for results.")
                return None
            print(f"Request ID: {request_id}")
            print(f"Polling URL: {polling_url}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error making initial API request: {e}")
            return None

        print("Polling for results...")
        while True:
            time.sleep(0.8)
            
            try:
                result_response = requests.get(
                    polling_url,  # Use the dynamic polling URL
                    headers={
                        'accept': 'application/json',
                        'x-key': BFL_API_KEY,
                    },
                    params={
                        'id': request_id,
                    },
                ).json()
                
                print(f"Status: {result_response['status']}")
                
                if result_response["status"] == "Ready":
                    image_url = result_response['result']['sample']
                    
                    # Download the image from the URL
                    try:
                        image_response = requests.get(image_url)
                        image_response.raise_for_status()
                        image_data = Image.open(BytesIO(image_response.content))
                        result_image = image_data
                        break
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading image from URL: {e}")
                        continue
                elif result_response["status"] == "Error":
                    print(f"API returned error: {result_response}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Error polling API for result: {e}")
                time.sleep(1)  # Wait a bit before retrying

        log_generation(prompt, result_image, seed)
        print("Generation completed successfully")
        
        return result_image
        
    except Exception as e:
        print(f"Error in generate_image_fill: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_image_edit(prompt, seed, input_image):
    try:
        seed = int(seed)
        
        print(f"Starting image edit with prompt: {prompt}")
        
        # Check if input_image is provided
        if input_image is None:
            print("No input image provided")
            return None
        
        # Convert the input image to base64
        if isinstance(input_image, Image.Image):
            # If it's already a PIL Image
            image = input_image
        else:
            # If it's from Gradio, it might be a numpy array or other format
            try:
                image = Image.fromarray(input_image)
            except:
                print("Error converting input image to PIL Image")
                return None
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Input image size: {image.size}, mode: {image.mode}")
        
        # Convert the image to base64 string
        buffered_image = BytesIO()
        image.save(buffered_image, format="JPEG")
        buffered_image.seek(0)
        image_base64 = base64.b64encode(buffered_image.getvalue()).decode('utf-8')
        
        print("Image converted to base64")
        
        post_endpoint = 'https://api.bfl.ai/v1/flux-kontext-pro'
        headers = {
            'accept': 'application/json',
            'x-key': BFL_API_KEY,
            'Content-Type': 'application/json',
        }
        request_data = {
            'prompt': prompt,
            'input_image': image_base64,
            'seed': seed
        }
        
        print("Sending request to Flux Kontext Pro API...")
        
        try:
            response = requests.post(post_endpoint, headers=headers, json=request_data)
            print(f"API Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            response_json = response.json()
            request_id = response_json["id"]
            polling_url = response_json.get("polling_url")  # Extract the dynamic polling URL
            if not polling_url:
                print("No polling_url found in response. Cannot poll for results.")
                return None
            print(f"Request ID: {request_id}")
            print(f"Polling URL: {polling_url}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error making initial API request: {e}")
            return None

        print("Polling for results...")
        while True:
            time.sleep(0.8)
            
            try:
                result_response = requests.get(
                    polling_url,  # Use the dynamic polling URL
                    headers={
                        'accept': 'application/json',
                        'x-key': BFL_API_KEY,
                    },
                    params={
                        'id': request_id,
                    },
                ).json()
                
                print(f"Status: {result_response['status']}")
                
                if result_response["status"] == "Ready":
                    image_url = result_response['result']['sample']
                    
                    # Download the image from the URL
                    try:
                        image_response = requests.get(image_url)
                        image_response.raise_for_status()
                        image_data = Image.open(BytesIO(image_response.content))
                        result_image = image_data
                        break
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading image from URL: {e}")
                        continue
                elif result_response["status"] == "Error":
                    print(f"API returned error: {result_response}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Error polling API for result: {e}")
                time.sleep(1)  # Wait a bit before retrying

        log_generation(prompt, result_image, seed)
        print("Generation completed successfully")
        
        return result_image
        
    except Exception as e:
        print(f"Error in generate_image_edit: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_ui():
    """
    Create the Gradio UI.
    """
    # CSS for styling
    css = """
        .gradio-container {
            width: 100% !important;
            max-width: 1200px !important;
            min-width: 320px !important;
            margin: auto !important;
            padding-top: 20px !important;
            box-sizing: border-box !important;
        }
        @media (max-width: 1200px) {
            .gradio-container {
                padding: 10px !important;
            }
        }
        
        @media (max-width: 768px) {
            .gradio-container {
                padding: 5px !important;
            }
        }
        .header-text {
            text-align: center;
            margin-bottom: 10px;
        }
        #intro-text::first-line {
            font-weight: bold;
        }
        .body-text {
            text-align: left;
            line-height: 1.6;
            font-size: 16px;
            color: #333;
        }
        
        .body-text strong {
            font-weight: 700;
        }
        .theme-section {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
        }
        """
    
    # Gradio App
    with gr.Blocks(theme=Ocean(), css=css) as demo:
        
        gr.Markdown(
            """
            # AI Image Generation Suite hosted by Pratt Institute
            """,
            elem_classes=["header-text"],
        )
        
        with gr.Tab("Google Gemini"):
            gr.Markdown(load_content("gemini.md"), elem_classes=["body-text"])

            with gr.Row():
                gemini_prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    value="Create a picture of a cat eating a nano-banana in a fancy restaurant under the Gemini constellation",
                    lines=3
                )

            with gr.Row():
                # Allow multiple optional reference images
                gemini_input_images = gr.Files(
                    label="Upload reference images (optional, multiple)",
                    file_types=["image"],
                    file_count="multiple"
                )

            with gr.Row():
                gemini_seed = gr.Number(label="Seed", value=random.randrange(0, 1000000000), minimum=0, maximum=1000000000)

            with gr.Row():
                gemini_generate_button = gr.Button("Generate with Gemini", variant="primary")

            with gr.Row():
                gemini_output_image = gr.Image(label="Generated Image", type="pil", format="png")

            gemini_generate_button.click(
                fn=generate_image_gemini,
                inputs=[gemini_prompt_input, gemini_seed, gemini_input_images],
                outputs=gemini_output_image,
                show_progress=True,
            )
        
        with gr.Tab("OpenAI GPT Image"):
            gr.Markdown(load_content("openai.md"), elem_classes=["body-text"])

            with gr.Row():
                openai_prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    value="A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter.",
                    lines=3
                )

            with gr.Row():
                openai_input_images = gr.Files(
                    label="Upload reference images (optional, multiple for edits)",
                    file_types=["image"],
                    file_count="multiple"
                )

            with gr.Row():
                openai_seed = gr.Number(label="Seed (not used by OpenAI)", value=random.randrange(0, 1000000000), minimum=0, maximum=1000000000)

            with gr.Row():
                openai_generate_button = gr.Button("Generate with OpenAI", variant="primary")

            with gr.Row():
                openai_output_image = gr.Image(label="Generated Image", type="pil", format="png")

            openai_generate_button.click(
                fn=generate_image_openai,
                inputs=[openai_prompt_input, openai_seed, openai_input_images],
                outputs=openai_output_image,
                show_progress=True,
            )
            
        with gr.Tab("Flux-1.1-Pro-Ultra"):
            gr.Markdown(load_content("flux_ultra.md"), elem_classes=["body-text"])

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    value="A robot made of exotic candies and chocolates..."
                )
                seed = gr.Number(label="Seed", value=random.randrange(0, 1000000000), minimum=0, maximum=1000000000)

            with gr.Row():
                generate_button = gr.Button("Generate", variant="primary")
            
            with gr.Row():
                output_image = gr.Image(label="Generated Image", type="pil", format="png")
            
            generate_button.click(
                fn=generate_image,
                inputs=[prompt_input, seed],
                outputs=output_image
            )
        
        
        with gr.Tab("Generative Fill"):
            gr.Markdown(load_content("generative_fill.md"), elem_classes=["body-text"])

            with gr.Row():
                img_edit = gr.ImageEditor(
                    label="Upload Image and Draw Mask", 
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                
            with gr.Row():
                fill_prompt_input = gr.Textbox(
                    label="Enter your prompt for the masked area",
                    value="A beautiful landscape with mountains and a lake"
                )
                fill_seed = gr.Number(label="Seed", value=random.randrange(0, 1000000000), minimum=0, maximum=1000000000)
                
            with gr.Row():
                fill_generate_button = gr.Button("Generate Fill", variant="primary")
 
            with gr.Row():
                fill_output_image = gr.Image(label="Generated Image", type="pil", format="png")
                
            fill_generate_button.click(
                fn=generate_image_fill,
                inputs=[fill_prompt_input, fill_seed, img_edit],
                outputs=fill_output_image
            )
            
        
        with gr.Tab("Kontext"):
            gr.Markdown(load_content("kontext.md"), elem_classes=["body-text"])
                    
            with gr.Row():
                edit_img_input = gr.Image(
                    label="Upload Image to Edit", 
                    type="pil",
                )

            with gr.Row():
                edit_prompt_input = gr.Textbox(
                    label="Describe what you want to edit or change in the image",
                    value="Add a beautiful sunset in the background",
                    lines=3
                )
                
            with gr.Row():
                edit_seed = gr.Number(label="Seed", value=random.randrange(0, 1000000000), minimum=0, maximum=1000000000)
                
            with gr.Row():
                edit_generate_button = gr.Button("Edit Image", variant="primary")
                
            with gr.Row():
                edit_output_image = gr.Image(label="Edited Image", type="pil", format="png")
                
            edit_generate_button.click(
                fn=generate_image_edit,
                inputs=[edit_prompt_input, edit_seed, edit_img_input],
                outputs=edit_output_image,
                show_progress=True,
            )
            
        gr.Markdown("<br><center>Made with ❤️ by <strong>Pratt Technology</strong></a></center>")
        
    return demo

def main():
    """
    Main function to run the Gradio app.
    """
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7867, root_path="/flux-pro", show_api=False)

if __name__ == "__main__":
    main()