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
from typing import List, Union, Optional, Dict, Any
import tempfile  # Added for temporary video file handling

# Import RunwayML library
try:
    from runwayml import RunwayML
except ImportError:
    print("RunwayML library not found. Please install it using 'pip install runwayml'")
    RunwayML = None # Set to None if not installed

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
with io.open(r'.config.yml','r') as stream:
    db = yaml.safe_load(stream)

BFL_API_KEY = db['BFL_API_KEY']
GOOGLE_API_KEY = db.get('GOOGLE_VEO3_KEY')
RUNWAY_API_KEY = db.get('RUNWAY_API_KEY') # Get Runway API key

# Initialize Google Gemini client
if GOOGLE_API_KEY:
    client_gemini = genai.Client(api_key=GOOGLE_API_KEY)
else:
    client_gemini = None
    print("Google API key not found. Gemini generation will be disabled.")

# Initialize RunwayML client
client_runway = None
if RUNWAY_API_KEY and RunwayML:
    # RunwayML library expects the API key in the env var RUNWAYML_API_SECRET
    os.environ['RUNWAYML_API_SECRET'] = RUNWAY_API_KEY
    try:
        client_runway = RunwayML()
        print("RunwayML client initialized successfully.")
    except Exception as e:
        print(f"Error initializing RunwayML client: {e}")
        client_runway = None
else:
    print("RunwayML API key not found or library not installed. RunwayML generation will be disabled.")


# Logging Functionality
def create_log_directory():
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join('./logs', current_date)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Modified log_generation to handle both images and videos
def log_generation(prompt: str, output: Union[Image.Image, str], seed: int, model_type: str, metadata: Optional[Dict[str, Any]] = None):
    log_dir = create_log_directory()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_data = {
        "timestamp": timestamp,
        "prompt": prompt,
        "seed": seed,
        "model_type": model_type,
        "metadata": metadata if metadata else {}
    }

    output_filename = None
    if isinstance(output, Image.Image):
        # Handle image output
        output_filename = f"{timestamp}_{model_type}_seed_{seed}.png"
        image_path = os.path.join(log_dir, output_filename)
        output.save(image_path)
        log_data["output_filename"] = output_filename
    elif isinstance(output, str) and (output.startswith("http") or os.path.isfile(output)):
        # Handle video URL or local file path
        if output.startswith("http"):
            log_data["output_url"] = output
        else:
            log_data["output_filename"] = os.path.basename(output)
        output_filename = f"{timestamp}_{model_type}_seed_{seed}_video_log.json"
    else:
        print(f"Warning: Unknown output type for logging: {type(output)}")
        return

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
    # Ensure input_images is iterable if it's not None and not a single Image
    if not isinstance(input_images, list):
        input_images = [input_images]

    for item in input_images:
        try:
            if isinstance(item, Image.Image):
                img = item
            elif isinstance(item, bytes):
                img = Image.open(BytesIO(item))
            elif isinstance(item, str) and os.path.exists(item): # Check if string is a valid path
                img = Image.open(item)
            elif hasattr(item, "name") and os.path.exists(item.name):  # gr.Files file object
                img = Image.open(item.name)
            else:
                # Try numpy array or dict with 'name'/'path'
                try:
                    import numpy as np
                    if isinstance(item, dict) and item.get("name") and os.path.exists(item["name"]):
                        img = Image.open(item["name"])
                    elif isinstance(item, dict) and item.get("path") and os.path.exists(item["path"]):
                        img = Image.open(item["path"])
                    elif isinstance(item, np.ndarray):
                        img = Image.fromarray(item)
                    else:
                        continue # Skip if not a recognized image type
                except Exception:
                    continue

            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_list.append(img)
        except Exception as e:
            print(f"Error processing image item {item}: {e}")
            continue
    return pil_list

# Helper to convert PIL Image to Base64 Data URI
def pil_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG") # PNG is a good default for quality and transparency support
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# RunwayML Video Generation Function
def generate_video_runway(
    prompt_text: str,
    input_image: Optional[gr.File], # Use gr.File for single image input
    model: str,
    ratio: str,
    seed: int,
    duration: int,
    progress=gr.Progress() # Gradio injects this object
):
    if not client_runway:
        return None, gr.Warning("RunwayML API key not configured or client not initialized.")
    if not input_image:
        return None, gr.Warning("Please upload an image to generate a video with RunwayML.")
    if not prompt_text:
        return None, gr.Warning("Please provide a text prompt for RunwayML video generation.")

    try:
        # Convert Gradio File object to PIL Image, then to Data URI
        pil_image_list = _coerce_to_pil_list(input_image)
        if not pil_image_list:
            return None, gr.Error("Failed to process the uploaded image.")
        
        prompt_image_data_uri = pil_to_data_uri(pil_image_list[0])

        print(f"RunwayML request: model={model}, ratio={ratio}, seed={seed}, duration={duration}")

        # Start the video generation task
        # This returns an ImageToVideoCreateResponse object, which has an 'id' but not 'status'
        task_submission_response = client_runway.image_to_video.create(
            model=model,
            prompt_image=prompt_image_data_uri, # Use data URI
            prompt_text=prompt_text,
            ratio=ratio,
            seed=seed,
            duration=duration,
        )

        # Get the task ID from the submission response
        task_id = task_submission_response.id

        progress(0, desc="RunwayML video generation started. This may take a few minutes...")

        # Retrieve the full task object immediately to get its initial status
        task = client_runway.tasks.retrieve(id=task_id)

        # Now, the 'task' object has the 'status' attribute and can be polled
        while task.status in ["PENDING", "RUNNING", "THROTTLED"]:
            if task.status == "RUNNING" and task.progress is not None:
                progress(task.progress, desc=f"Generating video... ({int(task.progress*100)}%)")
            else:
                # Provide a small progress update even if precise progress is not available yet
                progress(0.1, desc=f"Task status: {task.status}...") 
            time.sleep(5) # Poll every 5 seconds
            task = client_runway.tasks.retrieve(id=task_id) # Re-fetch task to get updated status

        if task.status == "SUCCEEDED":
            video_url = task.output[0] # Assuming one video output
            log_generation(
                prompt_text,
                video_url,
                seed,
                f"runway-{model}",
                metadata={"model": model, "ratio": ratio, "duration": duration}
            )
            return video_url, gr.Info("Video generated successfully with RunwayML!")
        else:
            error_message = f"RunwayML task failed with status: {task.status}"
            if task.failure:
                error_message += f"\nReason: {task.failure}"
            if task.failureCode:
                error_message += f"\nCode: {task.failureCode}"
            print(error_message)
            return None, gr.Error(error_message)

    except Exception as e:
        print(f"Error in generate_video_runway: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, gr.Error(f"Error generating video with RunwayML: {str(e)}")

# New Veo3 Video Generation Function (Text-to-Video or Image-to-Video)
def generate_video_veo3(
    prompt: str,
    input_image: Optional[gr.File] = None,
    progress=gr.Progress()
):
    if not client_gemini:
        return None, gr.Warning("Google API key not configured or client not initialized.")
    if not prompt:
        return None, gr.Warning("Please provide a text prompt for Veo3 video generation.")

    try:
        # Process optional image for image-to-video
        image = None
        if input_image:
            pil_images = _coerce_to_pil_list(input_image)
            if pil_images:
                image = pil_images[0]  # Use the first processed image
                print("Veo3 request: image-to-video")
            else:
                return None, gr.Warning("Failed to process the uploaded image for Veo3.")
        else:
            print("Veo3 request: text-to-video")

        # Start video generation
        operation = client_gemini.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            image=image,  # None for text-to-video, PIL Image for image-to-video
        )

        progress(0, desc="Veo3 video generation started. This may take several minutes...")

        # Poll until done
        while not operation.done:
            progress(0.5, desc="Waiting for Veo3 video generation to complete...")
            time.sleep(10)
            operation = client_gemini.operations.get(operation)

        if operation.done and hasattr(operation.response, 'generated_videos') and operation.response.generated_videos:
            generated_video = operation.response.generated_videos[0]
            
            # Create a temporary file for the video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Download the video using the correct method
            progress(0.9, desc="Downloading generated video...")
            
            # The correct way to download - without the 'path' parameter
            video_data = client_gemini.files.download(file=generated_video.video)
            
            # Write the downloaded data to the temporary file
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Log the generation (use a dummy seed since Veo3 doesn't support it)
            dummy_seed = 0
            log_generation(
                prompt,
                temp_path,
                dummy_seed,
                "veo3",
                metadata={"mode": "image-to-video" if image else "text-to-video"}
            )
            
            return temp_path, gr.Info("Video generated successfully with Veo3!")
        else:
            return None, gr.Error("Veo3 video generation failed or returned no video.")

    except Exception as e:
        error_message = str(e)
        if "API_KEY_HTTP_REFERRER_BLOCKED" in error_message:
            return None, gr.Error("API key restriction error. Please check your Google Cloud Console API key settings and remove HTTP referrer restrictions or add localhost to allowed referrers.")
        elif "PERMISSION_DENIED" in error_message:
            return None, gr.Error("Permission denied. Please ensure your API key has access to the Generative Language API and Veo3 model.")
        else:
            print(f"Error in generate_video_veo3: {error_message}")
            import traceback
            traceback.print_exc()
            return None, gr.Error(f"Error generating video with Veo3: {error_message}")

# Define model-specific ratios
RUNWAY_MODEL_RATIOS = {
    "gen3a_turbo": ["1280:768", "768:1280"],
    "gen4_turbo": ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"]
}

def update_runway_ratios(model_name):
    """Dynamically update ratio dropdown based on selected model."""
    ratios = RUNWAY_MODEL_RATIOS.get(model_name, [])
    return gr.Dropdown(choices=ratios, value=ratios[0] if ratios else None, interactive=True)


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
            # AI Video Generation Suite hosted by Pratt Institute
            """,
            elem_classes=["header-text"],
        )
        
                
        with gr.Tab("RunwayML (Video Generation)"):
            gr.Markdown(load_content("runway.md"), elem_classes=["body-text"])

            with gr.Row():
                runway_input_image = gr.File(
                    label="Upload a single reference image (required)",
                    file_types=["image"],
                    file_count="single",
                    interactive=bool(client_runway)
                )
            
            with gr.Row():
                runway_prompt_text = gr.Textbox(
                    label="Enter your video prompt",
                    value="A cat eating a nano-banana in a fancy restaurant, slowly zooming out.",
                    lines=3,
                    max_lines=5,
                    interactive=bool(client_runway)
                )
            
            with gr.Row():
                runway_model = gr.Radio(
                    label="Model",
                    choices=list(RUNWAY_MODEL_RATIOS.keys()),
                    value="gen4_turbo",
                    interactive=bool(client_runway)
                )
                runway_ratio = gr.Dropdown(
                    label="Video Ratio (Resolution)",
                    choices=RUNWAY_MODEL_RATIOS["gen4_turbo"], # Default to gen4_turbo ratios
                    value="1280:720",
                    interactive=bool(client_runway)
                )
            
            # Dynamic update for ratios based on model selection
            runway_model.change(
                fn=update_runway_ratios,
                inputs=runway_model,
                outputs=runway_ratio
            )

            with gr.Row():
                runway_seed = gr.Number(
                    label="Seed",
                    value=random.randrange(0, 4294967295),
                    minimum=0,
                    maximum=4294967295,
                    step=1,
                    interactive=bool(client_runway)
                )
                runway_duration = gr.Radio(
                    label="Video Duration (seconds)",
                    choices=[5, 10],
                    value=10,
                    interactive=bool(client_runway)
                )
            
            with gr.Row():
                runway_generate_button = gr.Button("Generate Video with RunwayML", variant="primary", interactive=bool(client_runway))

            with gr.Row():
                runway_output_video = gr.Video(label="Generated Video")
                runway_message_output = gr.Textbox(label="Status/Messages", interactive=False, lines=2) # New message component

            runway_generate_button.click(
                fn=generate_video_runway,
                inputs=[
                    runway_prompt_text,
                    runway_input_image,
                    runway_model,
                    runway_ratio,
                    runway_seed,
                    runway_duration
                ],
                outputs=[runway_output_video, runway_message_output], # Updated outputs
                show_progress=True,
            )
        
        with gr.Tab("Google Veo3 (Video Generation)"):
            gr.Markdown(load_content("veo3.md"), elem_classes=["body-text"])

            with gr.Row():
                veo3_prompt_input = gr.Textbox(
                    label="Enter your video prompt",
                    value="A whimsical stop-motion animation of a tiny robot tending to a garden of glowing mushrooms on a miniature planet.",
                    lines=3,
                    max_lines=5,
                    interactive=bool(client_gemini)
                )

            with gr.Row():
                veo3_input_image = gr.File(
                    label="Upload a reference image (optional, for image-to-video)",
                    file_types=["image"],
                    file_count="single",
                    interactive=bool(client_gemini)
                )

            with gr.Row():
                veo3_generate_button = gr.Button("Generate Video with Veo3", variant="primary", interactive=bool(client_gemini))

            with gr.Row():
                veo3_output_video = gr.Video(label="Generated Video")
                veo3_message_output = gr.Textbox(label="Status/Messages", interactive=False, lines=2)

            veo3_generate_button.click(
                fn=generate_video_veo3,
                inputs=[veo3_prompt_input, veo3_input_image],
                outputs=[veo3_output_video, veo3_message_output],
                show_progress=True,
            )
            
        gr.Markdown("<br><center>Made with ❤️ by <strong>Pratt Technology</strong></a></center>")
        
    return demo

def main():
    """
    Main function to run the Gradio app.
    """
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7871, root_path="/vidgen", show_api=False)

if __name__ == "__main__":
    main()