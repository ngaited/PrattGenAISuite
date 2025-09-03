# PrattGenAISuite

An AI generation platform hosted by Pratt Institute that provides access to multiple AI models for image and video generation through user-friendly Gradio interfaces.

## Features

### Image Generation Suite (`gradio_fluxPro.py`)
- **Google Gemini 2.5 Flash**: Text-to-image and image-to-image generation
- **OpenAI GPT Image**: Text-to-image and image editing capabilities  
- **Flux-1.1-Pro-Ultra**: High-quality text-to-image generation
- **Generative Fill**: Inpainting/outpainting with mask-based editing
- **Kontext**: Advanced image editing and modification

### Video Generation Suite (`gradio_vidGen.py`)
- **RunwayML Gen-4**: Image-to-video with customizable aspect ratios and durations
- **Google Veo3**: Text-to-video and image-to-video with 720p output

## Setup

### 1. Install Dependencies

```bash
pip install gradio pillow requests pyyaml google-genai openai runwayml numpy
```

### 2. Configure API Keys

1. Copy the example configuration file:
   ```bash
   cp .config.yml.example .config.yml
   ```

2. Edit `.config.yml` and add your API keys:
   ```yaml
   BFL_API_KEY: your_black_forest_labs_api_key_here
   GOOGLE_API_KEY: your_google_api_key_here
   GOOGLE_VEO3_KEY: your_google_veo3_api_key_here
   OPENAI_API_KEY: your_openai_api_key_here
   RUNWAY_API_KEY: your_runway_api_key_here
   ```

### 3. Run the Applications

**Image Generation Suite:**
```bash
python3 gradio_fluxPro.py
```
Access at: `http://localhost:7867/flux-pro`

**Video Generation Suite:**
```bash
python3 gradio_vidGen.py
```
Access at: `http://localhost:7871/vidgen`

## API Key Requirements

- **BFL_API_KEY**: Black Forest Labs API key for Flux models
- **GOOGLE_API_KEY**: Google Cloud API key with Gemini API access
- **GOOGLE_VEO3_KEY**: Google Cloud API key with Veo3 access (may be same as GOOGLE_API_KEY)
- **OPENAI_API_KEY**: OpenAI API key with image generation access
- **RUNWAY_API_KEY**: RunwayML API key for video generation

## Generated Content

All generated images and videos are automatically logged to `./logs/YYYY-MM-DD/` with metadata including prompts, seeds, and model information.

## Educational Use

This platform is designed for educational use at Pratt Institute. All generated content should comply with Pratt Institute's [Statement on Artificial Intelligence](https://www.pratt.edu/resources/statement-on-artificial-intelligence/).

## Copyright Notice

Outputs generated solely by AI systems are not eligible for copyright protection under U.S. law. See the [Copyright Registration Guidance](https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence) for more information.