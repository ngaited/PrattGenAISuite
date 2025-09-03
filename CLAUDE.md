# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The PrattGenAISuite is a Gradio-based AI generation platform hosted by Pratt Institute that provides access to multiple AI models for image and video generation. The suite includes:

- **gradio_fluxPro.py**: Multi-model image generation interface supporting Flux Ultra, Google Gemini, OpenAI GPT Image, Generative Fill, and Kontext editing
- **gradio_vidGen.py**: Video generation interface supporting RunwayML Gen-4 and Google Veo3 models

## Architecture

### Core Components

**Image Generation Models** (gradio_fluxPro.py):
- Flux-1.1-Pro-Ultra: Text-to-image via Black Forest Labs API
- Google Gemini 2.5 Flash: Text-to-image and image-to-image via Google Gemini API
- OpenAI GPT Image: Text-to-image and image editing via OpenAI API
- Generative Fill: Inpainting/outpainting using Flux Pro Fill API with mask generation
- Kontext: Image editing via Flux Kontext Pro API

**Video Generation Models** (gradio_vidGen.py):
- RunwayML Gen-4: Image-to-video with customizable aspect ratios and durations
- Google Veo3: Text-to-video and image-to-video with 8-second 720p output

### Configuration System

Both applications load API keys from `.config.yml`. Copy `.config.yml.example` to `.config.yml` and add your API keys:
```yaml
BFL_API_KEY: your_black_forest_labs_api_key_here
GOOGLE_API_KEY: your_google_api_key_here
GOOGLE_VEO3_KEY: your_google_veo3_api_key_here
OPENAI_API_KEY: your_openai_api_key_here
RUNWAY_API_KEY: your_runway_api_key_here
```

**Important**: The `.config.yml` file is excluded from git via `.gitignore` to protect API keys.

### Logging System

Both apps implement comprehensive logging:
- Creates daily log directories in `./logs/YYYY-MM-DD/`
- Logs generated images with metadata (timestamp, prompt, seed, model type)
- Video generations log URLs/paths with extended metadata

## Development Commands

### Running the Applications

**Image Generation Suite:**
```bash
python3 gradio_fluxPro.py
# Launches on http://0.0.0.0:7867/flux-pro
```

**Video Generation Suite:**
```bash
python3 gradio_vidGen.py  
# Launches on http://0.0.0.0:7871/vidgen
```

### Dependencies

The project requires these Python packages:
- gradio
- PIL (Pillow)
- requests
- pyyaml
- google-genai
- openai
- runwayml (optional, for video generation)
- numpy (for image processing)

Install missing dependencies:
```bash
pip install gradio pillow requests pyyaml google-genai openai runwayml numpy
```

## Key Implementation Details

### Image Processing Pipeline

The `_coerce_to_pil_list()` function normalizes various Gradio input formats (single images, file objects, numpy arrays) into PIL Image lists for consistent processing across all models.

### API Integration Patterns

All model integrations follow a similar pattern:
1. Input validation and preprocessing
2. API request with proper headers and authentication  
3. Polling for results (for async APIs like Flux)
4. Image/video download and processing
5. Logging of generation metadata

### Error Handling

Each generation function includes comprehensive error handling for:
- Missing API keys
- Network request failures
- API response errors
- File processing errors
- Invalid input formats

## Content Documentation

The `/content/` directory contains markdown files with model-specific documentation, prompting guides, and usage instructions that are displayed in the Gradio interface tabs.