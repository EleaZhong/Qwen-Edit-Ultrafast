import copy
import math
import random
import os
import tempfile
import sys

import numpy as np
import torch
from PIL import Image
import gradio as gr
import spaces

from qwenimage.datamodels import QwenConfig
from qwenimage.debug import ctimed, ftimed
from qwenimage.experiments.experiments_qwen import ExperimentRegistry
from qwenimage.finetuner import QwenLoraFinetuner
from qwenimage.foundation import QwenImageFoundation
from qwenimage.prompt import build_camera_prompt

# --- Model Loading ---

foundation = QwenImageFoundation(QwenConfig(
    vae_image_size=1024 * 1024,
    regression_base_pipe_steps=4,
))
finetuner = QwenLoraFinetuner(foundation, foundation.config)
finetuner.load("checkpoints/reg-mse-pixel-lpips_005000", lora_rank=32)



MAX_SEED = np.iinfo(np.int32).max


@spaces.GPU
def run_pipe(
    image,
    prompt,
    seed,
    randomize_seed,
    num_inference_steps,
    shift,
    prev_output = None,
    progress=gr.Progress(track_tqdm=True)
):
    with ctimed("pre pipe"):

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        # Choose input image (prefer uploaded, else last output)
        pil_images = []
        if image is not None:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            elif hasattr(image, "name"):
                pil_images.append(Image.open(image.name).convert("RGB"))
        elif prev_output:
            pil_images.append(prev_output.convert("RGB"))

        if len(pil_images) == 0:
            raise gr.Error("Please upload an image first.")
        
        print(f"{len(pil_images)=}")

    finetuner.enable()
    foundation.scheduler.config["base_shift"] = shift
    foundation.scheduler.config["max_shift"] = shift

    result = foundation.base_pipe(foundation.INPUT_MODEL(
        image=pil_images,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ))[0]

    return result, seed


# --- UI ---


with gr.Blocks(theme=gr.themes.Citrus()) as demo:

    gr.Markdown("Qwen Image Demo")

    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Input Image", type="pil")
            prev_output = gr.Image(value=None, visible=False)
            is_reset = gr.Checkbox(value=False, visible=False)
            prompt = gr.Textbox(label="Prompt", placeholder="Prompt", lines=2)


            run_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=2)
                shift = gr.Slider(label="Timestep Shift", minimum=0.0, maximum=4.0, step=0.1, value=2.0)

        with gr.Column():
            result = gr.Image(label="Output Image", interactive=False)
                    
    inputs = [
        image,
        prompt,
        seed, 
        randomize_seed,
        num_inference_steps,
        shift,
        prev_output,
    ]
    outputs = [result, seed]

    
    run_event = run_btn.click(
        fn=run_pipe, 
        inputs=inputs, 
        outputs=outputs
    )

    run_event.then(lambda img, *_: img, inputs=[result], outputs=[prev_output])

demo.launch()