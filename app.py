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

import subprocess
GIT_TOKEN = os.environ.get("GIT_TOKEN")
import subprocess

cmd = [
    "pip",
    "install",
    "git+https://eleazhong:${GIT_TOKEN}@github.com/wand-ai/wand-ml",
]

# If GIT_TOKEN is a Python variable, build the string in Python instead:
# cmd = f"pip install git+https://eleazhong:{GIT_TOKEN}@github.com/wand-ai/wand-ml"

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,       # or encoding="utf-8" on older Python
    bufsize=1,
)

for line in proc.stdout:
    print(line, end="")   # already has newline

proc.wait()
print("Exit code:", proc.returncode)

from qwenimage.debug import ctimed
from qwenimage.foundation import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

# --- Model Loading ---

# foundation = QwenImageFoundation(QwenConfig(
#     vae_image_size=1024 * 1024,
#     regression_base_pipe_steps=4,
# ))
# finetuner = QwenLoraFinetuner(foundation, foundation.config)
# finetuner.load("checkpoints/reg-mse-pixel-lpips_005000", lora_rank=32)


dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"


pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map=device
    ),
    torch_dtype=dtype,
)
pipe = pipe.to(device=device, dtype=dtype)
pipe.load_lora_weights(
    "checkpoints/distill_5k_lora.safetensors",
    adapter_name="fast_5k",
)
# pipe.unload_lora_weights()

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

    # finetuner.enable()
    pipe.scheduler.config["base_shift"] = shift
    pipe.scheduler.config["max_shift"] = shift

    result = pipe(
        image=pil_images,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

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