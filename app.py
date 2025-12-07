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

from qwenimage.models.attention_processors import QwenDoubleStreamAttnProcessorFA3
from qwenimage.optimization import optimize_pipeline_
GIT_TOKEN = os.environ.get("GIT_TOKEN")
import subprocess

# cmd = f"pip install git+https://eleazhong:{GIT_TOKEN}@github.com/wand-ai/wand-ml"

# proc = subprocess.Popen(
#     cmd,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.STDOUT,
#     text=True,       # or encoding="utf-8" on older Python
#     bufsize=1,
# )

# for line in proc.stdout:
#     print(line, end="")   # already has newline

# proc.wait()
# print("Exit code:", proc.returncode)

from qwenimage.debug import ctimed
from qwenimage.models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.models.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.experiments.quantize_experiments import conf_fp8darow_nolast, quantize_transformer_fp8darow_nolast

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
pipe.set_adapters(["fast_5k"], adapter_weights=[1.0])
pipe.fuse_lora(adapter_names=["fast_5k"], lora_scale=1.0)
pipe.unload_lora_weights()

@spaces.GPU(duration=1500)
def optim_pipe():
    print(f"func cuda: {torch.cuda.is_available()=}")

    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    pipe.transformer.fuse_qkv_projections()
    pipe.transformer.check_fused_qkv()

    optimize_pipeline_(
        pipe,
        cache_compiled=True,
        quantize=True,
        suffix="_fp8darow_nolast_fa3_fast5k",
        quantize_config=conf_fp8darow_nolast(),
        pipe_kwargs={
            "image": [Image.new("RGB", (1024, 1024))],
            "prompt":"prompt",
            "num_inference_steps":2,
        }
    )

optim_pipe()

MAX_SEED = np.iinfo(np.int32).max


@spaces.GPU
def run_pipe(
    image,
    prompt,
    num_runs,
    seed,
    randomize_seed,
    num_inference_steps,
    shift,
    prompt_cached,
):
    with ctimed("pre pipe"):

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

        # Choose input image (prefer uploaded, else last output)
        pil_images = []
        if image is None:
            raise gr.Error("Please upload an image first.")
        if isinstance(image, Image.Image):
            pil_images.append(image.convert("RGB"))
        elif hasattr(image, "name"):
            pil_images.append(Image.open(image.name).convert("RGB"))

    # finetuner.enable()
    pipe.scheduler.config["base_shift"] = shift
    pipe.scheduler.config["max_shift"] = shift

    gallery_images = []
    
    for i in range(num_runs):
        result = pipe(
            image=pil_images,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            vae_image_override=1024 * 1024, #512 * 512,
            latent_size_override=1024 * 1024,
            prompt_cached=prompt_cached,
            return_dict=True,
        ).images[0]
        prompt_cached = True
        gallery_images.append(result)

        yield gallery_images, seed, prompt_cached


# --- UI ---

def reset_prompt_cache():
    return False

with gr.Blocks(theme=gr.themes.Citrus()) as demo:

    gr.Markdown("Qwen Image Demo")

    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Input Image", type="pil")
            prompt = gr.Textbox(label="Prompt", placeholder="Prompt", lines=2)

            num_runs = gr.Slider(label="Run Consecutively", minimum=0, maximum=100, step=1, value=4)

            run_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                prompt_cached = gr.Checkbox(label="Auto-Cached embeds", value=False)
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=2)
                shift = gr.Slider(label="Timestep Shift", minimum=0.0, maximum=4.0, step=0.1, value=2.0)

        with gr.Column():
            result = gr.Gallery(
                label="Output Image",
                interactive=False,
                columns=2,
                height=800,
                object_fit="scale-down",
            )
                    
    inputs = [
        image,
        prompt,
        num_runs,
        seed, 
        randomize_seed,
        num_inference_steps,
        shift,
        prompt_cached,
    ]
    outputs = [result, seed, prompt_cached]

    
    run_event = run_btn.click(
        fn=run_pipe, 
        inputs=inputs, 
        outputs=outputs
    )


    image.upload(
        fn=reset_prompt_cache,
        inputs=[],
        outputs=[prompt_cached],
    )

    prompt.input(
        fn=reset_prompt_cache,
        inputs=[],
        outputs=[prompt_cached],
    )

demo.launch()