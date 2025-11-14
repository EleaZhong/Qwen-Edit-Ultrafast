import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import os

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
from spaces.zero.torch.aoti import ZeroGPUCompiledModel, ZeroGPUWeights
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig, Int8DynamicActivationInt4WeightConfig, Int8DynamicActivationInt8WeightConfig, quantize_
from torchao.quantization import Int8WeightOnlyConfig
import spaces
import torch
from torch.utils._pytree import tree_map
from torchao.utils import get_model_size_in_bytes

from qwenimage.debug import ftimed, print_first_param
from qwenimage.models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.models.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.models.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
from qwenimage.experiment import AbstractExperiment, ExperimentConfig
from qwenimage.debug import ProfileSession, ftimed
from qwenimage.optimization import INDUCTOR_CONFIGS, TRANSFORMER_DYNAMIC_SHAPES, aoti_apply, drain_module_parameters, optimize_pipeline_
from qwenimage.prompt import build_camera_prompt


class ExperimentRegistry:
    registry = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(experiment_class):
            if name in cls.registry:
                raise ValueError(f"Experiment '{name}' is already registered")
            cls.registry[name] = experiment_class
            experiment_class.registry_name = name
            return experiment_class
        return decorator
    
    @classmethod
    def get(cls, name: str):
        if name not in cls.registry:
            raise KeyError(f"{name} not in {list(cls.registry.keys())}")
        return cls.registry[name]
    
    @classmethod
    def keys(cls):
        return list(cls.registry.keys())
    
    @classmethod
    def dict(cls):
        return dict(cls.registry)


class PipeInputs:
    images = Path("scripts/assets/").iterdir()

    camera_params = {
        "rotate_deg": [-90,-45,0,45,90],
        "move_forward": [0,5,10],
        "vertical_tilt": [-1,0,1],
        "wideangle": [True, False]
    }
    
    def __init__(self, seed=42):
        self.seed=seed
        param_keys = list(self.camera_params.keys())
        param_values = list(self.camera_params.values())
        param_keys.append("image")
        param_values.append(self.images)
        self.total_inputs = []
        for comb in itertools.product(*param_values):
            inp = {key:val for key,val in zip(param_keys, comb)}
            self.total_inputs.append(inp)
        print(f"{len(self.total_inputs)} input combinations")
        random.seed(seed)
        random.shuffle(self.total_inputs)
        self.generator = None

    def __len__(self):
        return len(self.total_inputs)

    def __getitem__(self, ind):
        if self.generator is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        inputs = self.total_inputs[ind]
        cam_prompt_params = {k:v for k,v in inputs.items() if k in self.camera_params}
        prompt = build_camera_prompt(**cam_prompt_params)
        image = [Image.open(inputs["image"]).convert("RGB")]
        return {
            "image": image,
            "prompt": prompt,
            "generator": self.generator,
            "num_inference_steps": 4,
            "true_cfg_scale": 1.0,
            "num_images_per_prompt": 1,
        }



@ExperimentRegistry.register(name="qwen_base")
class QwenBaseExperiment(AbstractExperiment):
    def __init__(self, config: ExperimentConfig | None = None, pipe_inputs: PipeInputs | None = None):
        self.config = config if config is not None else ExperimentConfig()
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        self.profile_report = ProfileSession().start()
        self.pipe_inputs = pipe_inputs if pipe_inputs is not None else PipeInputs()
    
    @ftimed
    def load(self):
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"experiment load cuda: {torch.cuda.is_available()=}")

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", 
            transformer=QwenImageTransformer2DModel.from_pretrained(
                "linoyts/Qwen-Image-Edit-Rapid-AIO", 
                subfolder='transformer',
                torch_dtype=dtype,
                device_map=device),
            torch_dtype=dtype,
        ).to(device)

        pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles", 
            weight_name="镜头转换.safetensors", adapter_name="angles"
        )

        pipe.set_adapters(["angles"], adapter_weights=[1.])
        pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
        pipe.unload_lora_weights()
        self.pipe = pipe

    @ftimed
    def optimize(self):
        pass
    
    @ftimed
    def run_once(self, *args, **kwargs):
        return self.pipe(*args, **kwargs).images[0]

    def run(self):
        output_save_dir = self.config.report_dir / f"{self.config.name}_outputs"
        output_save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.config.iterations):
            inputs = self.pipe_inputs[i]
            output = self.run_once(**inputs)
            output.save(output_save_dir / f"{i:03d}.jpg")

    def report(self):
        print(self.profile_report)
        
        raw_data = dict(self.profile_report.recorded_times)
        with open(self.config.report_dir/  f"{self.config.name}_raw.json", "w") as f:
            json.dump(raw_data, f, indent=2)
        
        data = []
        for name, times in self.profile_report.recorded_times.items():
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            size = len(times)
            data.append({
                'name': name,
                'mean': mean,
                'std': std,
                'len': size
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.config.report_dir/f"{self.config.name}.csv")
        return df, raw_data
    
    def cleanup(self):
        del self.pipe.transformer
        del self.pipe

@ExperimentRegistry.register(name="qwen_lightning_lora")
class Qwen_Lightning_Lora(QwenBaseExperiment):
    @ftimed
    def load(self):
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Scheduler configuration for Lightning
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation 
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config) # TODO: check scheduler sync issue mentioned by https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/

        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", 
            transformer=QwenImageTransformer2DModel.from_pretrained( # use our own model
                "Qwen/Qwen-Image-Edit-2509",
                subfolder='transformer',
                torch_dtype=dtype,
                device_map=device
            ),
            scheduler=scheduler,
            torch_dtype=dtype,
        ).to(device)

        pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles", 
            weight_name="镜头转换.safetensors",
            adapter_name="angles"
        )

        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", 
            weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            adapter_name="lightning",
        )

        pipe.set_adapters(["angles", "lightning"], adapter_weights=[1.25, 1.])
        pipe.fuse_lora(adapter_names=["angles", "lightning"], lora_scale=1.0)
        pipe.unload_lora_weights()
        self.pipe = pipe


@ExperimentRegistry.register(name="qwen_lightning_lora_3step")
class Qwen_Lightning_Lora_3step(Qwen_Lightning_Lora):
    @ftimed
    def run_once(self, *args, **kwargs):
        kwargs["num_inference_steps"] = 3
        return self.pipe(*args, **kwargs).images[0]

@ExperimentRegistry.register(name="qwen_base_3step")
class Qwen_Base_3step(QwenBaseExperiment):
    @ftimed
    def run_once(self, *args, **kwargs):
        kwargs["num_inference_steps"] = 3
        return self.pipe(*args, **kwargs).images[0]

@ExperimentRegistry.register(name="qwen_lightning_lora_2step")
class Qwen_Lightning_Lora_2step(Qwen_Lightning_Lora):
    @ftimed
    def run_once(self, *args, **kwargs):
        kwargs["num_inference_steps"] = 2
        return self.pipe(*args, **kwargs).images[0]

@ExperimentRegistry.register(name="qwen_base_2step")
class Qwen_Base_2step(QwenBaseExperiment):
    @ftimed
    def run_once(self, *args, **kwargs):
        kwargs["num_inference_steps"] = 2
        return self.pipe(*args, **kwargs).images[0]

@ExperimentRegistry.register(name="qwen_fuse")
class Qwen_Fuse(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.fuse_qkv_projections()


@ExperimentRegistry.register(name="qwen_fuse_aot")
class Qwen_Fuse_AoT(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.fuse_qkv_projections()
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=False,
            suffix="_fuse",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

@ExperimentRegistry.register(name="qwen_fa3_fuse")
class Qwen_FA3_Fuse(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.pipe.transformer.fuse_qkv_projections()


@ExperimentRegistry.register(name="qwen_fa3")
class Qwen_FA3(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

@ExperimentRegistry.register(name="qwen_aot")
class Qwen_AoT(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=False,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_fa3_aot")
class Qwen_FA3_AoT(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=False,
            suffix="_fa3",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_fa3_aot_int8")
class Qwen_FA3_AoT_int8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fa3",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_fp8")
class Qwen_fp8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        quantize_(self.pipe.transformer, Float8WeightOnlyConfig())


@ExperimentRegistry.register(name="qwen_int8")
class Qwen_int8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        quantize_(self.pipe.transformer, Int8WeightOnlyConfig())




@ExperimentRegistry.register(name="qwen_fa3_aot_fp8")
class Qwen_FA3_AoT_fp8(QwenBaseExperiment):
    @ftimed
    # @spaces.GPU()
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        pipe_kwargs={
            "image": [Image.new("RGB", (1024, 1024))],
            "prompt":"prompt",
            "num_inference_steps":4
        }
        suffix="_fa3"

        cache_compiled=self.config.cache_compiled
        
        transformer_pt2_cache_path = f"checkpoints/transformer_fp8{suffix}_archive.pt2"
        transformer_weights_cache_path = f"checkpoints/transformer_fp8{suffix}_weights.pt"

        print(f"original model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")
        quantize_(self.pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print_first_param(self.pipe.transformer)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")

        inductor_config = INDUCTOR_CONFIGS

        if os.path.isfile(transformer_pt2_cache_path) and cache_compiled:
            drain_module_parameters(self.pipe.transformer)
            zerogpu_weights = torch.load(transformer_weights_cache_path, weights_only=False)
            compiled_transformer = ZeroGPUCompiledModel(transformer_pt2_cache_path, zerogpu_weights)
        else:
            with spaces.aoti_capture(self.pipe.transformer) as call:
                self.pipe(**pipe_kwargs)

            dynamic_shapes = tree_map(lambda t: None, call.kwargs)
            dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES
            
            exported = torch.export.export(
                mod=self.pipe.transformer,
                args=call.args,
                kwargs=call.kwargs,
                dynamic_shapes=dynamic_shapes,
            )

            compiled_transformer = spaces.aoti_compile(exported, inductor_config)
            with open(transformer_pt2_cache_path, "wb") as f:
                f.write(compiled_transformer.archive_file.getvalue())
            torch.save(compiled_transformer.weights, transformer_weights_cache_path)


        aoti_apply(compiled_transformer, self.pipe.transformer)

# FA3_AoT_fp8_fuse
@ExperimentRegistry.register(name="qwen_fa3_aot_fp8_fuse")
class Qwen_FA3_AoT_fp8_fuse(QwenBaseExperiment):
    @ftimed
    # @spaces.GPU()
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.pipe.transformer.fuse_qkv_projections()

        pipe_kwargs={
            "image": [Image.new("RGB", (1024, 1024))],
            "prompt":"prompt",
            "num_inference_steps":4
        }
        suffix="_fa3_fuse"

        cache_compiled=self.config.cache_compiled
        
        transformer_pt2_cache_path = f"checkpoints/transformer_fp8{suffix}_archive.pt2"
        transformer_weights_cache_path = f"checkpoints/transformer_fp8{suffix}_weights.pt"

        print(f"original model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")
        quantize_(self.pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print_first_param(self.pipe.transformer)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")

        inductor_config = INDUCTOR_CONFIGS

        if os.path.isfile(transformer_pt2_cache_path) and cache_compiled:
            drain_module_parameters(self.pipe.transformer)
            zerogpu_weights = torch.load(transformer_weights_cache_path, weights_only=False)
            compiled_transformer = ZeroGPUCompiledModel(transformer_pt2_cache_path, zerogpu_weights)
        else:
            with spaces.aoti_capture(self.pipe.transformer) as call:
                self.pipe(**pipe_kwargs)

            dynamic_shapes = tree_map(lambda t: None, call.kwargs)
            dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES
            
            exported = torch.export.export(
                mod=self.pipe.transformer,
                args=call.args,
                kwargs=call.kwargs,
                dynamic_shapes=dynamic_shapes,
            )

            compiled_transformer = spaces.aoti_compile(exported, inductor_config)
            with open(transformer_pt2_cache_path, "wb") as f:
                f.write(compiled_transformer.archive_file.getvalue())
            torch.save(compiled_transformer.weights, transformer_weights_cache_path)


        aoti_apply(compiled_transformer, self.pipe.transformer)



# FA3_AoT_int8_fuse
@ExperimentRegistry.register(name="qwen_fa3_aot_int8_fuse")
class Qwen_FA3_AoT_int8_fuse(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.pipe.transformer.fuse_qkv_projections()
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fa3_fuse",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

# lightning_FA3_AoT_fp8_fuse

@ExperimentRegistry.register(name="qwen_lightning_fa3_aot_fp8_fuse")
class Qwen_lightning_FA3_AoT_fp8_fuse(Qwen_Lightning_Lora):
    @ftimed
    # @spaces.GPU()
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.pipe.transformer.fuse_qkv_projections()

        pipe_kwargs={
            "image": [Image.new("RGB", (1024, 1024))],
            "prompt":"prompt",
            "num_inference_steps":4
        }
        suffix="_fa3_fuse"

        cache_compiled=self.config.cache_compiled
        
        transformer_pt2_cache_path = f"checkpoints/transformer_fp8{suffix}_archive.pt2"
        transformer_weights_cache_path = f"checkpoints/transformer_fp8{suffix}_weights.pt"

        print(f"original model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")
        quantize_(self.pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print_first_param(self.pipe.transformer)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.transformer) / 1024 / 1024} MB")

        inductor_config = INDUCTOR_CONFIGS

        if os.path.isfile(transformer_pt2_cache_path) and cache_compiled:
            drain_module_parameters(self.pipe.transformer)
            zerogpu_weights = torch.load(transformer_weights_cache_path, weights_only=False)
            compiled_transformer = ZeroGPUCompiledModel(transformer_pt2_cache_path, zerogpu_weights)
        else:
            with spaces.aoti_capture(self.pipe.transformer) as call:
                self.pipe(**pipe_kwargs)

            dynamic_shapes = tree_map(lambda t: None, call.kwargs)
            dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES
            
            exported = torch.export.export(
                mod=self.pipe.transformer,
                args=call.args,
                kwargs=call.kwargs,
                dynamic_shapes=dynamic_shapes,
            )

            compiled_transformer = spaces.aoti_compile(exported, inductor_config)
            with open(transformer_pt2_cache_path, "wb") as f:
                f.write(compiled_transformer.archive_file.getvalue())
            torch.save(compiled_transformer.weights, transformer_weights_cache_path)


        aoti_apply(compiled_transformer, self.pipe.transformer)


# lightning_FA3_AoT_int8_fuse

@ExperimentRegistry.register(name="qwen_lightning_fa3_aot_int8_fuse")
class Qwen_Lightning_FA3_AoT_int8_fuse(Qwen_Lightning_Lora):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.pipe.transformer.fuse_qkv_projections()
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fa3_fuse",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_lightning_fa3_aot_int8_fuse_2step")
class Qwen_Lightning_FA3_AoT_int8_fuse_2step(Qwen_Lightning_FA3_AoT_int8_fuse):
    @ftimed
    def run_once(self, *args, **kwargs):
        kwargs["num_inference_steps"] = 2
        return self.pipe(*args, **kwargs).images[0]


@ExperimentRegistry.register(name="qwen_channels_last")
class Qwen_Channels_Last(QwenBaseExperiment):
    """
    This experiment is fully useless: channels last format only works with NCHW tensors, 
    i.e. 2D CNNs, transformer is 1D and vae is 3D, plus, for it to work the inputs need to 
    be converted in-pipe as well. left for reference.
    """
    @ftimed
    def optimize(self):
        self.pipe.vae = self.pipe.vae.to(memory_format=torch.channels_last)
        self.pipe.transformer = self.pipe.transformer.to(memory_format=torch.channels_last)