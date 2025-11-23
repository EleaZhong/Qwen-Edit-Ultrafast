

from collections import OrderedDict
from PIL import Image
from torchao import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, ModuleFqnToConfig, PerRow
from torchao.utils import get_model_size_in_bytes
from qwenimage.debug import ftimed, print_first_param
from qwenimage.experiments.experiments_qwen import ExperimentRegistry, QwenBaseExperiment
from qwenimage.models.attention_processors import QwenDoubleStreamAttnProcessorFA3
from qwenimage.optimization import optimize_pipeline_

# ModuleFqnToConfig

# @ExperimentRegistry.register(name="qwen_fa3_aot")
# class Qwen_FA3_AoT(QwenBaseExperiment):
#     @ftimed
#     def optimize(self):
#         self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
#         optimize_pipeline_(
#             self.pipe,
#             cache_compiled=self.config.cache_compiled,
#             quantize=False,
#             suffix="_fa3",
#             pipe_kwargs={
#                 "image": [Image.new("RGB", (1024, 1024))],
#                 "prompt":"prompt",
#                 "num_inference_steps":4
#             }
#         )


@ExperimentRegistry.register(name="qwen_fa3_aot_fp8wo")
class Qwen_FA3_AoT_fp8wo(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            quantize_config=Float8WeightOnlyConfig(),
            suffix="_fp8wo_fa3",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

@ExperimentRegistry.register(name="qwen_fa3_aot_int8wo")
class Qwen_FA3_AoT_int8wo(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            quantize_config=Int8WeightOnlyConfig(),
            suffix="_int8wo_fa3",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

@ExperimentRegistry.register(name="qwen_fa3_aot_fp8da")
class Qwen_FA3_AoT_fp8da(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            quantize_config=Float8DynamicActivationFloat8WeightConfig(),
            suffix="_fp8da_fa3",
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

@ExperimentRegistry.register(name="qwen_fa3_aot_int8da")
class Qwen_FA3_AoT_int8da(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_int8da_fa3",
            quantize_config=Int8DynamicActivationInt8WeightConfig(),
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

@ExperimentRegistry.register(name="qwen_fa3_aot_fp8darow")
class Qwen_FA3_AoT_fp8darow(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fp8dqrow_fa3",
            quantize_config=Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

ATTENTION_QKV_REGEX = "re:^transformer_blocks\\.\\d+\\.attn\\.(to_q|to_k|to_v|to_qkv|to_added_qkv|add_q_proj|add_k_proj|add_v_proj)$"
ATTENTION_QKV_REGEX = r"re:^transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_qkv|to_added_qkv|add_q_proj|add_k_proj|add_v_proj)$"
# Attention QKV projections (all Linear)
# Attention output projections (Linear)
ATTENTION_OUT_REGEX = r"re:^transformer_blocks\.\d+\.attn\.to_out\.0$"
ATTENTION_ADD_OUT_REGEX = r"re:^transformer_blocks\.\d+\.attn\.to_add_out$"

# Image modulation Linear layer
IMG_MOD_LINEAR_REGEX = r"re:^transformer_blocks\.\d+\.img_mod\.1$"

# Image MLP Linear layers
IMG_MLP_LINEAR1_REGEX = r"re:^transformer_blocks\.\d+\.img_mlp\.net\.0\.proj$"
IMG_MLP_LINEAR2_REGEX = r"re:^transformer_blocks\.\d+\.img_mlp\.net\.2$"

# Text modulation Linear layer
TXT_MOD_LINEAR_REGEX = r"re:^transformer_blocks\.\d+\.txt_mod\.1$"

# Text MLP Linear layers
TXT_MLP_LINEAR1_REGEX = r"re:^transformer_blocks\.\d+\.txt_mlp\.net\.0\.proj$"
TXT_MLP_LINEAR2_REGEX = r"re:^transformer_blocks\.\d+\.txt_mlp\.net\.2$"

# Top-level Linear layers (these were already fine)
IMG_IN_REGEX = r"re:^img_in$"
TXT_IN_REGEX = r"re:^txt_in$"
PROJ_OUT_REGEX = r"re:^proj_out$"

ATTN_LAST_LAYER = r"re:^transformer_blocks\.59\..*$"
ATTN_FIRST_LAYER = r"re:^transformer_blocks\.0\..*$"

@ExperimentRegistry.register(name="qwen_fa3_aot_qkvint4oint8")
class Qwen_FA3_AoT_qkvint4oint8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (ATTENTION_QKV_REGEX,Int4WeightOnlyConfig(),),
                ("_default",Int8WeightOnlyConfig(),),
            ])
        )
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_qkvint4oint8_fa3",
            quantize_config=module_fqn_to_config,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_fa3_aot_qkvfp8oint8")
class Qwen_FA3_AoT_qkvfp8oint8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (ATTENTION_QKV_REGEX,Float8DynamicActivationFloat8WeightConfig(),),
                ("_default",Int8WeightOnlyConfig(),),
            ])
        )
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_qkvfp8oint8_fa3",
            quantize_config=module_fqn_to_config,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )



@ExperimentRegistry.register(name="qwen_fa3_aot_fp8darow_nolast")
class Qwen_FA3_AoT_fp8darow_nolast(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (ATTN_LAST_LAYER, None),
                ("_default",Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),),
            ])
        )
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fp8darow_nolast_fa3",
            quantize_config=module_fqn_to_config,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )

def quantize_transformer_fp8darow_nolast(model):
    module_fqn_to_config = ModuleFqnToConfig(
        OrderedDict([
            (ATTN_LAST_LAYER, None),
            # ("_default",Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),),
            ("_default",Float8DynamicActivationFloat8WeightConfig(),),
        ])
    )
    print(f"original model size: {get_model_size_in_bytes(model) / 1024 / 1024} MB")
    quantize_(model, module_fqn_to_config)
    print_first_param(model)
    print(f"quantized model size: {get_model_size_in_bytes(model) / 1024 / 1024} MB")


@ExperimentRegistry.register(name="qwen_fa3_aot_fp8darow_nofirstlast")
class Qwen_FA3_AoT_fp8darow_nofirstlast(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (ATTN_LAST_LAYER, None),
                (ATTN_FIRST_LAYER, None),
                ("_default",Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),),
            ])
        )
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fp8darow_nofirstlast_fa3",
            quantize_config=module_fqn_to_config,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )


@ExperimentRegistry.register(name="qwen_fa3_aot_fp8darow_nolast_cint8")
class Qwen_FA3_AoT_fp8darow_nolast_cint8(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (ATTN_LAST_LAYER, None),
                (IMG_IN_REGEX, Int8WeightOnlyConfig()),
                (TXT_IN_REGEX, Int8WeightOnlyConfig()),
                (PROJ_OUT_REGEX, Int8WeightOnlyConfig()),
                ("_default",Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),),
            ])
        )
        optimize_pipeline_(
            self.pipe,
            cache_compiled=self.config.cache_compiled,
            quantize=True,
            suffix="_fp8darow_nolast_cint8_fa3",
            quantize_config=module_fqn_to_config,
            pipe_kwargs={
                "image": [Image.new("RGB", (1024, 1024))],
                "prompt":"prompt",
                "num_inference_steps":4
            }
        )



