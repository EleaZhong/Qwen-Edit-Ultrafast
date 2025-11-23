

from collections import OrderedDict
from PIL import Image
from torchao import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, ModuleFqnToConfig, PerRow
from torchao.utils import get_model_size_in_bytes
from qwenimage.debug import ftimed, print_first_param
from qwenimage.experiments.experiments_qwen import ExperimentRegistry, QwenBaseExperiment
from qwenimage.models.attention_processors import QwenDoubleStreamAttnProcessorFA3
from qwenimage.optimization import simple_quantize_model


# ============================
# LINEAR / WEIGHTED LAYERS
# ============================

# ---- Vision branch ----

# Conv3d patch embed (often quantized like Linear)
VISION_PATCH_EMBED_LINEAR_REGEX = r"re:^model\.visual\.patch_embed\.proj$"

# Vision attention QKV and output projections (Linear)
VISION_ATTENTION_QKV_LINEAR_REGEX = (
    r"re:^model\.visual\.blocks\.\d+\.attn\.qkv$"
)
VISION_ATTENTION_OUT_LINEAR_REGEX = (
    r"re:^model\.visual\.blocks\.\d+\.attn\.proj$"
)

# Vision MLP projections (all Linear)
VISION_MLP_LINEAR_REGEX = (
    r"re:^model\.visual\.blocks\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
)

# Vision patch merger MLP (Sequential: indices 0 and 2 are Linear)
VISION_MERGER_MLP_LINEAR_REGEX = (
    r"re:^model\.visual\.merger\.mlp\.(0|2)$"
)


# ---- Text / language branch ----

# Token embedding (optional: treat as linear for quantization)
TEXT_EMBED_LINEAR_REGEX = r"re:^model\.language_model\.embed_tokens$"

# Text attention Q, K, V, O projections (Linear)
TEXT_ATTENTION_QKV_LINEAR_REGEX = (
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)$"
)
TEXT_ATTENTION_OUT_LINEAR_REGEX = (
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.o_proj$"
)

# Text MLP projections (all Linear)
TEXT_MLP_LINEAR_REGEX = (
    r"re:^model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
)

# LM head (Linear classifier / output proj)
LM_HEAD_LINEAR_REGEX = r"re:^lm_head$"



VISION_FIRST_BLOCK_REGEX = r"re:^model\.visual\.blocks\.0\..*$"
VISION_LAST_BLOCK_REGEX  = r"re:^model\.visual\.blocks\.31\..*$"

TEXT_FIRST_LAYER_REGEX   = r"re:^model\.language_model\.layers\.0\..*$"
TEXT_LAST_LAYER_REGEX    = r"re:^model\.language_model\.layers\.27\..*$"



@ExperimentRegistry.register(name="qwen_te_int8wo")
class Qwen_te_int8wo(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        simple_quantize_model(self.pipe.text_encoder, "int8wo")

@ExperimentRegistry.register(name="qwen_te_int4wo")
class Qwen_te_int4wo(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        simple_quantize_model(self.pipe.text_encoder, "int4wo")

@ExperimentRegistry.register(name="qwen_te_fp8row")
class Qwen_te_fp8row(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        simple_quantize_model(self.pipe.text_encoder, "fp8row")

@ExperimentRegistry.register(name="qwen_te_int4wo_qkv")
class Qwen_te_int4wo_qkv(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (VISION_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                ("_default",Int8WeightOnlyConfig(),),
            ])
        )
        print(f"original model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")
        quantize_(self.pipe.text_encoder, module_fqn_to_config)
        print_first_param(self.pipe.text_encoder)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")


@ExperimentRegistry.register(name="qwen_te_int4wo_linear")
class Qwen_te_int4wo_linear(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (VISION_PATCH_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (VISION_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (VISION_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (VISION_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (VISION_MERGER_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (TEXT_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (LM_HEAD_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                ("_default",Int8WeightOnlyConfig(),),
            ])
        )
        print(f"original model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")
        quantize_(self.pipe.text_encoder, module_fqn_to_config)
        print_first_param(self.pipe.text_encoder)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")


@ExperimentRegistry.register(name="qwen_te_int4wo_linear_nofirstlast")
class Qwen_te_int4wo_linear_nofirstlast(QwenBaseExperiment):
    @ftimed
    def optimize(self):
        module_fqn_to_config = ModuleFqnToConfig(
            OrderedDict([
                (VISION_FIRST_BLOCK_REGEX, None),
                (VISION_LAST_BLOCK_REGEX, None),
                (TEXT_FIRST_LAYER_REGEX, None),
                (TEXT_LAST_LAYER_REGEX, None),
                (VISION_PATCH_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (VISION_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (VISION_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (VISION_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (VISION_MERGER_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (TEXT_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                # (TEXT_MLP_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                (LM_HEAD_LINEAR_REGEX,Int4WeightOnlyConfig(),),
                ("_default",Int8WeightOnlyConfig(),),
            ])
        )
        print(f"original model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")
        quantize_(self.pipe.text_encoder, module_fqn_to_config)
        print_first_param(self.pipe.text_encoder)
        print(f"quantized model size: {get_model_size_in_bytes(self.pipe.text_encoder) / 1024 / 1024} MB")


def quantize_text_encoder_int4wo_linear(model):
    module_fqn_to_config = ModuleFqnToConfig(
        OrderedDict([
            (VISION_PATCH_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (VISION_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (VISION_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (TEXT_EMBED_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (TEXT_ATTENTION_QKV_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (TEXT_ATTENTION_OUT_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            (LM_HEAD_LINEAR_REGEX,Int4WeightOnlyConfig(),),
            ("_default",Int8WeightOnlyConfig(),),
        ])
    )
    print(f"original model size: {get_model_size_in_bytes(model) / 1024 / 1024} MB")
    quantize_(model, module_fqn_to_config)
    print_first_param(model)
    print(f"quantized model size: {get_model_size_in_bytes(model) / 1024 / 1024} MB")
