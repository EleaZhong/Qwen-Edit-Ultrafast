"""
"""
import os

from typing import Any
from typing import Callable
from typing import ParamSpec
from spaces.zero.torch.aoti import ZeroGPUCompiledModel, ZeroGPUWeights
from torchao.quantization import quantize_
from torchao.quantization import Int8WeightOnlyConfig
import spaces
import torch
from torch.utils._pytree import tree_map
from torchao.utils import get_model_size_in_bytes

from qwenimage.debug import ftimed, print_first_param


P = ParamSpec('P')


TRANSFORMER_IMAGE_SEQ_LENGTH_DIM = torch.export.Dim('image_seq_length')
TRANSFORMER_TEXT_SEQ_LENGTH_DIM = torch.export.Dim('text_seq_length')

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {
        1: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'encoder_hidden_states_mask': {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    'image_rotary_emb': ({
        0: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    }, {
        0: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    }),
}


INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}

def aoti_apply(
    compiled: ZeroGPUCompiledModel,
    module: torch.nn.Module,
    call_method: str = 'forward',
):
    setattr(module, call_method, compiled)
    drain_module_parameters(module)


def drain_module_parameters(module: torch.nn.Module):
    state_dict_meta = {name: {'device': tensor.device, 'dtype': tensor.dtype} for name, tensor in module.state_dict().items()}
    state_dict = {name: torch.nn.Parameter(torch.empty(tensor.size(), device='cpu')) for name, tensor in module.state_dict().items()}
    module.load_state_dict(state_dict, assign=True)
    for name, param in state_dict.items():
        meta = state_dict_meta[name]
        param.data = torch.Tensor([]).to(**meta)


@ftimed
def optimize_pipeline_(
        pipeline: Callable[P, Any],
        cache_compiled=True,
        quantize=True,
        inductor_config=None,
        suffix="",
        pipe_kwargs={}
    ):

    if quantize:
        transformer_pt2_cache_path = f"checkpoints/transformer_int8{suffix}_archive.pt2"
        transformer_weights_cache_path = f"checkpoints/transformer_int8{suffix}_weights.pt"
        print(f"original model size: {get_model_size_in_bytes(pipeline.transformer) / 1024 / 1024} MB")
        quantize_(pipeline.transformer, Int8WeightOnlyConfig())
        print_first_param(pipeline.transformer)
        print(f"quantized model size: {get_model_size_in_bytes(pipeline.transformer) / 1024 / 1024} MB")
    else:
        transformer_pt2_cache_path = f"checkpoints/transformer{suffix}_archive.pt2"
        transformer_weights_cache_path = f"checkpoints/transformer{suffix}_weights.pt"

    if inductor_config is None:
        inductor_config = INDUCTOR_CONFIGS

    if os.path.isfile(transformer_pt2_cache_path) and cache_compiled:
        drain_module_parameters(pipeline.transformer)
        zerogpu_weights = torch.load(transformer_weights_cache_path, weights_only=False)
        compiled_transformer = ZeroGPUCompiledModel(transformer_pt2_cache_path, zerogpu_weights)
    else:
        @spaces.GPU(duration=1500)
        def compile_transformer():
            with spaces.aoti_capture(pipeline.transformer) as call:
                pipeline(**pipe_kwargs)

            dynamic_shapes = tree_map(lambda t: None, call.kwargs)
            dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES
            
            exported = torch.export.export(
                mod=pipeline.transformer,
                args=call.args,
                kwargs=call.kwargs,
                dynamic_shapes=dynamic_shapes,
            )

            return spaces.aoti_compile(exported, inductor_config)
        compiled_transformer = compile_transformer()
        with open(transformer_pt2_cache_path, "wb") as f:
            f.write(compiled_transformer.archive_file.getvalue())
        torch.save(compiled_transformer.weights, transformer_weights_cache_path)


    aoti_apply(compiled_transformer, pipeline.transformer)

