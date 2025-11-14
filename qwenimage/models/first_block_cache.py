import functools
import unittest

import torch

from qwenimage.models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.models.transformer_qwenimage import QwenImageTransformer2DModel

from para_attn.first_block_cache import utils


def apply_cache_on_transformer(
    transformer: QwenImageTransformer2DModel,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            utils.CachedTransformerBlocks(
                transformer.transformer_blocks,
                transformer=transformer,
                return_hidden_states_first=False,
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(
    pipe: QwenImageEditPlusPipeline,
    *,
    shallow_patch: bool = False,
    **kwargs,
):
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs_):
            with utils.cache_context(utils.create_cache_context(**kwargs)):
                return original_call(self, *args, **kwargs_)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer)

    return pipe
