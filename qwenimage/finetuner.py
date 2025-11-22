import os
from typing import Optional
import uuid
import hashlib

from peft import LoraConfig

from wandml.utils.debug import ftimed
from wandml.finetune.lora.lora import LoraFinetuner

class QwenLoraFinetuner(LoraFinetuner):

    @ftimed
    def load(self, load_path, lora_rank=16, lora_config:Optional[LoraConfig]=None):
        """
        Loads new lora on flux transformer if not loaded. Loads lora safetensors from load_path. Specify specific lora config using lora_rank or lora_config.
        """
        if "transformer" in self.modules:
            return super().load(load_path)
        
        if lora_config:
            self.foundation.transformer = self.add_module(
                "transformer",
                self.foundation.transformer,
                lora_config=lora_config
            )
        else:
            target_modules = [
                'to_q',
                'to_k',
                'to_v',
                'to_qkv',

                'add_q_proj',
                'add_k_proj',
                'add_v_proj',
                'to_added_qkv',

                'proj',
                'txt_in',
                'img_in',
                'txt_mod.1',
                'img_mod.1',
                'proj_out',
                'to_add_out',
                'to_out.0'
                'net.2',
                'linear',
                'linear_2',
                'linear_1',
            ]
            self.foundation.transformer = self.add_module(
                "transformer",
                self.foundation.transformer,
                target_modules=target_modules,
                lora_rank=lora_rank,
            )
            self.foundation.transformer.to(dtype=self.foundation.dtype)

        return super().load(load_path)
