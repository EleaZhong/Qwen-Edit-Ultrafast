
import os
from pathlib import Path
import warnings

from PIL import Image
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
import lpips
import torch
from safetensors.torch import load_file, save_model
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange

from qwenimage.datamodels import QwenConfig, QwenInputs
from qwenimage.debug import clear_cuda_memory, ctimed, ftimed, print_gpu_memory, texam
from qwenimage.experiments.quantize_text_encoder_experiments import quantize_text_encoder_int4wo_linear
from qwenimage.experiments.quantize_experiments import quantize_transformer_fp8darow_nolast
from qwenimage.loss import LossAccumulator
from qwenimage.models.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, QwenImageEditPlusPipeline, calculate_dimensions
from qwenimage.models.pipeline_qwenimage_edit_save_interm import QwenImageEditSaveIntermPipeline
from qwenimage.models.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.optimization import simple_quantize_model
from qwenimage.sampling import TimestepDistUtils
from wandml import WandModel
from wandml.core.logger import wand_logger
from wandml.trainers.experiment_trainer import ExperimentTrainer


class QwenImageFoundation(WandModel):
    SOURCE = "Qwen/Qwen-Image-Edit-2509"
    INPUT_MODEL = QwenInputs
    CACHE_DIR = "qwen_image_edit_2509"
    PIPELINE = QwenImageEditPlusPipeline

    serialize_modules = ["transformer"]

    def __init__(self, config:QwenConfig, device=None):
        super().__init__()
        self.config:QwenConfig = config
        self.dtype = torch.bfloat16
        if device is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = default_device
        else:
            self.device = device
        print(f"{self.device=}")

        pipe = self.PIPELINE.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", 
            transformer=QwenImageTransformer2DModel.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                subfolder='transformer',
                torch_dtype=self.dtype,
                device_map=self.device
            ),
            torch_dtype=self.dtype,
        )
        pipe = pipe.to(device=self.device, dtype=self.dtype)


        if config.load_multi_view_lora:
            pipe.load_lora_weights(
                "dx8152/Qwen-Edit-2509-Multiple-angles", 
                weight_name="镜头转换.safetensors", adapter_name="angles"
            )
            pipe.set_adapters(["angles"], adapter_weights=[1.])
            pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
            pipe.unload_lora_weights()
        
        self.pipe = pipe
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler

        self.vae.to(self.dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder_device = None
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.timestep_dist_utils = TimestepDistUtils(
            min_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            min_mu=self.scheduler.config.base_shift,
            max_mu=self.scheduler.config.max_shift,
            train_dist=self.config.train_dist,
            train_shift=self.config.train_shift,
            inference_dist=self.config.inference_dist,
            inference_shift=self.config.inference_shift,
            static_mu=self.config.static_mu,
            loss_weight_dist=self.config.loss_weight_dist,
        )

        if self.config.quantize_text_encoder:
            quantize_text_encoder_int4wo_linear(self.text_encoder)
        
        if self.config.quantize_transformer:
            quantize_transformer_fp8darow_nolast(self.transformer)


    def load(self, load_path):
        if not isinstance(load_path, Path): 
            load_path = Path(load_path)
        if not load_path.is_dir():
            raise ValueError(f"Expected {load_path=} to be a directory")
        for module_name in self.serialize_modules:
            model_state_dict = load_file(load_path / f"{module_name}.safetensors")
            missing, unexpected = getattr(self, module_name).load_state_dict(model_state_dict, strict=False, assign=True)
            if missing: 
                warnings.warn(f"{module_name} missing {missing}")
            if unexpected: 
                warnings.warn(f"{module_name} unexpected {unexpected}")

    def save(self, save_path, skip=False):
        if skip: return

        if not isinstance(save_path, Path): 
            save_path = Path(save_path)
        if not save_path.is_dir():
            raise ValueError(f"Expected {save_path=} to be a directory")

        save_path.mkdir(parents=True, exist_ok=True)

        for module_name in self.serialize_modules:
            save_model(getattr(self, module_name), save_path / f"{module_name}.safetensors")
            print(f"Saved {module_name} to {save_path}")
    
    def get_train_params(self):
        return [{"params": [p for p in self.transformer.parameters() if p.requires_grad]}]
    
    def pil_to_latents(self, images):
        image = self.pipe.image_processor.preprocess(images)

        h,w = image.shape[-2:]
        h_r, w_r = calculate_dimensions(self.config.vae_image_size, h/w)
        image = TF.resize(image, (h_r, w_r))

        print("pil_to_latents.image")
        texam(image)

        image = image.unsqueeze(2) # N, C, F=1, H, W
        image = image.to(device=self.device, dtype=self.dtype)
        latents = self.pipe.vae.encode(image).latent_dist.mode() # argmax

        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        latents = latents.squeeze(2)
        print("pil_to_latents.latents")
        texam(latents)
        return latents.to(dtype=self.dtype)

    def latents_to_pil(self, latents, h=None, w=None, with_grad=False):
        if not with_grad:
            latents = latents.clone().detach()
        if latents.dim() == 3:  # 1d latent
            if h is None or w is None:
                raise ValueError(f"auto unpack needs h,w, got {h=}, {w=}")
            latents = self.unpack_latents(latents, h=h, w=w)
        latents = latents.unsqueeze(2)

        latents = latents.to(self.dtype)
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        latents = latents.to(device=self.device, dtype=self.dtype)
        image = self.pipe.vae.decode(latents, return_dict=False)[0][:, :, 0] # F = 1

        if with_grad:
            texam(image, "latents_to_pil.image")
            return image

        image = self.pipe.image_processor.postprocess(image)
        return image

    @staticmethod
    def pack_latents(latents):
        packed = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return packed
    
    @staticmethod
    def unpack_latents(packed, h, w):
        latents = rearrange(packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=h, w=w)
        return latents

    @ftimed
    def offload_text_encoder(self, device=str|torch.device):
        if self.text_encoder_device == device:
            return
        print(f"Moving text encoder to {device}")
        self.text_encoder_device = device
        self.text_encoder.to(device)
        if device == "cpu" or device == torch.device("cpu"):
            print_gpu_memory(clear_mem="pre")

    @ftimed
    def preprocess_batch(self, batch):
        prompts = batch["text"]
        references = batch["reference"]

        h,w = references.shape[-2:]
        h_r, w_r = calculate_dimensions(CONDITION_IMAGE_SIZE, h/w)
        references = TF.resize(references, (h_r, w_r))

        print("preprocess_batch.references")
        texam(references)
        
        self.offload_text_encoder("cuda")

        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                prompts,
                references.mul(255), # scaled to RGB
                device="cuda",
                max_sequence_length = self.config.train_max_sequence_length,
            )
        prompt_embeds = prompt_embeds.cpu().clone().detach()
        prompt_embeds_mask = prompt_embeds_mask.cpu().clone().detach()


        batch["prompt_embeds"] = (prompt_embeds, prompt_embeds_mask)
        batch["reference"] = batch["reference"].cpu()
        batch["image"] = batch["image"].cpu()

        return batch

    @ftimed
    def single_step(self, batch) -> torch.Tensor:
        self.offload_text_encoder("cpu")

        if "prompt_embeds" not in batch:
            batch = self.preprocess_batch(batch)
        prompt_embeds, prompt_embeds_mask = batch["prompt_embeds"]
        prompt_embeds = prompt_embeds.to(device=self.device)
        prompt_embeds_mask = prompt_embeds_mask.to(device=self.device)

        images = batch["image"].to(device=self.device, dtype=self.dtype)
        x_0 = self.pil_to_latents(images).to(device=self.device, dtype=self.dtype)
        x_1 = torch.randn_like(x_0).to(device=self.device, dtype=self.dtype)
        seq_len = self.timestep_dist_utils.get_seq_len(x_0)
        batch_size = x_0.shape[0]
        t = self.timestep_dist_utils.get_train_t([batch_size], seq_len=seq_len).to(device=self.device, dtype=self.dtype)
        x_t = (1.0 - t) * x_0 + t * x_1
        x_t_1d = self.pack_latents(x_t)

        references = batch["reference"].to(device=self.device, dtype=self.dtype)
        print("references")
        texam(references)
        assert references.shape[0] == 1
        refs = self.pil_to_latents(references).to(device=self.device, dtype=self.dtype)
        refs_1d = self.pack_latents(refs)
        print("refs refs_1d")
        texam(refs)
        texam(refs_1d)

        inp_1d = torch.cat([x_t_1d, refs_1d], dim=1)
        print("inp_1d")
        texam(inp_1d)

        l_height, l_width = x_0.shape[-2:]
        ref_height, ref_width = refs.shape[-2:]
        img_shapes = [
            [
                (1, l_height // 2, l_width // 2),
                (1, ref_height // 2, ref_width // 2),
            ]
        ] * batch_size
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        image_rotary_emb = self.transformer.pos_embed(img_shapes, txt_seq_lens, device=x_0.device)

        v_pred_1d = self.transformer(
            hidden_states=inp_1d,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=t,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        v_pred_1d = v_pred_1d[:, : x_t_1d.size(1)]

        v_pred_2d = self.unpack_latents(v_pred_1d, h=l_height//2, w=l_width//2)
        v_gt_2d = x_1 - x_0

        if self.config.loss_weight_dist is not None:
            loss = F.mse_loss(v_pred_2d, v_gt_2d, reduction="none").mean(dim=[1,2,3])
            weights = self.timestep_dist_utils.get_loss_weighting(t)
            loss = torch.mean(loss * weights)
        else:
            loss = F.mse_loss(v_pred_2d, v_gt_2d, reduction="mean")
        
        return loss


    def base_pipe(self, inputs: QwenInputs) -> list[Image]:
        print(inputs)
        self.offload_text_encoder("cuda")
        image = inputs.image[0]
        w,h = image.size
        h_r, w_r = calculate_dimensions(self.config.vae_image_size, h/w)
        image = TF.resize(image, (h_r, w_r))
        inputs.image = [image]
        return self.pipe(**inputs.model_dump()).images
    


class QwenImageFoundationSaveInterm(QwenImageFoundation):
    PIPELINE = QwenImageEditSaveIntermPipeline

    def base_pipe(self, inputs: QwenInputs) -> list[Image]:
        print(inputs)
        image = inputs.image[0]
        w,h = image.size
        h_r, w_r = calculate_dimensions(self.config.vae_image_size, h/w)
        image = TF.resize(image, (h_r, w_r))
        inputs.image = [image]
        return self.pipe(**inputs.model_dump())
    

class QwenImageRegressionFoundation(QwenImageFoundation):
    def __init__(self, config:QwenConfig, device=None):
        super().__init__(config, device=device)
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device=self.device)
    
    def preprocess_batch(self, batch):
        return batch

    @ftimed
    def single_step(self, batch) -> torch.Tensor:
        self.offload_text_encoder("cpu")

        out_dict = batch["data"]
        assert len(out_dict) == 1
        out_dict = out_dict[0]

        prompt_embeds = out_dict["prompt_embeds"]
        prompt_embeds_mask = out_dict["prompt_embeds_mask"]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device=self.device, dtype=self.dtype)

        h_f16 = out_dict["height"] // 16
        w_f16 = out_dict["width"] // 16
        
        refs_1d = out_dict["image_latents"].to(device=self.device, dtype=self.dtype)
        t = out_dict["t"].to(device=self.device, dtype=self.dtype)
        x_0_1d = out_dict["output"].to(device=self.device, dtype=self.dtype)
        x_t_1d = out_dict["latents_start"].to(device=self.device, dtype=self.dtype)
        v_neg_1d = out_dict["noise_pred"].to(device=self.device, dtype=self.dtype)

        
        v_gt_1d = (x_t_1d - x_0_1d) / t

        inp_1d = torch.cat([x_t_1d, refs_1d], dim=1)
        print("inp_1d")
        texam(inp_1d)

        img_shapes = out_dict["img_shapes"]
        txt_seq_lens = out_dict["txt_seq_lens"]
        image_rotary_emb = self.transformer.pos_embed(img_shapes, txt_seq_lens, device=self.device)

        v_pred_1d = self.transformer(
            hidden_states=inp_1d,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=t,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        v_pred_1d = v_pred_1d[:, : x_t_1d.size(1)]


        split = batch["split"]
        step = batch["step"]
        if split == "train":
            loss_terms = self.config.train_loss_terms.model_dump()
        elif split == "validation":
            loss_terms = self.config.validation_loss_terms.model_dump()
        loss_accumulator = LossAccumulator(
            terms=loss_terms,
            step=step,
            split=split,
        )

        if loss_accumulator.has("mse"):
            if self.config.loss_weight_dist is not None:
                mse_loss = F.mse_loss(v_pred_1d, v_gt_1d, reduction="none").mean(dim=[1,2,3])
                weights = self.timestep_dist_utils.get_loss_weighting(t)
                mse_loss = torch.mean(mse_loss * weights)
            else:
                mse_loss = F.mse_loss(v_pred_1d, v_gt_1d, reduction="mean")
            loss_accumulator.accum("mse", mse_loss)
        
        if loss_accumulator.has("triplet"):
            # eps = 1e-6
            margin = loss_terms["triplet_margin"]
            # v_span = (v_gt_1d - v_neg_1d).pow(2).sum(dim=(-2,-1))

            # triplet_weight = (v_gt_1d - v_neg_1d).pow(2).mean(dim=(-2,-1))

            diffv_gt_pred = (v_gt_1d - v_pred_1d).pow(2).sum(dim=(-2,-1))
            diffv_neg_pred = (v_neg_1d - v_pred_1d).pow(2).sum(dim=(-2,-1))
            # diffv_gt_pred_reg = diffv_gt_pred # / (v_span + eps)
            # diffv_neg_pred_reg = diffv_neg_pred # / (v_span + eps)
            
            # texam(v_span, name="v_span")
            # texam(triplet_weight, name="triplet_weight")
            texam(diffv_gt_pred, name="diffv_gt_pred")
            texam(diffv_neg_pred, name="diffv_neg_pred")
            # texam(diffv_gt_pred_reg, name="diffv_gt_pred_reg")
            # texam(diffv_neg_pred_reg, name="diffv_neg_pred_reg")
            # texam(diffv_gt_pred_reg - diffv_neg_pred_reg, name="diffv_gt_pred_reg - diffv_neg_pred_reg")
            
            triplet_loss = F.relu(diffv_gt_pred - diffv_neg_pred + margin).mean()
            # 
            # triplet_loss = F.relu(diffv_gt_pred_reg - diffv_neg_pred_reg + margin).mean()
            # triplet_loss = torch.mean(triplet_loss_batched * triplet_weight)
            loss_accumulator.accum("triplet", triplet_loss)

        
        if loss_accumulator.has("negative_mse"):
            neg_mse_loss = -F.mse_loss(v_pred_1d, v_neg_1d, reduction="mean")
            loss_accumulator.accum("negative_mse", neg_mse_loss)


        if loss_accumulator.has("distribution_matching"):
            dm_v = (v_pred_1d - v_neg_1d + v_gt_1d).detach()
            dm_mse = F.mse_loss(v_pred_1d, dm_v, reduction="mean")
            loss_accumulator.accum("distribution_matching", dm_mse)
        
        if loss_accumulator.has("negative_exponential"):
            raise NotImplementedError()
        
        if loss_accumulator.has("pixel_lpips") or loss_accumulator.has("pixel_mse"):
            x_0_pred = x_t_1d - t * v_pred_1d
            pixel_values_x0_gt = self.latents_to_pil(x_0_1d, h=h_f16, w=w_f16, with_grad=True).detach()
            pixel_values_x0_pred = self.latents_to_pil(x_0_pred, h=h_f16, w=w_f16, with_grad=True)

            if loss_accumulator.has("pixel_lpips"):
                lpips_loss = self.lpips_fn(pixel_values_x0_gt, pixel_values_x0_pred)
                loss_accumulator.accum("pixel_lpips", lpips_loss)
            
            if loss_accumulator.has("pixel_mse"):
                pixel_mse_loss = F.mse_loss(pixel_values_x0_pred, pixel_values_x0_gt, reduction="mean")
                loss_accumulator.accum("pixel_mse", pixel_mse_loss)
        
        if loss_accumulator.has("adversarial"):
            raise NotImplementedError()


        loss = loss_accumulator.total

        logs = loss_accumulator.logs()
        wand_logger.log(logs, step=step, commit=False)
        wand_logger.log({
            "t": t.float().cpu().item()
        }, step=step, commit=False)


        if self.should_log_training(step):
            self.log_single_step_images(
                h_f16,
                w_f16,
                t,
                x_0_1d,
                x_t_1d,
                v_gt_1d,
                v_neg_1d,
                v_pred_1d,
                visualize_velocities=True,
            )
        
        return loss

    def should_log_training(self, step) -> bool:
        return (
            self.training # don't log when validating
            and ExperimentTrainer._is_step_trigger(step, self.config.log_batch_steps)
        )

    def log_single_step_images(
        self,
        h_f16,
        w_f16,
        t,
        x_0_1d,
        x_t_1d,
        v_gt_1d,
        v_neg_1d,
        v_pred_1d,
        visualize_velocities=True,
    ):
        x_0_pred = x_t_1d - t * v_pred_1d
        x_0_neg = x_t_1d - t * v_neg_1d
        x_0_recon = x_t_1d - t * v_gt_1d
        log_pils = {
            "x_t_1d": self.latents_to_pil(x_t_1d, h=h_f16, w=w_f16),
            "x_0": self.latents_to_pil(x_0_1d, h=h_f16, w=w_f16),
            "x_0_recon": self.latents_to_pil(x_0_recon, h=h_f16, w=w_f16),
            "x_0_pred": self.latents_to_pil(x_0_pred, h=h_f16, w=w_f16),
            "x_0_neg": self.latents_to_pil(x_0_neg, h=h_f16, w=w_f16),
        }
        if visualize_velocities:
            log_pils.update({
                "v_gt_1d": self.latents_to_pil(v_gt_1d, h=h_f16, w=w_f16),
                "v_pred_1d": self.latents_to_pil(v_pred_1d, h=h_f16, w=w_f16),
                "v_neg_1d": self.latents_to_pil(v_neg_1d, h=h_f16, w=w_f16),
            })
        
        wand_logger.log({
            "train_images": log_pils,
        }, commit=False)

        
    def base_pipe(self, inputs: QwenInputs) -> list[Image]:
        inputs.num_inference_steps = self.config.regression_base_pipe_steps  # override
        return super().base_pipe(inputs)