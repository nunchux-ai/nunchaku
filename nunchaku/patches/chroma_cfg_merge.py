from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable

import numpy as np
import torch


def _cfg_merge_enabled() -> bool:
    value = os.getenv("CHROMA_CFG_MERGE")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "off", "no"}


def patch_chroma_pipeline_cfg_merge() -> bool:
    from diffusers.pipelines.chroma.pipeline_chroma import (
        XLA_AVAILABLE,
        ChromaPipeline,
        ChromaPipelineOutput,
        calculate_shift,
        retrieve_timesteps,
    )

    if getattr(ChromaPipeline, "_nunchaku_cfg_merge_patched", False):
        return False

    original_call = ChromaPipeline.__call__

    @torch.no_grad()
    @wraps(original_call)
    def patched_call(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 35,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        ip_adapter_image=None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        negative_ip_adapter_image=None,
        negative_ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if not _cfg_merge_enabled():
            return original_call(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        # Keep the patch intentionally conservative: CFG merge is only enabled
        # for the common text-only path. Unsupported features fall back.
        if any(
            x is not None
            for x in (
                ip_adapter_image,
                ip_adapter_image_embeds,
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
            )
        ):
            return original_call(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # If CFG is effectively disabled, use the upstream implementation.
        if not self.do_classifier_free_guidance:
            return original_call(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None

        (
            prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_text_ids,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if negative_prompt_embeds is None or negative_prompt_attention_mask is None:
            return original_call(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=prompt_attention_mask,
        )
        negative_attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=negative_prompt_attention_mask,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        merged_encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).contiguous()
        merged_attention_mask = torch.cat([negative_attention_mask, attention_mask], dim=0).contiguous()
        # Chroma text ids are sequence-position metadata shared by both branches.
        merged_text_ids = text_ids

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                merged_latents = torch.cat([latents, latents], dim=0)
                merged_timestep = torch.cat([timestep, timestep], dim=0)

                merged_noise_pred = self.transformer(
                    hidden_states=merged_latents,
                    timestep=merged_timestep / 1000,
                    encoder_hidden_states=merged_encoder_hidden_states,
                    txt_ids=merged_text_ids,
                    img_ids=latent_image_ids,
                    attention_mask=merged_attention_mask,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                neg_noise_pred, noise_pred = merged_noise_pred.chunk(2, dim=0)
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    if "negative_prompt_embeds" in callback_outputs:
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds")
                    if "prompt_attention_mask" in callback_outputs:
                        attention_mask = callback_outputs.pop("prompt_attention_mask")
                    if "negative_prompt_attention_mask" in callback_outputs:
                        negative_attention_mask = callback_outputs.pop("negative_prompt_attention_mask")

                    merged_encoder_hidden_states = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    ).contiguous()
                    merged_attention_mask = torch.cat([negative_attention_mask, attention_mask], dim=0).contiguous()

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm

                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ChromaPipelineOutput(images=image)

    ChromaPipeline._nunchaku_original_call = original_call
    ChromaPipeline.__call__ = patched_call
    ChromaPipeline._nunchaku_cfg_merge_patched = True
    return True
