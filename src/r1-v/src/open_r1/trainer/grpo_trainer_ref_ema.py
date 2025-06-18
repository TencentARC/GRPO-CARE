# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # Trainer,
    TrainerCallback,
    is_wandb_available,
)
from .ema_trainer import EmaTrainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

import copy

import gc

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



class Qwen2VLGRPOTrainerRefEMA(EmaTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        # second_train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        
        if script_args.freeze_visual_encoder:
            print("Freeze parameters of visual encoder!")
            for n, p in model.visual.named_parameters():
                p.requires_grad_(False)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if args.beta > 0 or script_args.use_care:
            if is_deepspeed_zero3_enabled():
                if "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif peft_config is None:
                # If PEFT configuration is not provided, create a reference model based on the initial model.
                self.ref_model = create_reference_model(model)
            else:
                # If PEFT is used, the reference model is not needed since the adapter can be disabled
                # to revert to the initial model.
                self.ref_model = None
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=script_args.customized_top_p,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        # self.shuffled_num_generations = self.num_generations // 2
        # self.shuffled_generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,
        #     top_p=script_args.customized_top_p,  
        #     temperature=1, # HACK
        #     num_return_sequences=self.shuffled_num_generations,
        #     pad_token_id=pad_token_id,
        # )
        
        # self.dummy_generation_config = GenerationConfig(
        #     max_new_tokens=1,
        #     do_sample=True,
        #     top_p=script_args.customized_top_p,  
        #     temperature=1, # HACK
        #     num_return_sequences=1,
        #     pad_token_id=pad_token_id,
        # )
        self.len_control = script_args.len_control
        self.beta = args.beta
        # self.remove_length_bias = script_args.remove_length_bias
        # self.remove_difficulty_bias = script_args.remove_difficulty_bias
        # self.kl_strategy = script_args.kl_strategy
        # self.kl_lower_bound = script_args.kl_lower_bound
        # self.soft_prm = script_args.soft_prm
        self.use_care = script_args.use_care # NOTE: replace soft_prm with soft_prm
        # self.soft_prm_upper_bound = script_args.soft_prm_upper_bound
        self.confidence_upper_bound = script_args.confidence_upper_bound # NOTE: replace soft_prm_upper_bound with confidence_upper_bound
        # self.soft_prm_multiply_to = script_args.soft_prm_multiply_to
        # self.soft_prm_margin = script_args.soft_prm_margin
        self.consistency_margin = script_args.consistency_margin # NOTE: replace soft_prm_margin with consistency_margin
        # self.strict_answer_mask = script_args.strict_answer_mask
        # self.acc_kl_only = script_args.acc_kl_only
        # self.better_acc_kl_only = script_args.better_acc_kl_only
        # self.balanced_acc_kl = script_args.balanced_acc_kl
        self.ref_ema_decay = script_args.ref_ema_decay
        self.ref_ema_update_every = script_args.ref_ema_update_every
        # self.prm_reward_coefficient = script_args.prm_reward_coefficient 
        self.bonus_coefficient = script_args.bonus_coefficient # NOTE: replace prm_reward_coefficient with bonus_coefficient

        self.max_gen_len = script_args.max_gen_len
        self.min_gen_len = script_args.min_gen_len

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            # second_train_dataset=second_train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        for x in inputs:
            for message in x["prompt"]:
                if isinstance(message["content"], list):
                    new_content = []
                    for ele in message["content"]:
                        new_ele = {k: v for k, v in ele.items() if v is not None}
                        new_content.append(new_ele)
                    message["content"] = new_content
        
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
          
        
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
               prompts[0][0]['content'][0]['image'] = os.path.join(os.environ.get('VIDEO_R1_DATA_ROOT'), 'Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png')
            elif inputs[0]['data_type'] == 'video':
               prompts[0][0]['content'][0]['video'] = os.path.join(os.environ.get('VIDEO_R1_DATA_ROOT'), 'LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4')
                
            image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)

        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            
        # if self.temporal and video_inputs:
        #     indices = torch.randperm(video_inputs[0].size(0))
        #     shuffled_video_inputs = [video_inputs[0][indices]]
        #     shuffled_prompt_inputs = self.processing_class(
        #         text=copy.deepcopy(prompts_text),
        #         images=image_inputs,
        #         videos=shuffled_video_inputs,
        #         return_tensors="pt",
        #         padding=True,
        #         padding_side="left",
        #         add_special_tokens=False,
        #     )
        #     shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
        #     shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
        #     if self.max_prompt_length is not None:
        #         shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
        #         shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        

        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in prompt_inputs:
                prompt_inputs[k] = prompt_inputs[k].repeat(len(prompt_completion_ids), 1)
        
        if 'second_per_grid_ts' in prompt_inputs:
            del prompt_inputs["second_per_grid_ts"]
        
        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)

        gc.collect()
        torch.cuda.empty_cache()
        
        if self.beta > 0 or self.use_care:
            with torch.inference_mode():
                try:
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
                except Exception as e:
                    print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                    # ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
                
            # Compute the KL divergence between the model and the reference model
            
            x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
            per_token_kl = torch.exp(x_clamped) - x_clamped - 1


        gc.collect()
        torch.cuda.empty_cache()
        
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            
            reward_kwargs['batch_completion_ids'] = [tuple(c[m].cpu().numpy().tolist()) for c, m in zip(completion_ids, completion_mask.bool())]

            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        rewards = rewards_per_func.sum(dim=1)
        
        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    if (self.max_gen_len is None) and (self.min_gen_len is None):
                        if 320 <= lenth_list[idx] <= 512:
                            rewards[idx] += 0.2
                    elif (self.max_gen_len is None) and (self.min_gen_len <= lenth_list[idx]):
                        rewards[idx] += 0.2
                    elif (self.min_gen_len is None) and (lenth_list[idx] <= self.max_gen_len):
                        rewards[idx] += 0.2
                    elif self.min_gen_len <= lenth_list[idx] <= self.max_gen_len:
                        rewards[idx] += 0.2


        # Get masks for the thinking process and the final answer
        def get_thinking_answer_masks(prompt_completion_ids, prompt_length, sequence_indices, completion_mask):
            answer_indices = []
            for completion in prompt_completion_ids:
                completion = completion[completion!=self.processing_class.tokenizer.pad_token_id]
                completion_len = completion.shape[-1]
                completion_tokens = self.processing_class.tokenizer.convert_ids_to_tokens(completion)
                answer_start_location = None
                for i in range(len(completion_tokens)):
                    if 'answer' in completion_tokens[-i]:
                        expanded_token = ''.join(completion_tokens[-i-1:-i+2])
                        if '<answer>' in expanded_token:
                            answer_start_location = -i
                            break

                if answer_start_location is not None:
                    completion_cot_only = completion[:answer_start_location+1]
                    if completion_cot_only.shape[-1] > prompt_length:
                        answer_indices.append(completion_len+answer_start_location)
                    else: # fail to generate the final answer in the correct format
                        answer_start_location = None
                        answer_indices.append(completion_len)
                else:
                    answer_indices.append(completion_len)

            answer_indices = torch.tensor(answer_indices, dtype=torch.long, device=device)
            answer_indices = answer_indices - prompt_length
            mask_thinking = (sequence_indices <= answer_indices.unsqueeze(1)).int()

            true_answer_end_idx = torch.max(answer_indices+1, eos_idx-3)
            mask_answer = (sequence_indices > answer_indices.unsqueeze(1)) * (sequence_indices < true_answer_end_idx.unsqueeze(1))

            return mask_thinking, mask_answer

        if self.use_care:
            mask_thinking, mask_answer = get_thinking_answer_masks(prompt_completion_ids, prompt_length, sequence_indices, completion_mask)
            
            accuracy_rewards = rewards_per_func[:, 0]
            avg_ref_logps_answer = (ref_per_token_logps.detach() * mask_answer).sum(dim=1) / (mask_answer.sum(dim=1) + 1e-6)
            avg_ref_ps_answer = torch.exp(avg_ref_logps_answer)
            orig_avg_ref_ps_answer = avg_ref_ps_answer
            if self.confidence_upper_bound is not None:
                avg_ref_ps_answer = torch.clamp(avg_ref_ps_answer, max=self.confidence_upper_bound)
            bonus_rewards = torch.zeros_like(accuracy_rewards)

            # Relative High-Accuracy Trajectory Selection
            mean_accuracy_rewards = accuracy_rewards.mean()
            accurate_mask = (accuracy_rewards > 0.1) * (accuracy_rewards >= mean_accuracy_rewards)

            # Relative Consistency Evaluation
            accurate_confidence = avg_ref_ps_answer[accurate_mask]
            mean_selected_confidence = accurate_confidence.mean()
            if self.consistency_margin is None:
                confident_mask = (avg_ref_ps_answer >= mean_selected_confidence)
            else:
                confident_mask = (avg_ref_ps_answer >= (mean_selected_confidence-self.consistency_margin))

            # Enhanced Reward Calculation
            bonus_mask = accurate_mask * confident_mask
            bonus_rewards[bonus_mask] = accuracy_rewards[bonus_mask]
            bonus_rewards = bonus_rewards * self.bonus_coefficient
            rewards += bonus_rewards

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("CONFIDENCE_LOG_PATH")
                with open(log_path, "a", encoding="utf-8") as f:
                    if 'problem_id' in inputs[0]:
                        problem_id = inputs[0]['problem_id']
                    else:
                        problem_id = inputs[0]['sample_id']
                    solution = inputs[0]['solution']
                        
                    for prompt, cur_completion, cur_reward, cur_accuracy_reward, cur_bonus_reward_for_prm, cur_avg_ref_ps_answer in zip(
                        prompts, completions, rewards, accuracy_rewards, bonus_rewards, orig_avg_ref_ps_answer):
                        f.write(f"------------- Confidence: {cur_avg_ref_ps_answer} -------------\n")
                        f.write(f"Problem ID: {problem_id}\n")
                        if mean_selected_confidence is not None:
                            f.write(f"Mean Confidence: {mean_selected_confidence.item()}\n")
                        f.write(f"Reward: {cur_reward}\n")
                        f.write(f"Accuracy Reward: {cur_accuracy_reward}\n")
                        f.write(f"Bonus Reward: {cur_bonus_reward_for_prm}\n")
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Content: {cur_completion}\n")
                        f.write(f"Solution: {solution}\n")


        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        if self.beta == 0:
            per_token_loss = -per_token_loss
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        else:
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            
                      
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        if self.use_care:
            bonus_rewards = self.accelerator.gather_for_metrics(bonus_rewards)
            self._metrics["rewards/bonus_rewards"].append(bonus_rewards.mean().item())
            
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        try:
            num_devices = gathered_rewards.size(0) // self.num_generations 
            rewards_per_device = gathered_rewards.view(-1, self.num_generations)
            wrong_devices = (rewards_per_device <= 1).all(dim=1)
            wrong_ratio = wrong_devices.sum().item() / num_devices
            
            correct_devices = (rewards_per_device >= 2).all(dim=1)
            correct_ratio = correct_devices.sum().item() / num_devices
            
            self._metrics["all_wrong"].append(wrong_ratio)
            self._metrics["all_correct"].append(correct_ratio)
        except Exception as e:
            print(e)
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if self.beta > 0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
            

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
