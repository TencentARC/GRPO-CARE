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
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, interleave_datasets
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainerRefEMA
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import random
from functools import partial
import math


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    bonus_coefficient: Optional[float] = field(
        default=1.0,
        metadata={"help": "The coefficient for consistency bonus."}
    )
    freeze_visual_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the visual encoder."},    
    )
    use_care: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use consistency-aware reward enhancement."},    
    )
    confidence_upper_bound: Optional[float] = field(
        default=None,
        metadata={"help": "The upper bound value for clamping reference likelihood."},    
    )
    ref_ema_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Decay used for updating the reference model using EMA"},    
    )
    ref_ema_update_every: Optional[float] = field(
        default=None,
        metadata={"help": "Step interval used for updating the reference model using EMA"},    
    )
    customized_top_p: Optional[float] = field(
        default=0.95,
        metadata={"help": "The top_p used for online sampling"},    
    )
    max_gen_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum generation length for control"},    
    )
    min_gen_len: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum generation length for control"},    
    )
    consistency_margin: Optional[float] = field(
        default=None,
        metadata={"help": "Do not punish the samples with reference likelihoods within a given margin below the mean confidence"},    
    )
   


def accuracy_reward(prompts, completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    
    if 'problem_id' in kwargs:
        problem_id = kwargs['problem_id']
    else:
        problem_id = kwargs['sample_id']

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for prompt, content, sol in zip(prompts, contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    elif math.isinf(out_number) or math.isnan(out_number):
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                elif math.isinf(out_number) or math.isnan(out_number):
                    reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Problem ID: {problem_id}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

        
    def make_conversation_image_and_video(example):
        if ("problem_type" in example) and (example["problem_type"] is not None):
            if example["problem_type"] == 'multiple choice':
                question = example['problem'] + "Options:\n"
                for op in example["options"]:
                    question += op + "\n"
            else:
                question = example['problem']
            
            msg ={
                "prompt": 
                [{
                        "role": "user",
                        "content": [
                            {
                                "type": example['data_type'],
                                example['data_type']: os.path.join(os.environ.get('VIDEO_R1_DATA_ROOT'), example['path'][2:])
                            },
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                            }
                        ]
                    }]
                }
        else:
            # process SEED-Bench-R1 training data
            if 'golden_choice_idx' not in example:
                negative_answers = random.sample(example["negative_answers"], 3)
                options = negative_answers + [example["answer"]]
            else:
                options = [example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']]

            random.shuffle(options)
            answer_index = options.index(example["answer"])
            question = example['question'] + "Options:\n" + \
                "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"


            solution = f"<answer>{chr(65 + answer_index)}</answer>"
            

            content = []
            if len(example['task_progress_metadata']) > 0:
                video_path = os.path.join(os.environ.get('SEED_BENCH_R1_DATA_ROOT'), 'videos', example['video_source'], example['video_basename'])
                content.append({"type": "video", "video": video_path})

            image_path = os.path.join(os.environ.get('SEED_BENCH_R1_DATA_ROOT'), 'images', example['video_source'], example['current_observation_basename'])
            content.extend([
                {"type": "image", "image": image_path},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']},
            ])

            msg = {
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                'solution': solution,
                'problem_type': 'multiple choice'
            }
        return msg

    dataset = dataset.map(make_conversation_image_and_video)


    trainer_cls = Qwen2VLGRPOTrainerRefEMA


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
