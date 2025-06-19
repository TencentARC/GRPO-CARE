import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info,set_fps_max_frames
import argparse
import numpy as np


QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

def make_conversation_seed_bench_r1(example):
    options = [
        example['choice_a'],
        example['choice_b'],
        example['choice_c'],
        example['choice_d'],
    ]

    answer_index = example['golden_choice_idx']
    question = example['question'] + "Options:\n" + \
                "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"
    solution = f"<answer>{answer_index}</answer>"

    content = []
    if len(example['task_progress_metadata']) > 0:
        video_path = os.path.join(os.environ.get('SEED_BENCH_R1_DATA_ROOT'), 'videos', f"{example['video_source']}", example['video_basename'])
        content.append({
            "type": "video", 
            "video": video_path,
        })

    image_path = os.path.join(os.environ.get('SEED_BENCH_R1_DATA_ROOT'), 'images', example['video_source'], example['current_observation_basename'])
    content.extend([
        {"type": "image", "image": image_path},
        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']},
    ])

    example.update(
        {
            'solution': solution,
            'problem_type': 'multiple choice'
        }
    )

    prompt = [
        {
            "role": "user",
            "content": content,
        },
    ]

    return prompt


BSZ = 4


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--output_dir', type=str, default=None, help="Path to save the evaluation results")
parser.add_argument('--eval_mode', type=str, default='seed_bench_r1', help="choose between seed_bench_r1 and general_video_bench")
parser.add_argument('--fps_max_frames', type=int, default=32)
args = parser.parse_args()

set_fps_max_frames(args.fps_max_frames)


MODEL_PATH = args.model_path
if args.output_dir is None:
    args.output_dir = MODEL_PATH


llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 1, "video": 1},
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


if args.eval_mode == "seed_bench_r1":
    dataset_names = ['seed_bench_r1_L1','seed_bench_r1_L2','seed_bench_r1_L3']
else:
    dataset_names = ['mmvu', 'videomme','videommmu', 'tempcompass', 'mvbench', 'vsibench']

for dataset_name in dataset_names:
    if ("Video-R1" in MODEL_PATH) and (dataset_name in ['mmvu']):
        sampling_params = SamplingParams(
            temperature=1,
            top_p=0.95,
            max_tokens=768,
            stop_token_ids=[],
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            max_tokens=1024,
            stop_token_ids=[],
        )
        

    OUTPUT_PATH = os.path.join(args.output_dir, f"eval_{args.fps_max_frames}frames_{dataset_name}_results.json")
    print(OUTPUT_PATH)
    if 'seed_bench_r1' in dataset_name:
        level = dataset_name.split('_')[-1]
        PROMPT_PATH = f"{os.environ.get('SEED_BENCH_R1_DATA_ROOT')}/annotations/validation_{level}.jsonl"
    else:
        PROMPT_PATH = f"./src/r1-v/Evaluation/eval_{dataset_name}.json"
    
    data = []
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }


    messages = []

    if 'seed_bench_r1' in dataset_name:
        for example in data:
            messages.append(make_conversation_seed_bench_r1(example))
            example['problem_type'] = 'multiple choice'
    else:
        for x in data:
            if x["problem_type"] == 'multiple choice':
                question = x['problem'] + "Options:\n"
                for op in x["options"]:
                    question += op + "\n"
            else:
                question = x['problem']
            input_text_final = QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x['problem_type']]
            msg = [{
                "role": "user",
                "content": [
                    {
                        "type": x['data_type'],
                        x['data_type']: x['path'],
                    },
                    {
                        "type": "text",
                        "text": input_text_final
                    }
                ]
            }]
            messages.append(msg)

    final_output = []
    rewards = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                for sample in final_output:
                    rewards.append(sample['reward'])
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")


    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()

    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    mean_acc = []
    mean_mra = [] 

    t = tqdm(range(start_idx, len(messages), BSZ), desc=f"Processing batches {dataset_name}")
    for i in t:
        batch_messages = messages[i:i + BSZ]
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            image_idx = 0
            video_idx = 0

            llm_inputs = []

            for idx, prompt in enumerate(prompts):
                sample_mm_data = {}
                sample_video_kw = {}

                for j in range(len(batch_messages[idx][0]['content'])):
                    mm_type = batch_messages[idx][0]['content'][j]['type']
                    if mm_type == 'image':
                        sample_mm_data["image"] = image_inputs[image_idx]
                        image_idx += 1
                    elif mm_type == 'video':
                        sample_mm_data["video"] = video_inputs[video_idx]
                        for key, value in video_kwargs.items():
                            sample_video_kw[key] = value[video_idx]
                        video_idx += 1
                        
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]

        except Exception as e:
            print(e)
            print('error')
            batch_output_text = ['<answer>error</answer>'] * BSZ
            continue

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample["correct"] = True if sample["reward"]==1.0 else False
            if sample['problem_type'] != 'regression':
                mean_acc.append(sample["reward"])
            else:
                mean_mra.append(sample["reward"])
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
            rewards.append(sample["reward"])
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

        t.update(1)
        t.set_postfix(reward_mean=np.mean(rewards))

    t.close()
    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0, 'mean_rewards': np.mean(rewards)}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")
