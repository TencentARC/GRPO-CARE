export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${OUTPUT_DIR}/debug_log_rollouts.txt"
export CONFIDENCE_LOG_PATH="${OUTPUT_DIR}/debug_log_confidence.txt"
export SEED_BENCH_R1_DATA_ROOT=[your path to seed-bench-r1 data]

QWEN_PATH="${PROJECT_ROOT}/ckpt/Qwen2-VL-7B-Instruct"
HF_DATASET="${SEED_BENCH_R1_DATA_ROOT}/annotations/training_6k.jsonl"


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --deepspeed configs/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 500 \
    --beta 0 \
    --max_grad_norm 5 \
    --save_only_model true \
    --use_care true \
    --ref_ema_decay 0.995 \
    --ref_ema_update_every 10 \
    --bonus_coefficient 0.5 \
    --confidence_upper_bound 0.95 \
    --num_generations 8 


<<COMMENT
export PROJECT_ROOT=[your path to grpo-care root]
export RUN_NAME="Qwen2.5-VL-7B-GRPO-CARE-Margin0-SEED-Bench-R1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate grpo-care
export OUTPUT_DIR="${PROJECT_ROOT}/ckpt/${RUN_NAME}"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
cd ${PROJECT_ROOT}
nohup bash scripts/run_grpo_care_margin0_seed_bench_r1.sh > scripts/run_grpo_care_margin0_seed_bench_r1.log 2>&1 &
tail -f scripts/run_grpo_care_margin0_seed_bench_r1.log
COMMENT