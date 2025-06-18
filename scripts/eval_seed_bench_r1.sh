model_paths=(
    "./ckpt/Qwen2.5-VL-7B-GRPO-CARE-Margin0-SEED-Bench-R1/checkpoint-1500"
)


export DECORD_EOF_RETRY_MAX=20480
export SEED_BENCH_R1_DATA_ROOT=[your path to seed-bench-r1 data]



for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    while true; do
        python -u ./src/eval_bench.py --model_path "$model" --eval_mode "seed_bench_r1"  --fps_max_frames 16
        if [ $? -eq 0 ]; then
            echo "The script exits normally."
            break
        fi
        echo "The script abnormally exited and was re-executed..."
        sleep 1
    done
done



<<COMMENT
export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT=[your path to grpo-care root]
conda activate grpo-care
cd ${PROJECT_ROOT}
nohup bash scripts/eval_general_video_bench.sh > scripts/eval_general_video_bench.log 2>&1 &
tail -f scripts/eval_general_video_bench.log
COMMENT