# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
cd ../qwen-vl-utils
pip install -e .[decord]

pip install torchvision
pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

pip install nltk
pip install rouge_score
pip install deepspeed

# fix transformers version
pip install transformers[torch]==4.51.3

