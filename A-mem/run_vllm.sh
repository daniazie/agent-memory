python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --dataset data/locomo10.json --output results_robust_qwen30b.json \
    --sglang_port 30000

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset data/locomo10.json --output results_robust_llama1b.json \
    --sglang_port 30000

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset data/locomo10.json --output results_robust_llama3b.json \
    --sglang_port 30000