export WORKING_DIR=$(pwd)
cd exp/a_mem

python run_amem.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset ../../data/locomo10.json --output ../../results/base/qwen4b-instruct_a-mem_results.json --batched_run \
    --quantization bitsandbytes --dtype bfloat16 --seed 42 # --cpu_offload_gb 8 # --gpu_memory_utilization 0.42
# $python run_amem.py --model Qwen/Qwen3-4B-Thinking-2507 --dataset ../../data/locomo10.json --output ../../results/base/qwen4b-thinking_a-mem_results.json --enable_thinking --batched_run \
#     --quantization bitsandbytes --dtype bfloat16 --seed 42 # --cpu_offload_gb 8 # --gpu_memory_utilization 0.42

python run_amem.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --dataset ../../data/locomo10.json --output ../../results/base/qwen30b-instruct_a-mem_results.json --batched_run \
    --quantization bitsandbytes --dtype bfloat16 --seed 42 --gpu_memory_utilization 0.75 # --cpu_offload_gb 8
# python run_amem.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset ../../data/locomo10.json --output ../../results/base/qwen30b-thinking_a-mem_results.json --enable_thinking --batched_run \
#     --quantization bitsandbytes --dtype bfloat16 --seed 42 --gpu_memory_utilization 0.75 # --cpu_offload_gb 8

python run_amem.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset ../../data/locomo10.json --output ../../results/mcq/qwen4b-instruct_a-mem_results.json --batched_run --use_mcq \
    --quantization bitsandbytes --dtype bfloat16 --seed 42 # --cpu_offload_gb 8
# python run_amem.py --model Qwen/Qwen3-4B-Thinking-2507 --dataset ../../data/locomo10.json --output ../../results/mcq/qwen4b-thinking_a-mem_results.json --enable_thinking --batched_run --use_mcq \
#     --quantization bitsandbytes --dtype bfloat16 --seed 42 # --cpu_offload_gb 8

python run_amem.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --dataset ../../data/locomo10.json --output ../../results/mcq/qwen30b-instruct_a-mem_results.json --batched_run --use_mcq \
    --quantization bitsandbytes --dtype bfloat16 --seed 42 --gpu_memory_utilization 0.75 # --cpu_offload_gb 8
# python run_amem.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset ../../data/locomo10.json --output ../../results/mcq/qwen30b-thinking_a-mem_results.json --enable_thinking --batched_run --use_mcq \
#     --quantization bitsandbytes --dtype bfloat16 --seed 42 --gpu_memory_utilization 0.75 # --cpu_offload_gb 8 

cd ../../..