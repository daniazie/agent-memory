export WORKING_DIR=$(pwd)
cd exp/a_mem

python run_amem.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset ${WORKING_DIR}/data/locomo10.json --output ${WORKING_DIR}/results/qwen4b-instruct_a-mem_results.json --batched_run \
    --quantization bitsandbytes --cpu_offload_gb 8 --dtype bfloat16 --enforce_eager --seed 42 --gpu_memory_utilization 0.42
python run_amem.py --model Qwen/Qwen3-4B-Thinking-2507 --dataset ${WORKING_DIR}/data/locomo10.json --output ${WORKING_DIR}/results/qwen4b-thinking_a-mem_results.json --enable_thinking --batched_run \
    --quantization bitsandbytes --cpu_offload_gb 8 --dtype bfloat16 --enforce_eager --seed 42 --gpu_memory_utilization 0.42

python run_amem.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --dataset ${WORKING_DIR}/data/locomo10.json --output ${WORKING_DIR}/results/qwen30b-instruct_a-mem_results.json --batched_run \
    --quantization bitsandbytes --cpu_offload_gb 8 --dtype bfloat16 --enforce_eager --seed 42 --gpu_memory_utilization 0.42
python run_amem.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset ${WORKING_DIR}/data/locomo10.json --output ${WORKING_DIR}/results/qwen30b-thinking_a-mem_results.json --enable_thinking --batched_run \
    --quantization bitsandbytes --cpu_offload_gb 8 --dtype bfloat16 --enforce_eager --seed 42 --gpu_memory_utilization 0.42

cd ../..