# python exp/no_memory.py --model Qwen/Qwen3-4B-Instruct-2507 --batch_size 64 --seed 42 --format_amem --output_dir results/base
# python exp/no_memory.py --model Qwen/Qwen3-4B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget --seed 42 --format_amem --output_dir results/base

# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --batch_size 64 --seed 42 --format_amem --output_dir results/base
# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget --seed 42 --format_amem --output_dir results/base

# python exp/no_memory.py --model Qwen/Qwen3-4B-Instruct-2507 --batch_size 64 --seed 42 --format_amem --use_mcq --output_dir results/mcq
# python exp/no_memory.py --model Qwen/Qwen3-4B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget --seed 42 --format_amem --use_mcq --output_dir results/mcq

# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --batch_size 64 --seed 42 --format_amem --use_mcq --output_dir results/mcq
# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget --seed 42 --format_amem --use_mcq --output_dir results/mcq

bash scripts/run_amem.sh