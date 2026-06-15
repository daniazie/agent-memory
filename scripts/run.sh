# python exp/no_memory.py --batch_size 64
# python exp/no_memory.py --model Qwen/Qwen3-4B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256
# python exp/no_memory.py --model Qwen/Qwen3-4B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget

# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --batch_size 64
# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256
# python exp/no_memory.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --batch_size 64 --enable_thinking --thinking_token_budget 256 --enable_thinking_budget

bash scripts/run_amem.sh