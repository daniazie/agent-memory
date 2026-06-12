models=(
    gpt-5.4-nano
    claude-sonnet-4-6
    gemini-3.1-flash-lite
)

for model in ${models[@]}; do
    python test_advanced_robust.py --backend openai --model ${model} \
        --dataset data/locomo10.json --output results_robust_${model}.json
done