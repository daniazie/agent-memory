from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from transformers import AutoTokenizer

from ast import literal_eval
from pydantic import BaseModel
from functools import partial
import argparse
import json
import os

from data_utils import load_dataset
from a_mem.utils import calculate_metrics, aggregate_metrics

class Answer(BaseModel):
    answer: str
    thinking_content: str

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument('--data_path', type=str, default='data/locomo10.json')
    parser.add_argument('--enable_thinking', action='store_true', default=False)
    parser.add_argument('--thinking_token_budget', type=int, default=0)
    parser.add_argument('--enable_thinking_budget', action='store_true', default=False)
    parser.add_argument('--use_mcq', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='results')
    return parser

def load_model(args):
    init_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        quantization="bitsandbytes",
        max_num_seqs=args.batch_size,
    )

    if args.enable_thinking:
        init_kwargs.update(dict(reasoning_config=ReasoningConfig(reasoning_parser="qwen3", reasoning_start_str="<think>", reasoning_end_str="I need to answer based on the reasoning directly now</think>")))
    model = LLM(
        **init_kwargs
    )

    return model

def decode_thinking(model: LLM, outputs_ids: list):
    tokenizer = model.get_tokenizer()
    _, end_think = tokenizer.convert_tokens_to_ids(tokens=["<think>", "</think>"])
    think_ids = []
    answer_ids = []
    for output_ids in outputs_ids:
        index = output_ids.index(end_think) + 1
        think_ids.append(output_ids[:index])
        answer_ids.append(output_ids[index:])
    
    reasoning_outputs = tokenizer.decode(think_ids, skip_special_tokens=True)
    answers = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    preds = []
    for reasoning, answer in zip(reasoning_outputs, answers):
        pred = {
            "reasoning": reasoning,
            "answer": answer.strip()
        }
        preds.append(pred)
    return preds

def main(args: argparse.Namespace):
    dataset = load_dataset(args.data_path, args.use_mcq)
    model = load_model(args)

    sampling_params = model.get_default_sampling_params()
    sampling_params.n = 1
    sampling_params.max_tokens = args.thinking_token_budget + args.max_tokens
    if args.enable_thinking_budget:
        sampling_params.thinking_token_budget = args.thinking_token_budget

    prompts = dataset['prompt']
    requests = model.enqueue_chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": args.enable_thinking})
    outputs = model.wait_for_completion()

    if args.enable_thinking:
        outputs_ids = [output.outputs[0].token_ids for output in outputs]
        preds = decode_thinking(model, outputs_ids)
    else:
        preds = [output.outputs[0].text for output in outputs]

    results = []
    metrics = []
    categories = []
    final_results = {
        "model": args.model.split('/')[-1],
        "dataset": args.data_path,
        "total_questions": dataset.num_rows,
    }
    per_sample_scores = []
    for pred, item in zip(preds, dataset):
        result = {
            k: v
            for k, v in item.items()
            if not k == 'prompt'
        }
        scores = calculate_metrics(pred, item['reference'])
        metrics.append(scores)
        categories.append(item['category'])
        if args.enable_thinking:
            result['prediction'] = pred['answer']
            result['reasoning'] = pred['reasoning']
        else:
            result['prediction'] = pred
        results.append(result)

        per_sample_scores.append({
            "category": item['category'],
            "question": item['question'],
            "answer": item['answer'],
            "reference": item['reference'],
            "evidence": item['dialogue'],
            "prediction": result.get("prediction"),
            "scores": scores
        })

    aggregated = aggregate_metrics(metrics, categories)

    final_results['aggregated_scores'] = aggregated
    final_results['per_sample_scores'] = per_sample_scores
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('preds', exist_ok=True)
    model_name = args.model.split('/')[-1].lower()
    if args.enable_thinking:
        if not 'think' in model_name.lower() or not 'reason' in model_name.lower():
            model_name = f"{model_name}-reasoning"
        if args.enable_thinking_budget:
            model_name = f"{model_name}_budgeted"
    if args.use_mcq:
        model_name = f"{model_name}_mcq"
    
    result_file = f"{model_name}_no-memory_results.json"
    with open(f"preds/{result_file}", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(f"{args.output_dir}/{result_file}", "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2)

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    main(args)