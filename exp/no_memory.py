from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.config import ReasoningConfig
from transformers import AutoTokenizer

from ast import literal_eval
from pydantic import BaseModel
from functools import partial
import argparse
import json
import os

from data_utils import load_dataset, prepare_dataset

class Answer(BaseModel):
    answer: str
    thinking_content: str

def choice_type(choice: str):
    if choice.isdigit():
        return int(choice)
    return choice

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument('--data_path', type=str, default='A-mem/data/locomo10.json')
    parser.add_argument('--enable_thinking', action='store_true', default=False)
    parser.add_argument('--category', type=choice_type, default='all')
    parser.add_argument('--batch_size', type=int, default=32)
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
            "answer": answer
        }
        preds.append(pred)
    return preds

def main(args: argparse.Namespace):
    dataset = load_dataset(args.data_path, category=args.category)
    model = load_model(args)

    sampling_params = model.get_default_sampling_params()
    sampling_params.n = 1
    sampling_params.max_tokens = 512 if args.enable_thinking else 16
    if args.enable_thinking:
        sampling_params.thinking_token_budget = sampling_params.max_tokens - 16

    prompts = dataset['prompt']
    requests = model.enqueue_chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": args.enable_thinking})
    outputs = model.wait_for_completion()

    if args.enable_thinking:
        outputs_ids = [output.outputs[0].token_ids for output in outputs]
        preds = decode_thinking(model, outputs_ids)
    else:
        preds = [output.outputs[0].text for output in outputs]

    results = []
    for pred, item in zip(preds, dataset):
        result = {
            k: v
            for k, v in item.items()
            if not k == 'prompt'
        }
        if args.enable_thinking:
            result['prediction'] = pred['answer']
            result['reasoning'] = pred['reasoning']
        else:
            result['prediction'] = pred
        results.append(result)
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_thinking:
        result_file = f"{args.model.split('/')[-1].lower()}_no-memory_{args.category}_results.json"
    else:
        result_file = f"{args.model.split('/')[-1].lower()}_no-memory_{args.category}_results.json"
    with open(f"{args.output_dir}/{result_file}", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    main(args)