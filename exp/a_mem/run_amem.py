"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""
from vllm import LLM
from memory_system import AgenticMemorySystem
from mem_agent import MemAgent
from llm_controller import LLMController
from llm_text_parsers import (
    parse_plain_text_answer,
)
import os
import json
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
import torch
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from config_utils import parse_vllm_kwargs
from datetime import datetime
import time
import gc

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
def load_embedding_model(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        model = LLM(
            embedding_model,
            runner='pooling'
        )
    except Exception as e:
        model = SentenceTransformer(embedding_model)
    except Exception as e:
        model = None
    return model

logger = logging.getLogger("amem")

def flush():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return gc.collect()


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger

def evaluate_dataset(dataset_path: str, model: str, embedding_model: str, batched_run: bool, output_path: Optional[str] = None,
                     ratio: float = 1.0,
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     enable_thinking: bool = False, use_mcq: bool = False, vllm_kwargs: dict | None = None):
    """Evaluate the robust agent on the LoComo dataset."""
    if batched_run:
        from eval_utils import calculate_metrics, aggregate_metrics
    else:
        from utils import calculate_metrics, aggregate_metrics

    assert Path(('/').join(output_path.split('/')[:-1])).exists() == True

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    think_mode = "thinking" if enable_thinking else "no-think"
    log_filename = f"eval_robust_{model}_{think_mode}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(model, think_mode),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]

    predictions, references = [], []
    agent = MemAgent(model, embedding_model, retrieve_k, temperature_c5, enable_thinking=enable_thinking, batched_run=batched_run, use_mcq=use_mcq, vllm_kwargs=vllm_kwargs)
    for sample_idx, sample in enumerate(samples):
        agent.reset_memory() # usah bazir memori, nak
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"No retriever cache found, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

            
            for _, turns in sample.conversation.sessions.items():
                conversation = []
                timestamps = []
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    if batched_run:
                        conversation.append(conversation_tmp)
                        timestamps.append(turn_datatime)
                    else:
                        agent.add_memory(conversation_tmp, timestamp=turn_datatime)
                if batched_run and len(conversation) > 0 and len(timestamps) > 0:
                    agent.add_memory(conversation, timestamps)

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                eval_logger.info(f"Question {total_questions}: {qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {qa.final_answer}")
                eval_logger.info(f"User Prompt: {user_prompt}")
                eval_logger.info(f"Category: {qa.category}")
                eval_logger.info(f"Raw Context: {raw_context}")

                final_answer = qa.final_answer if qa.category != 5 else "Not mentioned"
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": final_answer,
                    "category": qa.category,
                }

                if not batched_run:
                    metrics = calculate_metrics(prediction, final_answer) if final_answer else {
                        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                    }

                    all_metrics.append(metrics)
                    all_categories.append(qa.category)
                    result["metrics"] = metrics

                results.append(result)

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")
        if batched_run:
            predictions += [result['prediction'] for result in results]
            references += [result['reference'] for result in results]
        
        eval_logger.info(f"Running garbage collector: {flush()}")

    if batched_run:
        del agent
        time.sleep(3)
        sentence_model = load_embedding_model(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        all_metrics = calculate_metrics(predictions, references, sentence_model=sentence_model)

        for result, metrics in zip(results, all_metrics):
            result['metrics'] = metrics

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model to use")
    parser.add_argument('--embedding_model', type=str, default="Qwen/Qwen3-Embedding-4B",
                        help="Embedding model to use")
    parser.add_argument('--batched_run', action='store_true', default=False)
    parser.add_argument('--enable_thinking', action='store_true', default=False)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--use_mcq", action='store_true', default=False)
    args, vllm_kwargs = parser.parse_known_args()

    print(args)

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = args.dataset # os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = args.output # os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    if vllm_kwargs:
        vllm_kwargs = parse_vllm_kwargs(vllm_kwargs)
    else:
        vllm_kwargs = None

    evaluate_dataset(
        dataset_path, args.model, args.embedding_model, args.batched_run, output_path, args.ratio,
        args.temperature_c5, args.retrieve_k, args.enable_thinking, args.use_mcq, vllm_kwargs=vllm_kwargs
    )


if __name__ == "__main__":
    main()
