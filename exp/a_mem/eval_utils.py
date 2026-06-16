from vllm import LLM
from typing import List, Dict, Union, Tuple
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import torch

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_match(prediction: str, reference: str) -> tuple:
    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return exact_match, f1

def calculate_rouge_scores(prediction: List[str] | str, reference: List[str] | str) -> Dict[str, float] | List[Dict[str, float]]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    if not isinstance(prediction, list) and not isinstance(reference, list):
        scores = scorer.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    else:
        scores = []
        for pred, ref in zip(prediction, reference):
            score = scorer.score(ref, pred)
            scores.append({
                'rouge1_f': score['rouge1'].fmeasure,
                'rouge2_f': score['rouge2'].fmeasure,
                'rougeL_f': score['rougeL'].fmeasure
            })
        return scores

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores

def calculate_bert_scores(prediction: List[str] | str, reference: List[str] | str) -> Dict[str, float] | List[Dict[str, float]]:
    """Calculate BERTScore for semantic similarity."""
    if not isinstance(prediction, list) and not isinstance(reference, list):
        prediction = [prediction]
        reference = [reference]

    try:
        P, R, F1 = bert_score(prediction, reference, lang='en', verbose=False)
        if len(prediction) == 1:
            return {
                'bert_precision': P.item(),
                'bert_recall': R.item(),
                'bert_f1': F1.item()
            }
        else:
            scores = []
            for p, r, f1 in zip(P, R, F1):
                score = {
                    'bert_precision': p.item(),
                    'bert_recall': r.item(),
                    'bert_f1': f1.item()
                }
                scores.append(score)
            return scores
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: List[str] | str, reference: List[str] | str) -> float | List[float]:
    """Calculate METEOR score for the prediction."""
    scores = []
    try:
        if not isinstance(prediction, list) and not isinstance(reference, list):
            return meteor_score([reference.split()], prediction.split())
        else:
            for pred, ref in zip(prediction, reference):
                score = meteor_score([ref.split()], pred.split())
                scores.append(score)
            return scores
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        if not isinstance(prediction, list) and not isinstance(reference, list):
            return 0.0
        else:
            return [0.0] * len(prediction)

def calculate_sentence_similarity(prediction: List[str] | str, reference: List[str] | str, sentence_model: LLM | SentenceTransformer | None = None) -> float | List[float]:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        if not isinstance(prediction, list) and not isinstance(reference, list):
            return 0.0
        return [0.0] * len(prediction)
    try:
        # Encode sentences
        if not isinstance(prediction, list) and not isinstance(reference, list):
            prediction = [prediction]
            reference = [reference]
        if isinstance(sentence_model, SentenceTransformer):
            embedding1 = sentence_model.encode(prediction, convert_to_tensor=True)
            embedding2 = sentence_model.encode(reference, convert_to_tensor=True)
            
            # Calculate cosine similarity
            if not isinstance(prediction, list) and not isinstance(reference, list):
                similarity = pytorch_cos_sim(embedding1, embedding2).item()
                return float(similarity)
            similarity = []
            for emb1, emb2 in zip(embedding1, embedding2):
                sim = pytorch_cos_sim(emb1, emb2).item()
                similarity.append(float(sim))
            return similarity
        
        pred_outputs = sentence_model.embed(prediction)
        ref_outputs = sentence_model.embed(reference)
        if len(pred_outputs) == 1:
            pred_embeddings = pred_outputs[0].outputs.embedding
            ref_embeddings = ref_outputs[0].outputs.embedding
            similarity = pytorch_cos_sim(pred_embeddings, ref_embeddings).item()
            return float(similarity)
        similarity = []
        for pred, ref in zip(pred_outputs, ref_outputs):
            pred_embeddings = pred.outputs.embedding
            ref_embeddings = ref.outputs.embedding
            sim = pytorch_cos_sim(pred_embeddings, ref_embeddings).item()
            similarity.append(float(sim))
        return similarity

    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: List[str] | str, reference: List[str] | str, sentence_model: LLM | SentenceTransformer = None) -> Dict[str, float] | List[Dict[str, float]]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0
        }
    
    # Convert to strings if they're not already
    if not isinstance(prediction, list):
        prediction = str(prediction).strip()
        reference = str(reference).strip()
    
        exact_match, f1 = calculate_match(prediction, reference)
        bleu_scores = calculate_bleu_scores(prediction, reference)
    else:
        prediction = [str(pred).strip() for pred in prediction]
        reference = [str(ref).strip() for ref in reference]

        exact_match, f1 = [], []
        bleu_scores = []
        for pred, ref in zip(prediction, reference):
            exact, f = calculate_match(pred, ref)
            exact_match.append(exact)
            f1.append(f)
        
            bleu = calculate_bleu_scores(pred, ref)
            bleu_scores.append(bleu)
    
    # Calculate all scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    bert_scores = calculate_bert_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference, sentence_model=sentence_model)
    
    if isinstance(prediction, str):
        # Combine all metrics
        metrics = {
            "exact_match": exact_match,
            "f1": f1,
            **rouge_scores,
            **bleu_scores,
            **bert_scores,
            "meteor": meteor,
            "sbert_similarity": sbert_similarity
        }
        return metrics
    scores = []
    for i in range(len(prediction)):
        metrics = {
            "exact_match": exact_match[i],
            "f1": f1[i],
            **rouge_scores[i],
            **bleu_scores[i],
            **bert_scores[i],
            "meteor": meteor[i],
            "sbert_similarity": sbert_similarity[i]
        }
        scores.append(metrics)
    return scores

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}
    
    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))
    
    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)
    
    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results