from transformers import PreTrainedTokenizerBase
from datasets import Dataset, Value, concatenate_datasets, Json
import random
import json

PROMPT = """Answer the question based on the following conversation(s).

{conversation}

Question: {question}

Do not write lengthy or complete sentences; instead provide your answer using keywords. Whenever possible, answer with exact words from the conversation(s)..
If the answer cannot be found or inferred, do not share false information.
"""

def format_session(session, timestamp):
    session_conv = f"TIMESTAMP: {timestamp}\nCONVERSATION:\n"
    for dialogue in session:
        utterance = ""
        if dialogue.get("blip_caption") is not None:
            utterance += f"<image_description>{dialogue['blip_caption']}</image_description>"
        utterance += dialogue['text']
        session_conv += f"\t{dialogue['speaker']}: {utterance}\n"
    return session_conv + '\n'

def format_single_hop(evidence, conversation):
    session_no = int(evidence[0].strip('D').split(':')[0])
    session_id = f"session_{session_no}"
    session = conversation.get(session_id)
    session_date = conversation.get(f"{session_id}_date_time")
    context_conversation = format_session(session, session_date)
    return context_conversation

def format_multihop(conversation):
    session_ids = [item for i in range(len(conversation)) for item in conversation.keys() if item == 'session_%d' % (i+1)]
    full_conv = ""
    for session_id in session_ids:
        session_date = conversation[f"{session_id}_date_time"]
        session = conversation[session_id]
        session_conv = format_session(session, session_date)
        full_conv += session_conv 
    return full_conv

def remove_session(examples):
    conversation = examples['conversation']
    session_ids = [item for i in range(len(conversation)) for item in conversation.keys() if item == 'session_%d' % (i+1)]
    sessions = [item for item in conversation.keys() if 'date_time' in item and '_'.join(item.split('_')[:2]) not in set(session_ids)]
    conversation = {
        k: v
        for k, v in conversation.items()
        if not k in set(sessions)
    }
    examples['conversation'] = conversation
    return examples

def add_evidence(examples):
    qas = examples['qa']
    conversation = examples['conversation']
    for qa in qas:
        evidence = qa['evidence']
        session_ids = [item.split(":")[0].strip("D") for item in qa['evidence']]
        session_ids = list(set(['session_' + item for item in session_ids if item.isdigit()]))
        sessions = [conversation[session_id] for session_id in session_ids]
        dialogue = []
        for session in sessions:
            for item in session:
                if item.get("dia_id") in set(evidence):
                    text = f"{item['speaker']}: {item['text']}"
                    if text not in dialogue:
                        dialogue.append(text.strip())
                
        qa['dialogue'] = dialogue
    examples['qa'] = qas
    return examples

def format_dataset(examples, category='all'):
    examples = add_evidence(remove_session(examples))
    if category == 'all':
        qas = examples['qa']
        categories = []
    else:
        qas = [example for example in examples['qa'] if example['category'] == category]
        categories = None

    conversation = examples['conversation']

    questions = []
    prompts = []
    answers = []
    dialogue = []
    for qa in qas:
        if qa['category'] == 4:
            context_conv = format_single_hop(qa['evidence'], conversation)
        else:
            context_conv = format_multihop(conversation)
        query = qa['question']
        dialogue.append(qa['dialogue'])
        questions.append(query)

        if qa['category'] == 2:
            query += " Use the timestamps as a reference to approximate the answer when needed."
        if qa['category'] == 5:
            choices = "\nChoose the correct answer:\n(a) {}\n(b){}"
            if random.random() < 0.5:
                query += choices.format(qa['adversarial_answer'], "No information available")
                answer = {'a': qa['adversarial_answer'], 'b': "No information available"}
            else:
                query += choices.format("No information available", qa['adversarial_answer'])
                answer = {'a': "No information available", 'b': qa['adversarial_answer']}

        else:
            answer = qa['answer']

        prompt = PROMPT.format(conversation=context_conv, question=query)
        prompt = [
            {"role": "user", "content": prompt}
        ]
        
        prompts.append(prompt)
        answers.append(answer)
        if categories is not None:
            categories.append(qa['category'])

    dataset = {
        'question': questions,
        "prompt": prompts,
        "answer": answers,
    }

    if not category == 5:
        dataset['dialogue'] = dialogue

    if categories is not None:
        dataset['category'] = categories
    return dataset

def load_dataset(data_path, category):
    data = json.load(open(data_path, "r"))
    datasets = []
    for sample in data:
        dataset = Dataset.from_dict(format_dataset(sample, category=category), on_mixed_types='use_json')
        datasets.append(dataset)

    dataset: Dataset = concatenate_datasets(datasets)
        
    return dataset

def prepare_dataset(examples, tokenizer: PreTrainedTokenizerBase):
    prompts = []
    for example in examples['prompt']:
        prompt = tokenizer.apply_chat_template(
            example,
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    return {"prompt": prompts}