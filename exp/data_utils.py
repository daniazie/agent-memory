from transformers import PreTrainedTokenizerBase
from datasets import Dataset, Value, concatenate_datasets, Json
import random
import json

SYSTEM_PROMPT = """You are to answer a user query based on a series of conversations. 
Express your answer in keywords ONLY. Do NOT write full and lengthy sentences.
For example, given a scenario where Amy mentions her plan to pursue astrophysics in university:
```
Question: What does Amy plan to study in university?

[Examples of wrong format]
Amy plans to study astrophysics in university.
Amy wants to study astrophysics.
Amy wants to pursue astrophysics in university.

[Examples of correct format]
Astrophysics
astrophysics
```
Note that the timestamp (written in brackets ([]) before each conversation correspond to the time and date the conversation takes place.
Use the timestamp(s) as a reference to approximate the answer when needed.
For example, given a conversation that takes place on 4 May 2022, and Amber mentions watching a movie the previous day:
```
Question: When did Amber watch [MOVIE TITLE]?

[Acceptable answers]
3 May
3 May 2022
the day before 4 May 2022
```

Whenever possible, answer with exact words from the conversation(s). If you do not know the answer, do not share false information.
""".strip()

USER_PROMPT = """{conversation}

Question: {question}
"""

def format_session(session, timestamp):
    session_conv = f"[{timestamp}]\n"
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

def format_dataset(examples, use_mcq: bool = False):
    examples = add_evidence(remove_session(examples))
    qas = examples['qa']
    categories = []
    conversation = examples['conversation']

    questions = []
    prompts = []
    answers = []
    dialogue = []
    references = []
    for qa in qas:
        category = qa['category']
        if category == 4:
            context_conv = format_single_hop(qa['evidence'], conversation)
        else:
            context_conv = format_multihop(conversation)
        query = qa['question']
        dialogue.append(qa['dialogue'])
        questions.append(query)

        if use_mcq:
            if qa['category'] == 5:
                choices = "\nChoose the correct answer:\n(a) {}\n(b){}"
                if random.random() < 0.5:
                    query += choices.format(qa['adversarial_answer'], "Not mentioned")
                    answer = {"a": qa['adversarial_answer'], "b": "Not mentioned"}
                else:
                    query += choices.format("Not mentioned", qa['adversarial_answer'])
                    answer = {"b": "Not mentioned", "b": qa['adversarial_answer']}

            else:
                answer = qa['answer']
        else:
            answer = qa.get('answer') or qa.get('adversarial_answer')
            answer = str(answer)

        reference = str(qa.get('answer')) or "Not mentioned"
        

        user_prompt = USER_PROMPT.format(conversation=context_conv, question=query)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        prompts.append(prompt)
        answers.append(answer)
        references.append(reference)
        categories.append(qa['category'])

    dataset = {
        'question': questions,
        "prompt": prompts,
        "reference": references,
        'answer': answers,
        'dialogue': dialogue,
        'category': categories
    }

    return dataset

def format_sample_amem(category, context, question, answer, use_mcq: bool = False):
    if category == 5 and use_mcq:
        answer_tmp = list()
        if random.random() < 0.5:
            answer_tmp.append('Not mentioned in the conversation')
            answer_tmp.append(answer)
        else:
            answer_tmp.append(answer)
            answer_tmp.append('Not mentioned in the conversation')
        user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
    elif category == 2:
        user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
    elif category == 3:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
    else:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
    return user_prompt

def format_dataset_amem(examples, use_mcq: bool = False):
    SYS_PROMPT = "Follow the format specified in the prompt exactly. Do not add extra commentary."
    examples = add_evidence(remove_session(examples))
    qas = examples['qa']
    categories = []
    conversation = examples['conversation']

    questions = []
    prompts = []
    answers = []
    dialogue = []
    references = []

    for qa in qas:
        category = qa['category']
        if category == 4:
            context = format_single_hop(qa['evidence'], conversation)
        else:
            context = format_multihop(conversation)
        question = qa['question']
        dialogue.append(qa['dialogue'])
        questions.append(question)
        answer = qa.get('answer') or qa.get('adversarial_answer')
        answer = str(answer)
        reference = str(qa.get('answer')) or "Not mentioned"
        query = format_sample_amem(category, context, question, answer, use_mcq=use_mcq)
        prompt = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": query}
        ]

        prompts.append(prompt)
        answers.append(answer)
        references.append(reference)
        categories.append(category)

    return {
        'question': questions,
        "prompt": prompts,
        "reference": references,
        'answer': answers,
        'dialogue': dialogue,
        'category': categories
    }


def load_dataset(data_path, use_mcq: bool = False, format_amem: bool = False):
    data = json.load(open(data_path, "r"))
    datasets = []
    for sample in data:
        if format_amem:
            dataset = Dataset.from_dict(format_dataset_amem(sample, use_mcq=use_mcq), on_mixed_types='use_json')
        else: 
            dataset = Dataset.from_dict(format_dataset(sample, use_mcq=use_mcq), on_mixed_types='use_json')
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