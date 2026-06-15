from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
from typing import List, Dict, Literal, Union, Any

class LLMController:
    def __init__(self, model_name: str = 'Qwen/Qwen3-4B-Instruct-2507', vllm_kwargs: Dict[str, Any] = {"dtype": "bfloat16"}, enable_thinking: bool = False):
        self.SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."
        if enable_thinking:
            vllm_kwargs['reasoning_config'] = ReasoningConfig(
                reasoning_parser=model_name.split('/')[-1].split('-')[0],
                reasoning_start_str="<think>",
                reasoning_end_str="I need to answer based on reasoning directly now</think>"
            )

        self.model = LLM(
            model=model_name,
            **vllm_kwargs,
        )

        self.chat_template_kwargs = {"enable_thinking": enable_thinking or "think" in model_name.lower()}

    def batch_decode(self, outputs_ids):
        tokenizer = self.model.get_tokenizer()
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
    
    def decode(self, output_ids):
        tokenizer = self.model.get_tokenizer()
        _, end_think = tokenizer.convert_tokens_to_ids(tokens=["<think>", "</think>"])
        think_ids = []
        answer_ids = []
        
        index = output_ids.index(end_think) + 1
        think_ids = output_ids[:index]
        answer_ids = output_ids[index:]
        
        reasoning_output = tokenizer.decode(think_ids, skip_special_tokens=True)
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)[0]
        
        pred = {
            "reasoning": reasoning_output,
            "answer": answer.strip()
        }
        return pred
    
    def format_messages(self, prompts):
        messages = []
        idxs = []
        for prompt in prompts:
            if isinstance(prompt, tuple):
                idxs.append(prompt[0])
            messages.append(self.format_message(prompt if isinstance(prompt, str) else prompt[1]))
        return idxs, messages

    def format_message(self, prompt):
        return [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]

    def batch_generate(self, prompts: List[tuple[int, str]] | List[str], sampling_params: Dict[str, str] | None = None, ):
        idxs, messages = self.format_messages(prompts)
        enable_thinking = self.chat_template_kwargs['enable_thinking']

        if sampling_params is None:
            sampling_params = {
                "n": 1,
                "max_tokens": 32 if not enable_thinking else 288,
            }
        else:
            if sampling_params.get('n') is None:
                sampling_params['n'] = 1
            if sampling_params.get("max_tokens") is None:
                sampling_params['max_tokens'] = 32 if not enable_thinking else 288
                
        if enable_thinking:
            sampling_params['thinking_token_budget'] = 256

        sampling_params = SamplingParams(**sampling_params)
        requests = self.model.enqueue_chat(messages, sampling_params=sampling_params, chat_template_kwargs=self.chat_template_kwargs)
        request_outputs = self.model.wait_for_completion()
        
        if enable_thinking:
            outputs_ids = [output.outputs[0].token_ids for output in request_outputs]
            preds = self.batch_decode(outputs_ids)
        else:
            preds = [output.outputs[0].text for output in request_outputs]

        if idxs:
            outputs = [
                (idx, pred)
                for idx, pred in zip(idxs, preds)
            ]

            return outputs
        
        return preds
    
    def generate(self, prompt: str, sampling_params: Dict[str, str] | None = None, ):
        message = self.format_message(prompt)
        enable_thinking = self.chat_template_kwargs['enable_thinking']

        if sampling_params is None:
            sampling_params = {
                "n": 1,
                "max_tokens": 32 if not enable_thinking else 288,
            }
        else:
            if sampling_params.get('n') is None:
                sampling_params['n'] = 1
            if sampling_params.get("max_tokens") is None:
                sampling_params['max_tokens'] = 32 if not enable_thinking else 288
                
        if enable_thinking:
            sampling_params['thinking_token_budget'] = 256

        sampling_params = SamplingParams(**sampling_params)
        requests = self.model.enqueue_chat([message], sampling_params=sampling_params, chat_template_kwargs=self.chat_template_kwargs)
        outputs = self.model.wait_for_completion()
        
        if enable_thinking:
            output_ids = outputs[0].outputs[0].token_ids
            preds = self.decode(output_ids)
        else:
            preds = outputs[0].outputs[0].text

        
        return preds
        

