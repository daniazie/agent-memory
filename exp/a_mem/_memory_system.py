from typing import List, Dict, Optional, Literal, Callable, Any

from datetime import datetime
import uuid
from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize

import logging

from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
)

from llm_controller import LLMController
from retriever import get_retriever

logger = logging.getLogger("amem")

def simple_tokenize(text):
    return word_tokenize(text)

class MemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.generate(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.generate(retry_prompt, temperature=0.3)
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# AgenticMemorySystem
# ---------------------------------------------------------------------------

class AgenticMemorySystem:
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm: str = "Qwen/Qwen3-4B-Instruct-2507",
                 evo_threshold: int = 100,
                 **vllm_kwargs
                 ):

        self.memories: Dict[str, List[MemoryNote] | MemoryNote] = {}
        self.retriever = get_retriever(model_name)
        self.llm_controller = LLMController(
            llm,
            **vllm_kwargs
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        note = MemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags)
        ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        init_kwargs = {
            "model_name": model_name
        }
        if hasattr(self.retriever, 'alpha'):
            alpha = self.retriever.alpha
            retriever_type = 'hybrid'
            init_kwargs['alpha'] = alpha
        else:
            retriever_type = 'simple_embed'

        init_kwargs['retriever_type'] = retriever_type

        self.retriever = get_retriever(**init_kwargs)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- evolution (3 sequential plain-text calls) ----

    def process_memory(self, note: MemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.generate(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                strengthen_response = self.llm_controller.generate(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.generate(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note


class BatchedAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm: str = "Qwen/Qwen3-4B-Instruct-2507",
                 evo_threshold: int = 100,
                 **vllm_kwargs
                 ):

        self.memories: Dict[str, List[MemoryNote] | MemoryNote] = {}
        self.retriever = get_retriever(model_name)
        self.llm_controller = LLMController(
            llm,
            **vllm_kwargs
        )
        self.evo_cnts = None
        self.evo_threshold = evo_threshold

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, contents: List[str] | str, timestamps: List[str] | str = None, **kwargs) -> str:
        """Add a new memory note."""
        notes = []
        if isinstance(contents, str):
            contents = [contents]
            timestamps = [timestamps]

        if self.evo_cnts is None:
            self.evo_cnts = [0] * len(contents)
        for content, timestamp in zip(contents, timestamps):
            note = MemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=timestamp,
            **kwargs,
            )
            notes.append(note)
        
        outputs = self.process_memories(notes)
        note_ids = []
        for i, output in enumerate(outputs):
            evo_label, notes[i] = output
            self.memories[notes[i].id] = notes[i]
            self.retriever.add_documents([
                "content:" + notes[i].content +
                " context:" + notes[i].context +
                " keywords: " + ", ".join(notes[i].keywords) +
                " tags: " + ", ".join(notes[i].tags)
            ])
            if evo_label:
                self.evo_cnts[i] += 1
                if self.evo_cnt[i] % self.evo_threshold == 0:
                    self.consolidate_memories()
            note_ids.append(notes[i].id)
        return note_ids

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        init_kwargs = {
            "model_name": model_name
        }
        if hasattr(self.retriever, 'alpha'):
            alpha = self.retriever.alpha
            retriever_type = 'hybrid'
            init_kwargs['alpha'] = alpha
        else:
            retriever_type = 'simple_embed'

        init_kwargs['retriever_type'] = retriever_type

        self.retriever = get_retriever(**init_kwargs)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- evolution (3 sequential plain-text calls) ----

    def format_prompt(self, prompt: str, **kwargs):
        return prompt.format(
            **kwargs
        )
    
    def process_decision(self, idx, decision_response):
        decision = parse_evolution_decision(decision_response)
        logger.debug("Evolution decision: %s", decision)

        if decision["decision"] == "NO_EVOLUTION":
            return False, False

        should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
        should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")
        return should_strengthen, should_update
    
    def process_strengthen(self, idx, strengthen_response, notes_to_process):
        strengthen = parse_strengthen_details(strengthen_response)
        logger.debug("Strengthen details: %s", strengthen)

        notes_to_process[idx][1].links.extend(strengthen["connections"])
        if strengthen["tags"]:
            notes_to_process[idx][1].tags = strengthen["tags"]
        return notes_to_process

    def process_update(self, indices, update_response):
        neighbor_updates = parse_update_neighbors(update_response, len(indices))
        logger.debug("Neighbor updates: %s", neighbor_updates)

        noteslist = list(self.memories.values())
        notes_id = list(self.memories.keys())
        for i in range(min(len(indices), len(neighbor_updates))):
            upd = neighbor_updates[i]
            memorytmp_idx = indices[i]
            if memorytmp_idx >= len(noteslist):
                continue
            notetmp = noteslist[memorytmp_idx]
            if upd["tags"]:
                notetmp.tags = upd["tags"]
            if upd["context"]:
                notetmp.context = upd["context"]
            self.memories[notes_id[memorytmp_idx]] = notetmp
        return True

    def try_except(self, function: Callable, note: str, **kwargs):
        try:
            return function(**kwargs)
        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return
        
    def process_memories(self, notes: List[MemoryNote]) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memories = []
        idxs = []
        outputs = []
        for note in notes:
            neighbor_memory, indices = self.find_related_memories(note.content, k=5)
            if len(indices) == 0:
                outputs.append((False, note))
            else:
                outputs.append((True, note))
            neighbor_memories.append(neighbor_memory)
            idxs.append(indices)

        # ---- Call 1: Evolution decision ----
        decision_prompts = []
        for i, (note, neighbor_memory) in enumerate(zip(notes, neighbor_memories)):
            if outputs[i][0]:
                fn_kwargs = dict(
                    prompt=EVOLUTION_DECISION_PROMPT,
                    context=note.context,
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                decision_prompt = self.try_except(self.format_prompt, note, **fn_kwargs)
                if decision_prompt is not None:
                    decision_prompts.append((i, decision_prompt))
                else:
                    outputs[i] = (False, note)
                        
        decision_responses = self.llm_controller.batch_generate(decision_prompts)
        notes_to_process = []
        for response in decision_responses:
            idx = response[0]
            decision_response = response[1]
            fn_kwargs = {
                'idx': idx, 
                'decision_response': decision_response
            }
            output = self.try_except(self.process_decision, notes[idx], **fn_kwargs)
            if output is not None:
                should_strengthen, should_update = output
                if not should_strengthen and not should_update:
                    outputs[idx] = (False, notes[idx])
                else:
                    notes_to_process.append((idx, notes[idx], neighbor_memories[idx], should_strengthen, should_update))
            else:
                outputs[idx] = (False, notes[idx])

        # ---- Call 2: Strengthen details (conditional) ----
        strengthen_prompts = []
        for note in notes_to_process:
            if note[3]:
                fn_kwargs = dict(
                        prompt=STRENGTHEN_DETAILS_PROMPT,
                        content=note[1].content,
                        keywords=note[1].keywords,
                        nearest_neighbors_memories=note[2],
                    )
                strengthen_prompt = self.try_except(self.format_prompt, note[1], **fn_kwargs)
                if strengthen_prompt is not None:
                    strengthen_prompts.append(strengthen_prompt)
                else:
                    outputs[note[0]] = (False, note[1])

        strengthen_responses = self.llm_controller.batch_generate(strengthen_prompts)
        for response in strengthen_responses:
            idx = response[0]
            strengthen_response = response[1]
            note = notes[idx]
            fn_kwargs = dict(idx=idx, strengthen_response=strengthen_response, notes_to_process=notes_to_process)
            output = self.try_except(self.process_strengthen, note, fn_kwargs)
            if output is not None:
                notes_to_process = output
            else:
                outputs[idx] = (False, note)

        # ---- Call 3: Update neighbors (conditional) ----
        update_prompts = []
        for note in notes_to_process:
            if note[4]:
                indices = idxs[note[0]]
                fn_kwargs = dict(
                    prompt=UPDATE_NEIGHBORS_PROMPT,
                    content=note[1].content,
                    context=note[1].context,
                    nearest_neighbors_memories=note[2],
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )

                update_prompt = self.try_except(self.format_prompt, note[1], **fn_kwargs)
                if update_prompt is not None:
                    update_prompts.append(update_prompt)
                else:
                    outputs[note[0]] = (False, note[1])
        
        update_responses = self.llm_controller.batch_generate(update_prompts)
        for response in update_responses:
            idx = response[0]
            update_response = response[1]
            fn_kwargs = dict(indices=idxs[idx], update_response=update_response)
            output = self.try_except(self.process_update, notes[idx], **fn_kwargs)
            if output is not None:
                outputs[idx] = (True, notes[idx])
        
        return outputs

def run_tests():
    """Run system tests"""
    print("Starting Memory System Tests...")
    
    # Initialize memory system with OpenAI backend
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm="Qwen/Qwen3-4B-Instruct-2507"
    )
    
    print("\nAdding test memories...")
    
    # Add test memories - only content is required
    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))
    
    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))
    
    print("\nQuerying for related memories...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )
    
    related = memory_system.find_related_memories(query.content, k=2)
    print("related", related)
    print("\nResults:")
    for i, memory in enumerate(related, 1):
        print(f"\n{i}. Memory:")
        print(f"Content: {memory.content}")
        print(f"Category: {memory.category}")
        print(f"Keywords: {memory.keywords}")
        print(f"Tags: {memory.tags}")
        print(f"Context: {memory.context}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()