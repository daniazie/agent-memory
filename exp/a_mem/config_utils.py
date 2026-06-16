from typing import List, Union, Literal, get_args, get_origin
from types import UnionType, NoneType
from dataclasses import dataclass, asdict

@dataclass
class vLLMConfig:
    quantization: str | None = None
    max_num_seqs: int | None = None
    dtype: Literal['bfloat16', 'float16', 'float32'] = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    seed: int | None = None
    cpu_offload_gb: int = 0
    enforce_eager: bool = False

def parse_vllm_kwargs(vllm_kwargs: List[str]):
    vllm_config = vLLMConfig()
    fields = vLLMConfig.__annotations__
    for field_name, attr_types in fields.items():
        attr_vals = None
        attr_type = get_origin(attr_types) or fields[field_name]
        if attr_type is Union or attr_type is UnionType:
            attr_type = get_args(attr_types)[0]
            assert attr_type is not NoneType
        elif attr_type is Literal:
            attr_vals = get_args(attr_types)

        kwarg = f"--{field_name}"
        if attr_type is bool:
            field_val = kwarg in vllm_kwargs
        elif kwarg in vllm_kwargs:
            kwarg_idx = vllm_kwargs.index(kwarg)
            field_val = vllm_kwargs[kwarg_idx + 1]
            if attr_vals is not None:
                assert field_val in attr_vals
            else:
                field_val = attr_type(field_val)
        else:
            field_val = None

        setattr(vllm_config, field_name, field_val)
    
    vllm_config = {
        k: v
        for k, v in asdict(vllm_config).items()
        if v is not None
    }

    print(vLLMConfig(**vllm_config))

    return vllm_config