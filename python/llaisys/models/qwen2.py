from typing import Sequence
from ..libllaisys import DeviceType

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError(f"Invalid model path: {model_path}")

        if device == DeviceType.CPU:
            device_map = "cpu"
        elif device == DeviceType.NVIDIA:
            device_map = "cuda"
        else:
            raise ValueError(f"Unsupported device type: {device}")

        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        input_ids = torch.tensor([list(inputs)], dtype=torch.long, device=self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        return outputs[0].tolist()
