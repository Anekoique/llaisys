import json
import mmap
import struct
from pathlib import Path
from typing import Sequence, List

import ctypes

from .. import libllaisys
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..tensor import Tensor


class SafeTensorsReader:
    def __init__(self, path: Path):
        self._path = path
        self._file = path.open("rb")
        header_len = struct.unpack("<Q", self._file.read(8))[0]
        header_bytes = self._file.read(header_len)
        self._header = json.loads(header_bytes)
        self._data_start = 8 + header_len
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def info(self, name: str):
        if name not in self._header:
            raise KeyError(f"Tensor {name} not found in safetensors header")
        return self._header[name]

    def tensor_bytes(self, name: str) -> bytes:
        info = self.info(name)
        start, end = info["data_offsets"]
        start += self._data_start
        end += self._data_start
        return self._mmap[start:end]


class Qwen2:
    _DTYPE_MAP = {
        "BF16": DataType.BF16,
        "F16": DataType.F16,
        "F32": DataType.F32,
        "F64": DataType.F64,
        "I64": DataType.I64,
        "I32": DataType.I32,
        "U8": DataType.U8,
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._device = device
        self._model = None
        self._weights = None
        self._weights_tensors: List[Tensor] = []

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError(f"Invalid model path: {model_path}")

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        dtype_str = config.get("torch_dtype", "bfloat16")
        if dtype_str == "bfloat16":
            dtype = DataType.BF16
        elif dtype_str == "float16":
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        self._meta = LlaisysQwen2Meta(
            dtype=libllaisys.llaisysDataType_t(dtype),
            nlayer=config["num_hidden_layers"],
            hs=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            maxseq=config.get("max_position_embeddings", 0),
            voc=config["vocab_size"],
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config.get("rope_theta", 10000.0)),
            end_token=int(config.get("eos_token_id", -1)),
        )

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            libllaisys.llaisysDeviceType_t(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._load_weights(model_path)

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _dtype_from_info(self, info_dtype: str) -> DataType:
        if info_dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype: {info_dtype}")
        return self._DTYPE_MAP[info_dtype]

    def _dtype_size(self, dtype: DataType) -> int:
        if dtype in (DataType.F16, DataType.BF16):
            return 2
        if dtype in (DataType.F32, DataType.I32, DataType.U32):
            return 4
        if dtype in (DataType.F64, DataType.I64, DataType.U64):
            return 8
        if dtype == DataType.U8:
            return 1
        raise ValueError(f"Unsupported dtype size: {dtype}")

    def _load_tensor_from_reader(self, reader: SafeTensorsReader, name: str) -> Tensor:
        info = reader.info(name)
        dtype = self._dtype_from_info(info["dtype"])
        shape = tuple(info["shape"])
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        expected = 1
        for dim in shape:
            expected *= dim
        expected *= self._dtype_size(dtype)
        if expected != nbytes:
            raise ValueError(f"Tensor {name} byte size mismatch: expected {expected}, got {nbytes}")

        data = reader.tensor_bytes(name)
        buf = (ctypes.c_ubyte * nbytes).from_buffer_copy(data)
        ptr = ctypes.cast(buf, ctypes.c_void_p)

        t = Tensor(shape, dtype=dtype, device=self._device)
        t.load(ptr)
        self._weights_tensors.append(t)
        return t

    def _load_weights(self, model_path: Path):
        weights_ptr = self._weights
        if weights_ptr is None:
            raise RuntimeError("Weights pointer is null")

        weight_file = model_path / "model.safetensors"
        if not weight_file.is_file():
            raise FileNotFoundError(f"Missing model.safetensors in {model_path}")

        with SafeTensorsReader(weight_file) as reader:
            in_embed = self._load_tensor_from_reader(reader, "model.embed_tokens.weight")
            out_embed = self._load_tensor_from_reader(reader, "lm_head.weight")
            out_norm = self._load_tensor_from_reader(reader, "model.norm.weight")

            weights_ptr.contents.in_embed = in_embed.lib_tensor()
            weights_ptr.contents.out_embed = out_embed.lib_tensor()
            weights_ptr.contents.out_norm_w = out_norm.lib_tensor()

            for i in range(self._meta.nlayer):
                prefix = f"model.layers.{i}."

                attn_norm_w = self._load_tensor_from_reader(reader, prefix + "input_layernorm.weight")
                q_w = self._load_tensor_from_reader(reader, prefix + "self_attn.q_proj.weight")
                q_b = self._load_tensor_from_reader(reader, prefix + "self_attn.q_proj.bias")
                k_w = self._load_tensor_from_reader(reader, prefix + "self_attn.k_proj.weight")
                k_b = self._load_tensor_from_reader(reader, prefix + "self_attn.k_proj.bias")
                v_w = self._load_tensor_from_reader(reader, prefix + "self_attn.v_proj.weight")
                v_b = self._load_tensor_from_reader(reader, prefix + "self_attn.v_proj.bias")
                o_w = self._load_tensor_from_reader(reader, prefix + "self_attn.o_proj.weight")

                mlp_norm_w = self._load_tensor_from_reader(reader, prefix + "post_attention_layernorm.weight")
                gate_w = self._load_tensor_from_reader(reader, prefix + "mlp.gate_proj.weight")
                up_w = self._load_tensor_from_reader(reader, prefix + "mlp.up_proj.weight")
                down_w = self._load_tensor_from_reader(reader, prefix + "mlp.down_proj.weight")

                weights_ptr.contents.attn_norm_w[i] = attn_norm_w.lib_tensor()
                weights_ptr.contents.attn_q_w[i] = q_w.lib_tensor()
                weights_ptr.contents.attn_q_b[i] = q_b.lib_tensor()
                weights_ptr.contents.attn_k_w[i] = k_w.lib_tensor()
                weights_ptr.contents.attn_k_b[i] = k_b.lib_tensor()
                weights_ptr.contents.attn_v_w[i] = v_w.lib_tensor()
                weights_ptr.contents.attn_v_b[i] = v_b.lib_tensor()
                weights_ptr.contents.attn_o_w[i] = o_w.lib_tensor()

                weights_ptr.contents.mlp_norm_w[i] = mlp_norm_w.lib_tensor()
                weights_ptr.contents.mlp_gate_w[i] = gate_w.lib_tensor()
                weights_ptr.contents.mlp_up_w[i] = up_w.lib_tensor()
                weights_ptr.contents.mlp_down_w[i] = down_w.lib_tensor()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        if max_new_tokens is None:
            max_new_tokens = 0

        tokens = list(inputs)
        maxseq = len(tokens) + max_new_tokens
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model, ctypes.c_size_t(maxseq))

        for _ in range(max_new_tokens):
            arr = (ctypes.c_int64 * len(tokens))(*tokens)
            next_id = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, arr, ctypes.c_size_t(len(tokens))
            )
            tokens.append(int(next_id))
            if next_id == self._meta.end_token:
                break

        return tokens
