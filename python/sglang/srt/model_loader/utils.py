# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import contextlib
import logging
from typing import Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang.srt.configs.model_config import ModelConfig, ModelImpl

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def resolve_transformers_arch(model_config: ModelConfig, architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
            continue
        auto_map: dict[str, str] = (
            getattr(model_config.hf_config, "auto_map", None) or dict()
        )
        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        auto_modules = {
            name: get_class_from_dynamic_module(
                module, model_config.model_path, revision=model_config.revision
            )
            for name, module in sorted(auto_map.items(), key=lambda x: x[0])
        }
        model_module = getattr(transformers, arch, None)
        if model_module is None:
            if "AutoModel" not in auto_map:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom)."
                )
            model_module = auto_modules["AutoModel"]
        if model_config.impl == ModelImpl.TRANSFORMERS:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM."
                )
            architectures[i] = "TransformersForCausalLM"
        if model_config.impl == ModelImpl.AUTO:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"{arch} has no SGlang implementation and the Transformers "
                    "implementation is not compatible with SGLang."
                )
            logger.warning(
                "%s has no SGLang implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )
            architectures[i] = "TransformersForCausalLM"
    return architectures


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    # Get TP rank from current context
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    tp_rank = get_tensor_model_parallel_rank()

    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Starting model architecture resolution")
        print(f"🔧 [GET_MODEL_ARCH] Model config: {model_config}")
    
    architectures = getattr(model_config.hf_config, "architectures", [])
    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Original architectures: {architectures}")
    
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"]
    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Mixtral supported quantizations: {mixtral_supported}")
        print(f"🔧 [GET_MODEL_ARCH] Model quantization: {model_config.quantization}")

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        if tp_rank == 0:
            print(f"🔧 [GET_MODEL_ARCH] Special case: Using QuantMixtralForCausalLM for quantized Mixtral")
        architectures = ["QuantMixtralForCausalLM"]
        if tp_rank == 0:
            print(f"🔧 [GET_MODEL_ARCH] Updated architectures: {architectures}")

    supported_archs = ModelRegistry.get_supported_archs()
    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Supported architectures: {supported_archs}")
    
    is_native_supported = any(arch in supported_archs for arch in architectures)
    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Is native supported: {is_native_supported}")
        print(f"🔧 [GET_MODEL_ARCH] Model implementation: {model_config.impl}")

    if not is_native_supported or model_config.impl == ModelImpl.TRANSFORMERS:
        if tp_rank == 0:
            print(f"🔧 [GET_MODEL_ARCH] Using transformers backend, calling resolve_transformers_arch")
        architectures = resolve_transformers_arch(model_config, architectures)
        if tp_rank == 0:
            print(f"🔧 [GET_MODEL_ARCH] Resolved architectures: {architectures}")

    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] Calling ModelRegistry.resolve_model_cls with architectures: {architectures}")
    result = ModelRegistry.resolve_model_cls(architectures)
    if tp_rank == 0:
        print(f"🔧 [GET_MODEL_ARCH] ModelRegistry result: {result}")
        print(f"🔧 [GET_MODEL_ARCH] Resolved model class: {result[0].__name__}")
        print(f"🔧 [GET_MODEL_ARCH] Resolved architecture name: {result[1]}")
    
    return result


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
