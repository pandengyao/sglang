# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, List, Optional, Tuple, Type, Union

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        print(f"🔧 [TRY_LOAD_MODEL] Trying to load model class for: {model_arch}")
        print(f"🔧 [TRY_LOAD_MODEL] Available models: {list(self.models.keys())}")
        
        if model_arch not in self.models:
            print(f"🔧 [TRY_LOAD_MODEL] Model {model_arch} not found in registry")
            return None

        model_cls = self.models[model_arch]
        print(f"🔧 [TRY_LOAD_MODEL] Found model class: {model_cls}")
        print(f"🔧 [TRY_LOAD_MODEL] Model class name: {model_cls.__name__ if hasattr(model_cls, '__name__') else 'Unknown'}")
        print(f"🔧 [TRY_LOAD_MODEL] Model class module: {model_cls.__module__ if hasattr(model_cls, '__module__') else 'Unknown'}")
        
        return model_cls

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        print(f"🔧 [NORMALIZE_ARCHS] Starting architecture normalization")
        print(f"🔧 [NORMALIZE_ARCHS] Input architectures: {architectures}")
        print(f"🔧 [NORMALIZE_ARCHS] Available models: {list(self.models.keys())}")
        
        if isinstance(architectures, str):
            architectures = [architectures]
            print(f"🔧 [NORMALIZE_ARCHS] Converted string to list: {architectures}")
        
        if not architectures:
            logger.warning("No model architectures are specified")
            print(f"🔧 [NORMALIZE_ARCHS] No architectures specified")

        # filter out support architectures
        normalized_arch = list(
            filter(lambda model: model in self.models, architectures)
        )
        print(f"🔧 [NORMALIZE_ARCHS] Filtered architectures: {normalized_arch}")

        # make sure Transformers backend is put at the last as a fallback
        if len(normalized_arch) != len(architectures):
            print(f"🔧 [NORMALIZE_ARCHS] Adding TransformersForCausalLM as fallback")
            normalized_arch.append("TransformersForCausalLM")
        
        print(f"🔧 [NORMALIZE_ARCHS] Final normalized architectures: {normalized_arch}")
        return normalized_arch

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        print(f"🔧 [MODEL_REGISTRY] Starting model class resolution")
        print(f"🔧 [MODEL_REGISTRY] Input architectures: {architectures}")
        
        architectures = self._normalize_archs(architectures)
        print(f"🔧 [MODEL_REGISTRY] Normalized architectures: {architectures}")

        for arch in architectures:
            print(f"🔧 [MODEL_REGISTRY] Trying architecture: {arch}")
            model_cls = self._try_load_model_cls(arch)
            print(f"🔧 [MODEL_REGISTRY] Model class for {arch}: {model_cls}")
            if model_cls is not None:
                print(f"🔧 [MODEL_REGISTRY] Found model class: {model_cls.__name__}")
                print(f"🔧 [MODEL_REGISTRY] Model class module: {model_cls.__module__}")
                return (model_cls, arch)

        print(f"🔧 [MODEL_REGISTRY] No model class found, raising error")
        return self._raise_for_unsupported(architectures)


@lru_cache()
def import_model_classes():
    print(f"🔧 [IMPORT_MODELS] Starting model classes import")
    model_arch_name_to_cls = {}
    package_name = "sglang.srt.models"
    print(f"🔧 [IMPORT_MODELS] Package name: {package_name}")
    
    package = importlib.import_module(package_name)
    print(f"🔧 [IMPORT_MODELS] Package imported: {package}")
    
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        print(f"🔧 [IMPORT_MODELS] Processing module: {name}, is_package: {ispkg}")
        if not ispkg:
            try:
                module = importlib.import_module(name)
                print(f"🔧 [IMPORT_MODELS] Successfully imported module: {name}")
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. " f"{e}")
                print(f"🔧 [IMPORT_MODELS] Failed to import module {name}: {e}")
                continue
            
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                print(f"🔧 [IMPORT_MODELS] Found EntryClass in {name}: {entry}")
                
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    print(f"🔧 [IMPORT_MODELS] EntryClass is a list with {len(entry)} items")
                    for tmp in entry:
                        print(f"🔧 [IMPORT_MODELS] Processing list item: {tmp.__name__}")
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                        print(f"🔧 [IMPORT_MODELS] Added {tmp.__name__} to registry")
                else:
                    print(f"🔧 [IMPORT_MODELS] EntryClass is a single class: {entry.__name__}")
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry
                    print(f"🔧 [IMPORT_MODELS] Added {entry.__name__} to registry")
            else:
                print(f"🔧 [IMPORT_MODELS] No EntryClass found in module: {name}")

    print(f"🔧 [IMPORT_MODELS] Import completed. Total models: {len(model_arch_name_to_cls)}")
    print(f"🔧 [IMPORT_MODELS] Registered models: {list(model_arch_name_to_cls.keys())}")
    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry(import_model_classes())
