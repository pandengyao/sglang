# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/moe_wna16.py

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.utils import get_device_capability, set_weight_attrs

logger = logging.getLogger(__name__)


class MoeWNA16Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
        full_config: Dict[str, Any],
    ) -> None:
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] Initializing MoeWNA16Config")
                print(f"🔧 [MOE_WNA16_CONFIG] Parameters:")
                print(f"🔧 [MOE_WNA16_CONFIG]   - linear_quant_method: {linear_quant_method}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - weight_bits: {weight_bits}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - group_size: {group_size}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - has_zp: {has_zp}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - lm_head_quantized: {lm_head_quantized}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - modules_to_not_convert: {modules_to_not_convert}")
        except:
            pass
        
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        self.use_marlin = False
        # Avoid circular import

        if self.linear_quant_method == "gptq":
            self.use_marlin = GPTQMarlinConfig.is_gptq_marlin_compatible(full_config)
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] GPTQ method - use_marlin: {self.use_marlin}")
            except:
                pass
        elif self.linear_quant_method == "awq":
            capability_tuple = get_device_capability()
            device_capability = (
                -1
                if capability_tuple is None
                else capability_tuple[0] * 10 + capability_tuple[1]
            )
            awq_min_capability = AWQConfig.get_min_capability()
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] AWQ method - device_capability: {device_capability}, min_capability: {awq_min_capability}")
            except:
                pass
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert
        
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] MoeWNA16Config initialized successfully")
                print(f"🔧 [MOE_WNA16_CONFIG] Final config:")
                print(f"🔧 [MOE_WNA16_CONFIG]   - bit8_pack_factor: {self.bit8_pack_factor}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - use_marlin: {self.use_marlin}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - modules_to_not_convert count: {len(self.modules_to_not_convert)}")
        except:
            pass

    @classmethod
    def get_name(cls) -> str:
        return "moe_wna16"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MoeWNA16Config":
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] Creating config from config dict")
                print(f"🔧 [MOE_WNA16_CONFIG] Config keys: {list(config.keys())}")
        except:
            pass
        
        quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] Extracted parameters:")
                print(f"🔧 [MOE_WNA16_CONFIG]   - quant_method: {quant_method}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - weight_bits: {weight_bits}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - group_size: {group_size}")
                print(f"🔧 [MOE_WNA16_CONFIG]   - lm_head_quantized: {lm_head_quantized}")
        except:
            pass
        
        if quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] GPTQ config - has_zp: {has_zp}")
            except:
                pass
        elif quant_method == "awq":
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] AWQ config - has_zp: {has_zp}, modules_to_not_convert: {modules_to_not_convert}")
            except:
                pass
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] Creating MoeWNA16Config instance")
        except:
            pass
        
        return cls(
            quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant == "moe_wna16" and cls.is_moe_wna16_compatible(hf_quant_cfg):
            return cls.get_name()
        return None

    @classmethod
    def is_moe_wna16_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act")

        capability_tuple = get_device_capability()
        device_capability = (
            -1
            if all(capability is None for capability in capability_tuple)
            else capability_tuple[0] * 10 + capability_tuple[1]
        )
        # Avoid circular import
        awq_min_capability = AWQConfig.get_min_capability()

        gptq_compatible = quant_method == "gptq" and not desc_act and num_bits in [4, 8]
        awq_compatible = (
            quant_method == "awq"
            and num_bits == 4
            and device_capability >= awq_min_capability
        )

        return gptq_compatible or awq_compatible

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] Getting quant method for layer: {type(layer).__name__}, prefix: {prefix}")
        except:
            pass
        
        # avoid circular import
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] Layer skipped for quantization - returning UnquantizedLinearMethod")
            except:
                pass
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] LinearBase layer detected - using {self.linear_quant_method} method")
            except:
                pass

            if self.linear_quant_method == "gptq":
                if self.use_marlin:
                    try:
                        current_tp_rank = get_tensor_model_parallel_rank()
                        if current_tp_rank == 0:
                            print(f"🔧 [MOE_WNA16_CONFIG] Using GPTQ Marlin method")
                    except:
                        pass
                    return GPTQMarlinConfig.from_config(
                        self.full_config
                    ).get_quant_method(layer, prefix)
                else:
                    try:
                        current_tp_rank = get_tensor_model_parallel_rank()
                        if current_tp_rank == 0:
                            print(f"🔧 [MOE_WNA16_CONFIG] Using GPTQ method")
                    except:
                        pass
                    return GPTQConfig.from_config(self.full_config).get_quant_method(
                        layer, prefix
                    )
            elif self.linear_quant_method == "awq":
                try:
                    current_tp_rank = get_tensor_model_parallel_rank()
                    if current_tp_rank == 0:
                        print(f"🔧 [MOE_WNA16_CONFIG] Using AWQ method")
                except:
                    pass
                return AWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16_CONFIG] FusedMoE layer detected - returning MoeWNA16Method")
            except:
                pass
            return MoeWNA16Method(self)
        
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16_CONFIG] No matching quant method found - returning None")
        except:
            pass
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeWNA16Method:
    """Linear method for MOE WNA16 (W8A16/W4A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.
    """

    def __new__(cls, *args, **kwargs):
        # avoid circular import
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16] Creating MoeWNA16Method instance with dynamic inheritance from FusedMoEMethodBase")
        except:
            pass

        if not hasattr(cls, "_initialized"):
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16] First time initialization - creating dynamic class composition")
            except:
                pass
            # 保存原始的 __init__ 方法
            original_init = cls.__init__
            
            # 动态创建新类，继承自 FusedMoEMethodBase
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),  # 父类
                {
                    "__init__": original_init,  # 保持原始初始化方法
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},  # 复制其他方法
                },
            )
            
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16] Dynamic class created: {new_cls.__name__} inheriting from {FusedMoEMethodBase.__name__}")
            except:
                pass
            
            # 创建新类的实例并初始化
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            
            try:
                current_tp_rank = get_tensor_model_parallel_rank()
                if current_tp_rank == 0:
                    print(f"🔧 [MOE_WNA16] MoeWNA16Method instance created successfully with quant_config: {args[0].__class__.__name__ if args else 'None'}")
            except:
                pass
            
            return obj
        
        # 如果已经初始化过，使用正常的实例化流程
        try:
            current_tp_rank = get_tensor_model_parallel_rank()
            if current_tp_rank == 0:
                print(f"🔧 [MOE_WNA16] Using cached initialization - class already composed")
        except:
            pass
        return super().__new__(cls)

    def __init__(self, quant_config: MoeWNA16Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()
        
        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Starting weight creation for {type(layer).__name__}")
            print(f"🔧 [WNA16_MOE_CREATE] Parameters:")
            print(f"🔧 [WNA16_MOE_CREATE]   - num_experts: {num_experts}")
            print(f"🔧 [WNA16_MOE_CREATE]   - hidden_size: {hidden_size}")
            print(f"🔧 [WNA16_MOE_CREATE]   - intermediate_size_per_partition: {intermediate_size_per_partition}")
            print(f"🔧 [WNA16_MOE_CREATE]   - params_dtype: {params_dtype}")
            print(f"🔧 [WNA16_MOE_CREATE]   - extra_weight_attrs keys: {list(extra_weight_attrs.keys())}")
        
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Quantization config:")
            print(f"🔧 [WNA16_MOE_CREATE]   - bit8_pack_factor: {bit8_pack_factor}")
            print(f"🔧 [WNA16_MOE_CREATE]   - group_size: {group_size}")
            print(f"🔧 [WNA16_MOE_CREATE]   - weight_bits: {self.quant_config.weight_bits}")
            print(f"🔧 [WNA16_MOE_CREATE]   - has_zp: {self.quant_config.has_zp}")

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Adjusted group size:")
            print(f"🔧 [WNA16_MOE_CREATE]   - final group_size: {group_size}")
            print(f"🔧 [WNA16_MOE_CREATE]   - group_size_div_factor: {group_size_div_factor}")

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = MoeWNA16Method.get_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Creating w13_qweight parameter")
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] w13_qweight shape: {w13_qweight.shape}")

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Creating w2_qweight parameter")
        # Fused down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] w2_qweight shape: {w2_qweight.shape}")

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Creating w13_scales parameter")
        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] w13_scales shape: {w13_scales.shape}")

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Creating w2_scales parameter")
        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] w2_scales shape: {w2_scales.shape}")

        if self.quant_config.has_zp:
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] Creating zero point parameters (has_zp=True)")
                print(f"🔧 [WNA16_MOE_CREATE] Creating w13_qzeros parameter")
            w13_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition // bit8_pack_factor,
                    hidden_size // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] w13_qzeros shape: {w13_qzeros.shape}")

            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] Creating w2_qzeros parameter")
            w2_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // bit8_pack_factor,
                    intermediate_size_per_partition // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] w2_qzeros shape: {w2_qzeros.shape}")
        else:
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] Skipping zero point parameters (has_zp=False)")

        if self.quant_config.linear_quant_method == "gptq":
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] Creating GPTQ-specific parameters")
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys = ["w13_g_idx", "w2_g_idx"]
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_CREATE] Creating dummy parameter: {key}")
                param = torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int32), requires_grad=False
                )
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)
        else:
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_CREATE] Linear quant method: {self.quant_config.linear_quant_method}")

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE_CREATE] Weight creation completed for {type(layer).__name__}")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        # avoid circular import
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        
        tp_rank = get_tensor_model_parallel_rank()

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE] Starting WNA16 MoE inference")
            print(f"🔧 [WNA16_MOE] Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"🔧 [WNA16_MOE] Router logits shape: {router_logits.shape}")
            print(f"🔧 [WNA16_MOE] Top-k: {top_k}, activation: {activation}")
            print(f"🔧 [WNA16_MOE] Weight bits: {self.quant_config.weight_bits}")
            print(f"🔧 [WNA16_MOE] Has zero points: {self.quant_config.has_zp}")
            print(f"🔧 [WNA16_MOE] Linear quant method: {self.quant_config.linear_quant_method}")

        assert activation == "silu", "Only SiLU activation is supported."
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE] Expert selection completed")
            print(f"🔧 [WNA16_MOE] Top-k weights shape: {topk_weights.shape}")
            print(f"🔧 [WNA16_MOE] Top-k IDs shape: {topk_ids.shape}")
            print(f"🔧 [WNA16_MOE] W13 qweight shape: {layer.w13_qweight.shape}")
            print(f"🔧 [WNA16_MOE] W2 qweight shape: {layer.w2_qweight.shape}")
            print(f"🔧 [WNA16_MOE] W13 scales shape: {layer.w13_scales.shape}")
            print(f"🔧 [WNA16_MOE] W2 scales shape: {layer.w2_scales.shape}")
            if self.quant_config.has_zp:
                print(f"🔧 [WNA16_MOE] W13 qzeros shape: {layer.w13_qzeros.shape}")
                print(f"🔧 [WNA16_MOE] W2 qzeros shape: {layer.w2_qzeros.shape}")

        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE] Calling fused_experts with dequantization parameters:")
            print(f"🔧 [WNA16_MOE]   - use_int4_w4a16: {weight_bits == 4}")
            print(f"🔧 [WNA16_MOE]   - use_int8_w8a16: {weight_bits == 8}")
            print(f"🔧 [WNA16_MOE]   - block_shape: [0, {layer.group_size}]")
            print(f"🔧 [WNA16_MOE]   - has_zp: {has_zp}")

        result = fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if has_zp else None,
            w2_zp=layer.w2_qzeros if has_zp else None,
            block_shape=[0, layer.group_size],
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

        if tp_rank == 0:
            print(f"🔧 [WNA16_MOE] Fused experts completed, output shape: {result.shape}")
        return result

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            from sglang.srt.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            
            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_LOADER] Loading weight: {weight_name}")
                print(f"🔧 [WNA16_MOE_LOADER] Shard ID: {shard_id}, Expert ID: {expert_id}")
                print(f"🔧 [WNA16_MOE_LOADER] Loaded weight shape: {loaded_weight.shape}, dtype: {loaded_weight.dtype}")
            
            if "g_idx" in weight_name:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Skipping g_idx parameter")
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Skipping qzeros parameter (no zero points)")
                return

            device = get_tp_group().device
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_LOADER] Converting weight for method: {layer.quant_config.linear_quant_method}")
                print(f"🔧 [WNA16_MOE_LOADER] Weight bits: {layer.quant_config.weight_bits}")

            # convert gptq and awq weight to a standard format
            if layer.quant_config.linear_quant_method == "awq":
                assert layer.quant_config.weight_bits == 4
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Converting AWQ tensor")
                if "weight" in weight_name:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Converting AWQ qweight")
                    loaded_weight = convert_awq_tensor(loaded_weight, "qweight")
                elif "zeros" in weight_name:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Converting AWQ qzeros")
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Transposing non-weight/zeros tensor")
                    loaded_weight = loaded_weight.T
            elif layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [4, 8]
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Converting GPTQ tensor")
                if "weight" in weight_name:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Converting GPTQ qweight")
                    loaded_weight = loaded_weight.T.contiguous().view(torch.uint8)
                elif "zeros" in weight_name:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Converting GPTQ qzeros")
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        if tp_rank == 0:
                            print(f"🔧 [WNA16_MOE_LOADER] Converting GPTQ int4 qzeros")
                        loaded_weight = convert_gptq_int4_qzeros(loaded_weight).T
                    else:
                        if tp_rank == 0:
                            print(f"🔧 [WNA16_MOE_LOADER] Converting GPTQ int8 qzeros")
                        loaded_weight = loaded_weight.T + 1
                else:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Transposing non-weight/zeros tensor")
                    loaded_weight = loaded_weight.T

            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_LOADER] After conversion shape: {loaded_weight.shape}")

            # repeat the qzeros/scales to fit new group size
            if (
                layer.group_size_div_factor > 1
                and "qzeros" in weight_name
                or "scales" in weight_name
            ):
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Repeating tensor for group size adjustment")
                    print(f"🔧 [WNA16_MOE_LOADER] Group size div factor: {layer.group_size_div_factor}")
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1
                )
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] After repeat shape: {loaded_weight.shape}")

            if "w13_qzeros" in weight_name:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Processing w13_qzeros")
                tensor = loaded_weight.view(layer.tp_size, -1, loaded_weight.size(1))[
                    tp_rank
                ]
                if shard_id == "w1":
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Assigning to w1 part of w13")
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    if tp_rank == 0:
                        print(f"🔧 [WNA16_MOE_LOADER] Assigning to w3 part of w13")
                    param.data[expert_id, shard_size // 2 :] = tensor
            elif "w2_qzeros" in weight_name:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Processing w2_qzeros")
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.tp_size, -1
                )[:, tp_rank]
            else:
                if tp_rank == 0:
                    print(f"🔧 [WNA16_MOE_LOADER] Using default weight loader")
                weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

            if tp_rank == 0:
                print(f"🔧 [WNA16_MOE_LOADER] Weight loading completed for {weight_name}")

        return moe_wna16_weight_loader
