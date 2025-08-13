# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch

from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from sglang.srt.layers.moe.topk import select_experts


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    """
    将4位整数值打包成8位整数
    用于W4A8量化中的权重压缩，将两个4位值合并为一个8位值
    """
    print(f"[pack_int4_values_to_int8] Input shape: {int4_values_interleaved.shape}, dtype: {int4_values_interleaved.dtype}")
    
    # 检查最后一个维度是否为偶数（因为需要两个4位值打包成一个8位值）
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    # 转换为int8类型
    input_tensor_int8 = int4_values_interleaved.to(torch.int8)
    print(f"[pack_int4_values_to_int8] Converted to int8: shape: {input_tensor_int8.shape}, dtype: {input_tensor_int8.dtype}")

    # 提取低4位和高4位：每隔一个位置取一个值
    low_nibbles = input_tensor_int8[..., 0::2]   # 取偶数位置的值作为低4位
    high_nibbles = input_tensor_int8[..., 1::2]  # 取奇数位置的值作为高4位
    print(f"[pack_int4_values_to_int8] Low nibbles: shape: {low_nibbles.shape}, dtype: {low_nibbles.dtype}")
    print(f"[pack_int4_values_to_int8] High nibbles: shape: {high_nibbles.shape}, dtype: {high_nibbles.dtype}")

    # 位运算打包：高4位左移4位，低4位与0x0F相与，然后或运算合并
    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    print(f"[pack_int4_values_to_int8] Packed tensor: shape: {packed_tensor.shape}, dtype: {packed_tensor.dtype}")

    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    """
    对权重和尺度进行交织打包
    实现W4A8量化中的内存优化，提高GPU访问效率
    """
    print(f"[pack_interleave] Input ref_weight: shape: {ref_weight.shape}, dtype: {ref_weight.dtype}")
    print(f"[pack_interleave] Input ref_scale: shape: {ref_scale.shape}, dtype: {ref_scale.dtype}")
    print(f"[pack_interleave] num_experts: {num_experts}")
    
    # 获取隐藏层大小和中间层大小
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    print(f"[pack_interleave] n (hidden_size): {n}, k (intermediate_size): {k}")

    # 将权重从CPU打包后移到GPU
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    print(f"[pack_interleave] After pack_int4_values_to_int8: shape: {weight.shape}, dtype: {weight.dtype}")
    
    # 重塑权重张量：将打包后的权重重新组织为专家格式
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()  # 确保内存连续
    print(f"[pack_interleave] Final w_q: shape: {w_q.shape}, dtype: {w_q.dtype}")

    # 尺度交织重排：优化内存访问模式
    # 第一步：将尺度重塑为4组
    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0], ref_scale.shape[1], (ref_scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    print(f"[pack_interleave] Scale reshape 1: shape: {scale_interleaved.shape}, dtype: {scale_interleaved.dtype}")
    
    # 第二步：维度置换，实现交织
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    print(f"[pack_interleave] Scale permute: shape: {scale_interleaved.shape}, dtype: {scale_interleaved.dtype}")
    
    # 第三步：重塑为最终格式
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0], ref_scale.shape[2] // 4, ref_scale.shape[1] * 4
    )  # [E, K/4, N*4]
    print(f"[pack_interleave] Scale reshape 2: shape: {scale_interleaved.shape}, dtype: {scale_interleaved.dtype}")
    
    w_scale = scale_interleaved.contiguous()  # 确保内存连续
    print(f"[pack_interleave] Final w_scale: shape: {w_scale.shape}, dtype: {w_scale.dtype}")

    return w_q, w_scale


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])       # 批次大小
@pytest.mark.parametrize("N", [2048])                 # 隐藏层大小
@pytest.mark.parametrize("K", [7168])                 # 中间层大小
@pytest.mark.parametrize("E", [256])                  # 专家数量
@pytest.mark.parametrize("ep_size", [8])              # 专家并行大小
@pytest.mark.parametrize("topk", [8])                 # Top-K选择
@pytest.mark.parametrize("group_size", [128])         # 量化组大小
@pytest.mark.parametrize("dtype", [torch.bfloat16])   # 数据类型
def test_cutlass_w4a8_moe(M, N, K, E, ep_size, topk, group_size, dtype):
    """
    测试CUTLASS W4A8 MoE实现
    验证量化MoE的正确性和性能
    """
    print(f"\n[test_cutlass_w4a8_moe] Test parameters:")
    print(f"  M (batch_size): {M}, N (hidden_size): {N}, K (intermediate_size): {K}")
    print(f"  E (num_experts): {E}, ep_size (expert_parallel_size): {ep_size}")
    print(f"  topk: {topk}, group_size: {group_size}, dtype: {dtype}")
    
    # 计算本地专家数量（专家并行分片）
    local_e = E // ep_size    # 256 // 8 = 32
    print(f"[test_cutlass_w4a8_moe] local_e (local_experts): {local_e}")

    debug = False
    if debug:
        # 调试模式：使用固定值进行测试
        a = torch.ones((M, K), dtype=dtype, device="cuda") * 0.001
        ref_weight_1 = torch.ones((local_e, N * 2, K), dtype=torch.int8, device="cuda")
        ref_weight_2 = torch.ones((local_e, K, N), dtype=torch.int8, device="cuda")
        a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        scale_1 = torch.ones(
            (local_e, N * 2, K // group_size), dtype=dtype, device="cuda"
        )
        scale_2 = torch.ones((local_e, K, N // group_size), dtype=dtype, device="cuda")
    else:
        # 正常模式：使用随机值进行测试
        # 输入张量：批次大小 x 中间层大小
        a = torch.randn(M, K, dtype=dtype, device="cuda")   # [M, 7168]
        print(f"[test_cutlass_w4a8_moe] Input tensor a: shape: {a.shape}, dtype: {a.dtype}")
        
        # 第一个权重矩阵：本地专家数 x (隐藏层大小*2) x 中间层大小
        # 包含gate和up投影的融合权重
        ref_weight_1 = torch.randint(
            -8, 8, (local_e, N * 2, K), dtype=torch.int8, device="cuda"
        )
        print(f"[test_cutlass_w4a8_moe] ref_weight_1: shape: {ref_weight_1.shape}, dtype: {ref_weight_1.dtype}")
        
        # 第二个权重矩阵：本地专家数 x 中间层大小 x 隐藏层大小
        # down投影权重
        ref_weight_2 = torch.randint(
            -8, 8, (local_e, K, N), dtype=torch.int8, device="cuda"
        )
        print(f"[test_cutlass_w4a8_moe] ref_weight_2: shape: {ref_weight_2.shape}, dtype: {ref_weight_2.dtype}")
        
        # 激活量化尺度：用于输入激活的量化
        affine_coeff = 0.005
        a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")
        print(f"[test_cutlass_w4a8_moe] a1_scale: shape: {a1_scale.shape}, dtype: {a1_scale.dtype}")
        print(f"[test_cutlass_w4a8_moe] a2_scale: shape: {a2_scale.shape}, dtype: {a2_scale.dtype}")
        
        # 权重量化尺度：用于权重的反量化
        # 按组大小进行分组的尺度
        scale_1 = (
            torch.randn(local_e, N * 2, K // group_size, dtype=dtype, device="cuda")
            * affine_coeff
        )
        print(f"[test_cutlass_w4a8_moe] scale_1: shape: {scale_1.shape}, dtype: {scale_1.dtype}")
        
        scale_2 = (
            torch.randn(local_e, K, N // group_size, dtype=dtype, device="cuda")
            * affine_coeff
        )
        print(f"[test_cutlass_w4a8_moe] scale_2: shape: {scale_2.shape}, dtype: {scale_2.dtype}")

    # 对权重和尺度进行交织打包，优化内存访问模式
    print(f"\n[test_cutlass_w4a8_moe] Packing weights and scales...")
    w1_q, w1_scale = pack_interleave(local_e, ref_weight_1, scale_1)
    w2_q, w2_scale = pack_interleave(local_e, ref_weight_2, scale_2)
    print(f"[test_cutlass_w4a8_moe] Final w1_q: shape: {w1_q.shape}, dtype: {w1_q.dtype}")
    print(f"[test_cutlass_w4a8_moe] Final w1_scale: shape: {w1_scale.shape}, dtype: {w1_scale.dtype}")
    print(f"[test_cutlass_w4a8_moe] Final w2_q: shape: {w2_q.shape}, dtype: {w2_q.dtype}")
    print(f"[test_cutlass_w4a8_moe] Final w2_scale: shape: {w2_scale.shape}, dtype: {w2_scale.dtype}")

    device = "cuda"
    # 创建步长张量：用于CUTLASS内核的内存布局优化
    print(f"\n[test_cutlass_w4a8_moe] Creating stride tensors...")
    a_strides1 = torch.full((local_e, 3), K, device=device, dtype=torch.int64)      # 输入步长
    c_strides1 = torch.full((local_e, 3), 2 * N, device=device, dtype=torch.int64)  # 输出步长（gate+up）
    a_strides2 = torch.full((local_e, 3), N, device=device, dtype=torch.int64)      # 输入步长
    c_strides2 = torch.full((local_e, 3), K, device=device, dtype=torch.int64)      # 输出步长
    b_strides1 = a_strides1  # 权重步长
    s_strides13 = c_strides1 # 尺度步长
    b_strides2 = a_strides2  # 权重步长
    s_strides2 = c_strides2  # 尺度步长
    print(f"[test_cutlass_w4a8_moe] a_strides1: shape: {a_strides1.shape}, dtype: {a_strides1.dtype}")
    print(f"[test_cutlass_w4a8_moe] c_strides1: shape: {c_strides1.shape}, dtype: {c_strides1.dtype}")
    print(f"[test_cutlass_w4a8_moe] a_strides2: shape: {a_strides2.shape}, dtype: {a_strides2.dtype}")
    print(f"[test_cutlass_w4a8_moe] c_strides2: shape: {c_strides2.shape}, dtype: {c_strides2.dtype}")

    # 专家选择：使用路由器选择top-k专家
    print(f"\n[test_cutlass_w4a8_moe] Expert selection...")
    score = torch.randn((M, E), dtype=dtype, device=device)   # [M, 256] 路由器分数
    print(f"[test_cutlass_w4a8_moe] Router score: shape: {score.shape}, dtype: {score.dtype}")
    
    # 选择top-k专家和对应的权重
    topk_weights, topk_ids = select_experts(
        hidden_states=a,
        router_logits=score,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=False,
    )
    print(f"[test_cutlass_w4a8_moe] topk_weights: shape: {topk_weights.shape}, dtype: {topk_weights.dtype}")
    print(f"[test_cutlass_w4a8_moe] topk_ids: shape: {topk_ids.shape}, dtype: {topk_ids.dtype}")
    
    # 专家映射：将全局专家ID映射到本地专家ID
    expert_map = torch.arange(E, dtype=torch.int32, device=device)   # [0, 1, 2, ..., 255]
    expert_map[local_e:] = E    # 超出本地专家的设为E（无效标记）
    print(f"[test_cutlass_w4a8_moe] expert_map: shape: {expert_map.shape}, dtype: {expert_map.dtype}")
    print(f"[test_cutlass_w4a8_moe] expert_map values: {expert_map[:10]}... (first 10)")

    print(f"\n[test_cutlass_w4a8_moe] Running CUTLASS MoE...")
    output = cutlass_moe(
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        0,
        local_e - 1,
        E,
        a1_scale,
        a2_scale,
        expert_map,
    )
    print(f"[test_cutlass_w4a8_moe] CUTLASS output: shape: {output.shape}, dtype: {output.dtype}")

    print(f"\n[test_cutlass_w4a8_moe] Running reference implementation...")
    ref_output = ref(
        a,
        local_e,
        topk_weights,
        topk_ids,
        ref_weight_1,
        ref_weight_2,
        scale_1,
        scale_2,
        has_pre_quant=True,
        has_alpha=True,
        pre_quant_scale_1=a1_scale,
        pre_quant_scale_2=a2_scale,
        alpha_1=a1_scale,
        alpha_2=a2_scale,
    )
    print(f"[test_cutlass_w4a8_moe] Reference output: shape: {ref_output.shape}, dtype: {ref_output.dtype}")

    # compare
    torch.cuda.synchronize()

    # compare final output
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    print("SUCCESS: Final output tensors are close.")


def cutlass_moe(
    a: torch.Tensor,                 # 输入: [M, K]
    w1_q: torch.Tensor,             # 量化权重1: [local_e, N*2, K//2]
    w2_q: torch.Tensor,              # 量化权重2: [local_e, K, N//2]
    w1_scale: torch.Tensor,            # 尺度1: [local_e, K//4, N*4]
    w2_scale: torch.Tensor,            # 尺度2: [local_e, K//4, N*4]
    topk_weights: torch.Tensor,        # Top-K权重: [M, topk]
    topk_ids_: torch.Tensor,           # Top-K ID: [M, topk]
    a_strides1: torch.Tensor,          # 输入步长1: [local_e, 3]
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    E: int,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
):
    """
    CUTLASS MoE包装函数
    使用NVIDIA CUTLASS库实现高性能的量化MoE计算
    """
    print(f"[cutlass_moe] Input a: shape: {a.shape}, dtype: {a.dtype}")
    print(f"[cutlass_moe] w1_q: shape: {w1_q.shape}, dtype: {w1_q.dtype}")
    print(f"[cutlass_moe] w2_q: shape: {w2_q.shape}, dtype: {w2_q.dtype}")
    print(f"[cutlass_moe] w1_scale: shape: {w1_scale.shape}, dtype: {w1_scale.dtype}")
    print(f"[cutlass_moe] w2_scale: shape: {w2_scale.shape}, dtype: {w2_scale.dtype}")
    print(f"[cutlass_moe] topk_weights: shape: {topk_weights.shape}, dtype: {topk_weights.dtype}")
    print(f"[cutlass_moe] topk_ids_: shape: {topk_ids_.shape}, dtype: {topk_ids_.dtype}")
    print(f"[cutlass_moe] start_expert_id: {start_expert_id}, end_expert_id: {end_expert_id}, E: {E}")
    
    # 专家ID映射：将全局专家ID转换为本地专家ID
    local_topk_ids = topk_ids_
    local_topk_ids = torch.where(expert_map[topk_ids_] != E, expert_map[topk_ids_], E)
    print(f"[cutlass_moe] local_topk_ids: shape: {local_topk_ids.shape}, dtype: {local_topk_ids.dtype}")
    print(f"[cutlass_moe] local_topk_ids values: {local_topk_ids[:5]}... (first 5)")
    
    device = a.device

    # 计算本地专家数量
    local_num_experts = end_expert_id - start_expert_id + 1
    print(f"[cutlass_moe] local_num_experts: {local_num_experts}")
    
    # 为CUTLASS内核准备辅助张量
    expert_offsets = torch.empty(
        (local_num_experts + 1), dtype=torch.int32, device=device
    )  # 专家偏移量
    problem_sizes1 = torch.empty(
        (local_num_experts, 3), dtype=torch.int32, device=device
    )  # 第一个GEMM的问题大小
    problem_sizes2 = torch.empty(
        (local_num_experts, 3), dtype=torch.int32, device=device
    )  # 第二个GEMM的问题大小
    print(f"[cutlass_moe] expert_offsets: shape: {expert_offsets.shape}, dtype: {expert_offsets.dtype}")
    print(f"[cutlass_moe] problem_sizes1: shape: {problem_sizes1.shape}, dtype: {problem_sizes1.dtype}")
    print(f"[cutlass_moe] problem_sizes2: shape: {problem_sizes2.shape}, dtype: {problem_sizes2.dtype}")
    return cutlass_w4a8_moe(
        start_expert_id,
        end_expert_id,
        E,
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids_,
        local_topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input,
    )


def ref(
    x: torch.Tensor,
    num_experts: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ref_weight_1: torch.Tensor,
    ref_weight_2: torch.Tensor,
    ref_weight_scale_1: torch.Tensor,
    ref_weight_scale_2: torch.Tensor,
    has_pre_quant: bool = False,
    has_alpha: bool = False,
    pre_quant_scale_1: Optional[torch.Tensor] = None,
    pre_quant_scale_2: Optional[torch.Tensor] = None,
    alpha_1: Optional[torch.Tensor] = None,
    alpha_2: Optional[torch.Tensor] = None,
):
    """
    参考实现：传统的MoE计算方式
    用于验证CUTLASS实现的正确性
    """
    print(f"[ref] Input x: shape: {x.shape}, dtype: {x.dtype}")
    print(f"[ref] num_experts: {num_experts}")
    print(f"[ref] topk_weights: shape: {topk_weights.shape}, dtype: {topk_weights.dtype}")
    print(f"[ref] topk_ids: shape: {topk_ids.shape}, dtype: {topk_ids.dtype}")
    print(f"[ref] ref_weight_1: shape: {ref_weight_1.shape}, dtype: {ref_weight_1.dtype}")
    print(f"[ref] ref_weight_2: shape: {ref_weight_2.shape}, dtype: {ref_weight_2.dtype}")
    print(f"[ref] ref_weight_scale_1: shape: {ref_weight_scale_1.shape}, dtype: {ref_weight_scale_1.dtype}")
    print(f"[ref] ref_weight_scale_2: shape: {ref_weight_scale_2.shape}, dtype: {ref_weight_scale_2.dtype}")
    
    # 初始化结果张量
    results = torch.zeros_like(x)
    dtype = x.dtype
    print(f"[ref] results: shape: {results.shape}, dtype: {results.dtype}")
    
    # 遍历每个专家
    for e_idx in range(num_experts):
        if e_idx == 0:  # 只打印第一个专家的详细信息
            print(f"[ref] Processing expert {e_idx}...")
        
        # 创建专家掩码：找出分配给当前专家的token
        mask = topk_ids == e_idx
        activated_tokens = mask.sum(1).bool()  # 哪些token被激活
        act = x[activated_tokens, :]  # 提取激活的token
        
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - activated_tokens: {activated_tokens.sum()}, act: shape: {act.shape}, dtype: {act.dtype}")
        
        # 如果没有token分配给这个专家，跳过
        if act.shape[0] == 0:
            continue
            
        # 计算最终的权重缩放因子
        final_scale = (topk_weights * mask).sum(1)[activated_tokens].unsqueeze(1)
        
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - final_scale: shape: {final_scale.shape}, dtype: {final_scale.dtype}")

        # 输入激活量化：将激活值量化为FP8格式
        act = (
            torch.clamp((act / pre_quant_scale_1.float()), -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - act after pre_quant: shape: {act.shape}, dtype: {act.dtype}")
        
        # 第一个线性层：gate + up投影
        w3_w1 = ref_weight_1[e_idx]  # 获取当前专家的权重
        ref_w_scale_repeat = (
            ref_weight_scale_1[e_idx].repeat_interleave(128, dim=1).to(float)
        )  # 重复尺度以匹配权重维度
        w3_w1 = (w3_w1.to(float) * ref_w_scale_repeat).to(dtype)  # 反量化权重
        fc1 = ((torch.matmul(act, w3_w1.T)) * alpha_1).to(torch.float16)  # 矩阵乘法
        
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - w3_w1: shape: {w3_w1.shape}, dtype: {w3_w1.dtype}")
            print(f"[ref] Expert {e_idx} - fc1: shape: {fc1.shape}, dtype: {fc1.dtype}")

        # SwiGLU激活：分离gate和up，应用SiLU激活
        gate, fc1 = fc1.chunk(2, dim=-1)  # 分离gate和up部分
        fc1 = fc1 * torch.nn.functional.silu(gate)  # 应用SiLU激活
        act = (fc1 / pre_quant_scale_2.float()).to(torch.float8_e4m3fn)  # 再次量化
        act = act.to(dtype)
        
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - act after fc1: shape: {act.shape}, dtype: {act.dtype}")

        # 第二个线性层：down投影
        w2 = ref_weight_2[e_idx]  # 获取down投影权重
        ref_w_scale_repeat = (
            ref_weight_scale_2[e_idx].repeat_interleave(128, dim=1).to(float)
        )  # 重复尺度
        w2 = (w2.to(float) * ref_w_scale_repeat).to(dtype)  # 反量化权重
        fc2 = (torch.matmul(act, w2.T) * alpha_2).to(torch.float16)  # 矩阵乘法
        
        if e_idx == 0:
            print(f"[ref] Expert {e_idx} - w2: shape: {w2.shape}, dtype: {w2.dtype}")
            print(f"[ref] Expert {e_idx} - fc2: shape: {fc2.shape}, dtype: {fc2.dtype}")

        # 累加结果：将当前专家的输出加权累加到结果中
        results[activated_tokens, :] += (fc2 * final_scale).to(results.dtype)

    print(f"[ref] Final results: shape: {results.shape}, dtype: {results.dtype}")
    return results


if __name__ == "__main__":
    print("="*80)
    print("开始运行 CUTLASS W4A8 MoE 测试")
    print("="*80)
    
    # 单个测试参数
    M = 2        # 批次大小
    N = 2048     # 隐藏层大小
    K = 7168     # 中间层大小
    E = 256      # 专家数量
    ep_size = 8  # 专家并行大小
    topk = 8     # Top-K选择
    group_size = 128  # 量化组大小
    dtype = torch.bfloat16  # 数据类型
    
    print(f"测试参数: M={M}, N={N}, K={K}, E={E}, ep_size={ep_size}, topk={topk}, group_size={group_size}, dtype={dtype}")
    print("="*80)
    
    try:
        test_cutlass_w4a8_moe(M, N, K, E, ep_size, topk, group_size, dtype)
        print("✓ 测试通过")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
