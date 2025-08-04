import pytest
import torch
from sgl_kernel import cutlass_w4a8_moe_mm


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
    
    # 获取输出维度和输入维度
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    print(f"[pack_interleave] n (output_dim): {n}, k (input_dim): {k}")

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


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_int4_fp8_grouped_gemm_single_expert(batch_size):
    """
    测试单个专家的W4A8量化矩阵乘法
    验证CUTLASS内核在单专家情况下的正确性
    """
    print(f"\n[test_int4_fp8_grouped_gemm_single_expert] Test parameters:")
    print(f"  batch_size: {batch_size}")
    
    # Test parameters
    num_experts = 1
    m = batch_size  # batch size
    k = 512  # input dimension
    n = 1024  # output dimension
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    print(f"[test_int4_fp8_grouped_gemm_single_expert] m: {m}, k: {k}, n: {n}, num_experts: {num_experts}")

    # Create input tensors with ones
    if debug:
        # 调试模式：使用固定值进行测试
        a = torch.ones(m, k, dtype=torch.bfloat16, device=device)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] Debug mode - Input tensor a: shape: {a.shape}, dtype: {a.dtype}")
        
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] Debug mode - ref_w: shape: {ref_w.shape}, dtype: {ref_w.dtype}")
        
        a_scale = torch.ones(1, dtype=torch.float, device=device)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] Debug mode - a_scale: shape: {a_scale.shape}, dtype: {a_scale.dtype}")
        
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] Debug mode - ref_w_scale: shape: {ref_w_scale.shape}, dtype: {ref_w_scale.dtype}")
    else:
        # 正常模式：使用随机值进行测试
        a = torch.randn(m, k, dtype=dtype, device=device)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] Input tensor a: shape: {a.shape}, dtype: {a.dtype}")
        
        # 随机生成量化权重
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        print(f"[test_int4_fp8_grouped_gemm_single_expert] ref_w: shape: {ref_w.shape}, dtype: {ref_w.dtype}")
        
        # 生成激活量化尺度
        affine_coeff = 0.005
        a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        print(f"[test_int4_fp8_grouped_gemm_single_expert] a_scale: shape: {a_scale.shape}, dtype: {a_scale.dtype}")
        
        # 生成权重量化尺度（按组大小分组）
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )
        print(f"[test_int4_fp8_grouped_gemm_single_expert] ref_w_scale: shape: {ref_w_scale.shape}, dtype: {ref_w_scale.dtype}")

    # 对权重和尺度进行交织打包
    print(f"\n[test_int4_fp8_grouped_gemm_single_expert] Packing weights and scales...")
    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)
    print(f"[test_int4_fp8_grouped_gemm_single_expert] Final w: shape: {w.shape}, dtype: {w.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_single_expert] Final w_scale: shape: {w_scale.shape}, dtype: {w_scale.dtype}")

    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    print(f"[test_int4_fp8_grouped_gemm_single_expert] expert_offsets: shape: {expert_offsets.shape}, dtype: {expert_offsets.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_single_expert] problem_sizes: shape: {problem_sizes.shape}, dtype: {problem_sizes.dtype}")

    # 创建步长张量：用于CUTLASS内核的内存布局优化
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides
    print(f"[test_int4_fp8_grouped_gemm_single_expert] a_strides: shape: {a_strides.shape}, dtype: {a_strides.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_single_expert] c_strides: shape: {c_strides.shape}, dtype: {c_strides.dtype}")

    # Quantize input
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn).to(device)
    print(f"[test_int4_fp8_grouped_gemm_single_expert] Quantized input a_q: shape: {a_q.shape}, dtype: {a_q.dtype}")

    # Create output tensor
    c = torch.empty((m, n), dtype=torch.float16, device=device)
    print(f"[test_int4_fp8_grouped_gemm_single_expert] Output tensor c: shape: {c.shape}, dtype: {c.dtype}")
    # 运行CUTLASS内核进行量化矩阵乘法
    print(f"\n[test_int4_fp8_grouped_gemm_single_expert] Running CUTLASS kernel...")
    cutlass_w4a8_moe_mm(
        c,
        a_q,
        w,
        a_scale,
        w_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        128,  # 组大小
        8,    # 量化位数
    )
    c = c.to(dtype)
    print(f"[test_int4_fp8_grouped_gemm_single_expert] CUTLASS output c: shape: {c.shape}, dtype: {c.dtype}")

    # Reference implementation
    print(f"\n[test_int4_fp8_grouped_gemm_single_expert] Running reference implementation...")
    experts_selection_result = torch.full((m,), 0)  # 所有token都分配给专家0
    print(f"[test_int4_fp8_grouped_gemm_single_expert] experts_selection_result: shape: {experts_selection_result.shape}, dtype: {experts_selection_result.dtype}")
    
    c_ref = ref_grouped_gemm(
        c, a, a_scale, ref_w, ref_w_scale, num_experts, experts_selection_result
    )
    print(f"[test_int4_fp8_grouped_gemm_single_expert] Reference output c_ref: shape: {c_ref.shape}, dtype: {c_ref.dtype}")

    # Compare results
    print(f"\n[test_int4_fp8_grouped_gemm_single_expert] Comparing results...")
    try:
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)
        print(f"[test_int4_fp8_grouped_gemm_single_expert] SUCCESS: Results match!")
    except AssertionError as e:
        # torch.set_printoptions(threshold=10_000)
        print(f"  FAILURE: tensors are NOT close.")
        print(f"    Ref tensor: {c_ref.flatten()}")
        print(f"    Cutlass tensor: {c.flatten()}")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("num_experts", [2, 4, 6, 8])
def test_int4_fp8_grouped_gemm_multi_experts(batch_size, k, n, num_experts):
    """
    测试多个专家的W4A8量化矩阵乘法
    验证CUTLASS内核在多专家情况下的正确性
    """
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Test parameters:")
    print(f"  batch_size: {batch_size}, k: {k}, n: {n}, num_experts: {num_experts}")
    
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    print(
        f"\nTesting with batch_size={batch_size}, k={k}, n={n}, num_experts={num_experts}"
    )

    if debug:
        # 调试模式：使用固定值进行测试
        a = torch.ones(batch_size, k, dtype=torch.bfloat16, device=device)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] Debug mode - Input tensor a: shape: {a.shape}, dtype: {a.dtype}")
        
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] Debug mode - ref_w: shape: {ref_w.shape}, dtype: {ref_w.dtype}")
        
        a_scale = torch.ones(1, dtype=torch.float, device=device)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] Debug mode - a_scale: shape: {a_scale.shape}, dtype: {a_scale.dtype}")
        
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] Debug mode - ref_w_scale: shape: {ref_w_scale.shape}, dtype: {ref_w_scale.dtype}")
    else:
        # 正常模式：使用随机值进行测试
        a = torch.randn(batch_size, k, dtype=dtype, device=device)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] Input tensor a: shape: {a.shape}, dtype: {a.dtype}")
        
        # 随机生成量化权重
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] ref_w: shape: {ref_w.shape}, dtype: {ref_w.dtype}")
        
        # 生成激活量化尺度
        affine_coeff = 0.005
        a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] a_scale: shape: {a_scale.shape}, dtype: {a_scale.dtype}")
        
        # 生成权重量化尺度（按组大小分组）
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] ref_w_scale: shape: {ref_w_scale.shape}, dtype: {ref_w_scale.dtype}")

    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Packing weights and scales...")
    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Final w: shape: {w.shape}, dtype: {w.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Final w_scale: shape: {w_scale.shape}, dtype: {w_scale.dtype}")

    # random select experts
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Expert selection...")
    experts_selection_result = torch.randint(
        0, num_experts, (batch_size,), device=device
    )
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] experts_selection_result: shape: {experts_selection_result.shape}, dtype: {experts_selection_result.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] experts_selection_result values: {experts_selection_result}")
    
    # 根据专家ID对token进行排序，实现专家分组
    permutation = torch.argsort(experts_selection_result)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] permutation: shape: {permutation.shape}, dtype: {permutation.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] permutation values: {permutation}")
    
    # 统计每个专家分配的token数量
    expert_token_counts = torch.bincount(
        experts_selection_result, minlength=num_experts
    )
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] expert_token_counts: shape: {expert_token_counts.shape}, dtype: {expert_token_counts.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] expert_token_counts values: {expert_token_counts}")

    # Create problem sizes and offsets for active experts
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Creating problem sizes and offsets...")
    problem_sizes = []
    for i in range(num_experts):
        problem_sizes.append([n, expert_token_counts[i].item(), k])  # [输出维度, token数量, 输入维度]
    problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device=device)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] problem_sizes: shape: {problem_sizes.shape}, dtype: {problem_sizes.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] problem_sizes values: {problem_sizes}")

    # 计算每个专家在重排后输入中的起始位置
    expert_offsets = []
    offset = 0
    for i in range(num_experts):
        expert_offsets.append(offset)
        offset += problem_sizes[i][1].item()  # 累加每个专家的token数量
    expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=device)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] expert_offsets: shape: {expert_offsets.shape}, dtype: {expert_offsets.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] expert_offsets values: {expert_offsets}")

    # Permute input and quantize
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Input permutation and quantization...")
    a_perm = a[permutation]  # 根据专家ID重排输入
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Permuted input a_perm: shape: {a_perm.shape}, dtype: {a_perm.dtype}")
    
    # 将重排后的输入量化为FP8格式
    a_q_perm = (
        torch.clamp((a_perm / a_scale), -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Quantized permuted input a_q_perm: shape: {a_q_perm.shape}, dtype: {a_q_perm.dtype}")

    # Create stride tensors
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Creating stride tensors...")
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] a_strides: shape: {a_strides.shape}, dtype: {a_strides.dtype}")
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] c_strides: shape: {c_strides.shape}, dtype: {c_strides.dtype}")

    c_perm = torch.empty((batch_size, n), dtype=torch.float16, device=device)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Output tensor c_perm: shape: {c_perm.shape}, dtype: {c_perm.dtype}")
    # 运行CUTLASS内核进行分组矩阵乘法
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Running CUTLASS kernel...")
    cutlass_w4a8_moe_mm(
        c_perm,
        a_q_perm,
        w,
        a_scale,
        w_scale,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        128,  # 组大小
        8,    # 量化位数
    )
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] CUTLASS output c_perm: shape: {c_perm.shape}, dtype: {c_perm.dtype}")

    # Un-permute the result
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Un-permuting result...")
    c = torch.empty_like(c_perm)
    c[permutation] = c_perm  # 使用逆置换恢复原始顺序
    c = c.to(dtype)
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Final output c: shape: {c.shape}, dtype: {c.dtype}")

    # 运行参考实现进行验证
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Running reference implementation...")
    c_ref = ref_grouped_gemm(
        c, a, a_scale, ref_w, ref_w_scale, num_experts, experts_selection_result
    )
    print(f"[test_int4_fp8_grouped_gemm_multi_experts] Reference output c_ref: shape: {c_ref.shape}, dtype: {c_ref.dtype}")

    # Compare results
    print(f"\n[test_int4_fp8_grouped_gemm_multi_experts] Comparing results...")
    try:
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)
        print(f"[test_int4_fp8_grouped_gemm_multi_experts] SUCCESS: Results match!")
    except AssertionError as e:
        print(f"  FAILURE: tensors are NOT close.")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise


def ref_grouped_gemm(c, a, a_scale, w, w_scale, num_experts, experts_selection_result):
    """
    参考实现：传统的分组矩阵乘法
    用于验证CUTLASS实现的正确性
    """
    print(f"[ref_grouped_gemm] Input c: shape: {c.shape}, dtype: {c.dtype}")
    print(f"[ref_grouped_gemm] Input a: shape: {a.shape}, dtype: {a.dtype}")
    print(f"[ref_grouped_gemm] Input a_scale: shape: {a_scale.shape}, dtype: {a_scale.dtype}")
    print(f"[ref_grouped_gemm] Input w: shape: {w.shape}, dtype: {w.dtype}")
    print(f"[ref_grouped_gemm] Input w_scale: shape: {w_scale.shape}, dtype: {w_scale.dtype}")
    print(f"[ref_grouped_gemm] num_experts: {num_experts}")
    print(f"[ref_grouped_gemm] experts_selection_result: shape: {experts_selection_result.shape}, dtype: {experts_selection_result.dtype}")
    
    dtype = torch.bfloat16
    c_ref = torch.zeros_like(c)  # 初始化结果张量
    print(f"[ref_grouped_gemm] c_ref: shape: {c_ref.shape}, dtype: {c_ref.dtype}")
    
    # 量化输入：将激活值量化为FP8格式
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    print(f"[ref_grouped_gemm] Quantized input a_q: shape: {a_q.shape}, dtype: {a_q.dtype}")
    
    # 遍历每个专家
    for i in range(num_experts):
        print(f"[ref_grouped_gemm] Processing expert {i}...")
        # 找出分配给当前专家的token索引
        token_idx = torch.where(experts_selection_result == i)[0]
        print(f"[ref_grouped_gemm] Expert {i} - token_idx: shape: {token_idx.shape}, dtype: {token_idx.dtype}")
        print(f"[ref_grouped_gemm] Expert {i} - token_idx values: {token_idx}")
        
        # 如果没有token分配给这个专家，跳过
        if len(token_idx) == 0:
            print(f"[ref_grouped_gemm] Expert {i} - no tokens assigned, skipping")
            continue
        # 提取当前专家的输入
        a_expert = a_q[token_idx]
        print(f"[ref_grouped_gemm] Expert {i} - a_expert: shape: {a_expert.shape}, dtype: {a_expert.dtype}")

        # 重复尺度以匹配权重维度
        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(float)
        print(f"[ref_grouped_gemm] Expert {i} - ref_w_scale_repeat: shape: {ref_w_scale_repeat.shape}, dtype: {ref_w_scale_repeat.dtype}")
        
        # 反量化权重
        ref_w = (w[i].to(float) * ref_w_scale_repeat).to(dtype)
        print(f"[ref_grouped_gemm] Expert {i} - ref_w: shape: {ref_w.shape}, dtype: {ref_w.dtype}")
        
        # 执行矩阵乘法并应用激活尺度
        c_expert = torch.matmul(a_expert.to(dtype), ref_w.t().to(dtype)) * a_scale
        print(f"[ref_grouped_gemm] Expert {i} - c_expert before scale: shape: {c_expert.shape}, dtype: {c_expert.dtype}")
        
        c_expert = c_expert.to(dtype)
        print(f"[ref_grouped_gemm] Expert {i} - c_expert after scale: shape: {c_expert.shape}, dtype: {c_expert.dtype}")
        
        # 将当前专家的结果累加到最终输出中
        c_ref[token_idx] = c_expert.to(dtype)
        print(f"[ref_grouped_gemm] Expert {i} - updated c_ref: shape: {c_ref.shape}, dtype: {c_ref.dtype}")

    print(f"[ref_grouped_gemm] Final c_ref: shape: {c_ref.shape}, dtype: {c_ref.dtype}")
    return c_ref


if __name__ == "__main__":
    print("="*80)
    print("开始运行 CUTLASS W4A8 MoE MM 测试")
    print("="*80)
    
    # 单个测试参数
    batch_size = 2      # 批次大小
    k = 512            # 输入维度
    n = 1024           # 输出维度
    num_experts = 2    # 专家数量
    
    print(f"测试参数: batch_size={batch_size}, k={k}, n={n}, num_experts={num_experts}")
    print("="*80)
    
    try:
        print("\n--- 运行单专家测试 ---")
        test_int4_fp8_grouped_gemm_single_expert(batch_size)
        print("✓ 单专家测试通过")
        
        print("\n--- 运行多专家测试 ---")
        test_int4_fp8_grouped_gemm_multi_experts(batch_size, k, n, num_experts)
        print("✓ 多专家测试通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
