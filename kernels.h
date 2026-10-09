//kernels.h
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <type_traits>
#include <functional>
#include <cmath>
#include <ctime>
#include <thread>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <windows.h>
#include "uv.h"

__host__ __device__ __forceinline__ __half_raw h_zero() { return __half_raw{0x0000}; }
__host__ __device__ __forceinline__ __half_raw h_one() { return __half_raw{0x3C00}; }
__host__ __device__ __forceinline__ __half_raw h_minimum() { return __half_raw{0xFBFF}; }
__host__ __device__ __forceinline__ __half_raw h_maximum() { return __half_raw{0x7BFF}; }
__host__ __device__ __forceinline__ __half2_raw h2_zero() { return __half2_raw{0x0000, 0x0000}; }
__host__ __device__ __forceinline__ __half2_raw h2_one() { return __half2_raw{0x3C00, 0x3C00}; }
__host__ __device__ __forceinline__ __half2_raw h2_minimum() { return __half2_raw{0xFBFF, 0xFBFF}; }

__host__ __device__ __forceinline__ constexpr bool test_bit(action_type action, int position) { return (action >> position) & 1; }
__host__ __device__ __forceinline__ uint32_t generateUniqueSeed(int sample_index = 0)
{
	uint32_t final_seed = 0;
#ifdef __CUDA_ARCH__
	final_seed = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y * blockDim.z;
#else
	static thread_local uint32_t cpu_thread_id = []() { size_t hash = std::hash<std::thread::id>{}(std::this_thread::get_id()); return static_cast<uint32_t>(hash ^ (hash >> 16)); }();
	final_seed = cpu_thread_id;
#endif
	final_seed += static_cast<uint32_t>(sample_index) * 2654435761U;
	final_seed = (final_seed ^ 61) ^ (final_seed >> 16);
	final_seed *= 9;
	final_seed ^= final_seed >> 4;
	final_seed *= 668265261;
	final_seed ^= final_seed >> 15;
	return final_seed == 0 ? 123456789 : final_seed;
}
__host__ __device__ __forceinline__ uint32_t getFastRandomModifier(uint32_t seed)
{
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}
__host__ __device__ __forceinline__ uint32_t getFastRandomBounded(uint32_t seed, uint32_t n) { return (static_cast<uint64_t>(getFastRandomModifier(seed)) * static_cast<uint64_t>(n)) >> 32; }
__host__ __device__ __forceinline__ float getFastRandomFloatBoundedUniform(uint32_t seed, float n) { return getFastRandomModifier(seed) * (1.0f / 0xFFFFFFFFf) * n; }
__host__ __device__ __forceinline__ float getFastRandomFloatBoundedNormal(uint32_t seed, float n) { return -n + getFastRandomFloatBoundedUniform(seed, n) * 2.0f; }
template <typename type>
inline void generateNormalRandoms(universal_vector<type>& arr_it, int size, float offset, cudaStream_t stream, int seed = 1234)
{
	type* raw_ptr = arr_it.data();
	auto functor = [raw_ptr, size, offset, seed] __host__ __device__ (int idx) -> void { raw_ptr[idx] = __float2half(getFastRandomFloatBoundedNormal(generateUniqueSeed(idx * seed + size), offset)); };
	arr_it.for_each_n(size, functor, stream);
}

__device__ __forceinline__ int bfe_signed(uint32_t source, uint32_t bit_index, uint32_t num_bits)
{
	int result;
	asm("bfe.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(source), "r"(bit_index), "r"(num_bits));
	return result;
}
__device__ __forceinline__ float2 extract_ints4(uint32_t packed_u32, int pair_idx)
{
	const int shift = pair_idx << 3;
	return make_float2(static_cast<float>(bfe_signed(packed_u32, shift, 4)), static_cast<float>(bfe_signed(packed_u32, shift + 4, 4)));
}
__device__ __forceinline__ int32_t extract_int4(uint32_t packed_u32, int element_idx) { return bfe_signed(packed_u32, element_idx << 2, 4); }
__device__ __forceinline__ uint32_t load_u32(const uint32_t* __restrict__ ptr, int row_idx, int stride, int col_u32) { return ptr[row_idx * stride + col_u32]; }
__device__ __forceinline__ int8_t quantize_scale(float scale) { return static_cast<int8_t>(__float2int_rn(fmaxf(quant_scale_min, fminf(quant_scale_max, __log2f(scale))))); }
__device__ __forceinline__ float dequantize_scale(int8_t s_quant) { return fmaxf(exp2f(static_cast<float>(s_quant)) * (1.0f / quant_int4_max), epsilon); }
__device__ __forceinline__ int32_t quantize_scalar_to_int4(float scalar, float inv_out_scale) { return max(-8, min(static_cast<int>(quant_int4_max), __float2int_rn(scalar * inv_out_scale))) & 0xF; }
__device__ __forceinline__ int32_t quantize_scalar_to_int4_stochastic(float scalar, float inv_out_scale, int element_idx = 0)
{
	const float scaled = fmaxf(-8.0f, fminf(quant_int4_max, scalar * inv_out_scale));
	const int32_t floor_val_int = __float2int_rd(scaled);
	return (floor_val_int + (getFastRandomFloatBoundedUniform(generateUniqueSeed(element_idx), 1.0f) < scaled - __int2float_rd(floor_val_int))) & 0xF;
}
__device__ __forceinline__ float dequantize_scalar(uint32_t packed_u32, int element_idx, int8_t s_quant) { return extract_int4(packed_u32, element_idx) * dequantize_scale(s_quant); }
__device__ __forceinline__ uint32_t pack_8_elements_to_u32(float inv_out_scale, float* scalar)
{
	uint32_t packed_out = 0;
	#pragma unroll
	for (int b = 0; b < 8; ++b) packed_out |= quantize_scalar_to_int4(scalar[b], inv_out_scale) << (b << 2);
	return packed_out;
}
__device__ __forceinline__ int dot_int4(uint32_t a, uint32_t b) { return __dp4a(static_cast<int>(a & 0xF0F0F0F0), static_cast<int>(b & 0xF0F0F0F0), __dp4a(static_cast<int>((a << 4) & 0xF0F0F0F0), static_cast<int>((b << 4) & 0xF0F0F0F0), 0)) >> 8; }
__device__ __forceinline__ uint32_t bfi(uint32_t source, uint32_t insert, uint32_t pos, uint32_t len)
{
    uint32_t result;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(result) : "r"(insert), "r"(source), "r"(pos), "r"(len));
    return result;
}
__device__ __forceinline__ float silu_grad(float x, float dy)
{
	const float sig = __frcp_rn(1.0f + __expf(-x));
	return dy * sig * fmaf(x, 1.0f - sig, 1.0f);
}
__device__ __forceinline__ __half2 complex_mul_half2(__half2 a, __half2 b)
{
	const __half2 p1 = __hmul2(a, b), p2 = __hmul2(__lowhigh2highlow(a), b);
	return __halves2half2(__hsub(p1.x, p1.y), __hadd(p2.x, p2.y));
}
__device__ __forceinline__ void mma_m16n8k32_s4(int4& d, const uint2& a, const uint32_t b, const int4& c)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5}, "
                 "{%6}, "
                 "{%7, %8, %9, %10};"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y),
                   "r"(b),
                   "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
}
__device__ __forceinline__ float tensor_cores_m16n8k32(const uint32_t* __restrict__ input, const int8_t*  __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t*  __restrict__ weight_scales, int InC, int groups_per_in, int u32_stride_in, int spatial_blk_start, int outc_blk_start, int thread_m, int lane_id)
{
	int4 c_frag = make_int4(0, 0, 0, 0), d_frag = make_int4(0, 0, 0, 0);
	const int weight_outc_0 = outc_blk_start + ((lane_id & 3) << 1), matrix_a_row = spatial_blk_start + thread_m;
	#pragma unroll 4
	for (int k_idx = 0; k_idx < InC; k_idx += 32)
	{
		const int g = k_idx >> 5, weight_row_u32 = (g << 2) + (lane_id >> 3);
		uint2 a_reg;
		a_reg.x = input[matrix_a_row * u32_stride_in + (g << 2) + (lane_id & 3)];
		a_reg.y = input[matrix_a_row * u32_stride_in + (g << 2) + ((lane_id + 1) & 3)];
		mma_m16n8k32_s4(d_frag, a_reg, weights[weight_outc_0 * (groups_per_in << 2) + weight_row_u32], c_frag);
		const int accumulated_outc_0 = d_frag.x;
		d_frag = c_frag;
		mma_m16n8k32_s4(d_frag, a_reg, weights[(weight_outc_0 + 1) * (groups_per_in << 2) + weight_row_u32], c_frag);
		d_frag.x = accumulated_outc_0;
		c_frag = d_frag;
	}
	const float s_in = dequantize_scale(input_scales[matrix_a_row * groups_per_in]);
	const float s_w0 = dequantize_scale(weight_scales[weight_outc_0 * groups_per_in]);
	const float s_w1 = dequantize_scale(weight_scales[(weight_outc_0 + 1) * groups_per_in]);
	return (lane_id & 1) == 0 ? static_cast<float>(d_frag.x) * s_in * s_w0 : static_cast<float>(d_frag.y) * s_in * s_w1;
}
__device__ __forceinline__ float tensor_cores_m16n8k32_smem(const uint32_t* __restrict__ smem_input, const int8_t& input_scale, const uint32_t* __restrict__ weights, const int8_t*  __restrict__ weight_scales, int MidC, int outc_blk_start, int lane_id)
{
	int4 c_frag = make_int4(0, 0, 0, 0), d_frag = make_int4(0, 0, 0, 0);
	const int weight_outc_0 = outc_blk_start + ((lane_id & 3) << 1), groups_per_mid = MidC >> 5;
	#pragma unroll 4
	for (int k_idx = 0; k_idx < MidC; k_idx += 32)
	{
		const int g = k_idx >> 5, weight_row_u32 = (g << 2) + (lane_id >> 3);
		uint2 a_reg;
		a_reg.x = smem_input[(g << 2) + (lane_id & 3)];
		a_reg.y = smem_input[(g << 2) + ((lane_id + 1) & 3)];
		mma_m16n8k32_s4(d_frag, a_reg, weights[weight_outc_0 * (groups_per_mid << 2) + weight_row_u32], c_frag);
		const int accumulated_outc_0 = d_frag.x;
		d_frag = c_frag;
		mma_m16n8k32_s4(d_frag, a_reg, weights[(weight_outc_0 + 1) * (groups_per_mid << 2) + weight_row_u32], c_frag);
		d_frag.x = accumulated_outc_0;
		c_frag = d_frag;
	}
	const float s_in = dequantize_scale(input_scale);
	const float s_w0 = dequantize_scale(weight_scales[weight_outc_0 * groups_per_mid]);
	const float s_w1 = dequantize_scale(weight_scales[(weight_outc_0 + 1) * groups_per_mid]);
	return (lane_id & 1) == 0 ? static_cast<float>(d_frag.x) * s_in * s_w0 : static_cast<float>(d_frag.y) * s_in * s_w1;
}
template <typename T = __half>
__global__ void __launch_bounds__(gpu_block_threads, 4) quantizeKernel(const T* __restrict__ input, uint32_t* __restrict__ output, int8_t* __restrict__ scales, int total_elements)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int base_idx = idx << 5;
	if (base_idx >= total_elements) return;
	T local_data[32];
	const uint4* __restrict__ input_v4 = reinterpret_cast<const uint4*>(input + base_idx);
	#pragma unroll
	for (int i = 0; i < (sizeof(T) << 5) / sizeof(uint4); ++i) reinterpret_cast<uint4*>(local_data)[i] = input_v4[i];
	float abs_max = 0.0f;
	#pragma unroll
	for (int i = 0; i < 32; ++i) abs_max = fmaxf(abs_max, fabsf(static_cast<float>(local_data[i])));
	const float scale = abs_max < epsilon ? 1.0f : abs_max;
	scales[idx] = quantize_scale(scale);
	const float inv_scale = quant_int4_max * __frcp_rn(scale);
	uint32_t out_words[4] = {};
	const uint32_t thread_seed = generateUniqueSeed(idx);
	#pragma unroll
	for (int i = 0; i < 32; ++i) out_words[i >> 3] |= (quantize_scalar_to_int4_stochastic(static_cast<float>(local_data[i]), inv_scale, thread_seed ^ i) << ((i & 7) << 2));
	reinterpret_cast<uint4*>(output)[idx] = *reinterpret_cast<uint4*>(out_words);
}
template <typename T = __half>
__global__ void __launch_bounds__(gpu_block_threads, 4) updateParamsKernel(T* __restrict__ data, const T* __restrict__ grad, int size, float grad_norm_factor, float learning_rate, float penalty)
{
	constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(T);
	const float c_grad = -learning_rate * grad_norm_factor;
	const float c_data = fmaf(-learning_rate, penalty, 1.0f);
	const uint4* __restrict__ grad_v4 = reinterpret_cast<const uint4*>(grad);
	uint4* __restrict__ data_v4 = reinterpret_cast<uint4*>(data);
	const int size_v4 = size / elements_per_uint4;
	const int stride = blockDim.x * gridDim.x;
	#pragma unroll 4
	for (int v_idx = blockIdx.x * blockDim.x + threadIdx.x; v_idx < size_v4; v_idx += stride)
	{
		T local_d[elements_per_uint4], local_g[elements_per_uint4];
		*reinterpret_cast<uint4*>(local_d) = data_v4[v_idx];
		*reinterpret_cast<uint4*>(local_g) = grad_v4[v_idx];
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; ++i) local_d[i] = static_cast<T>(fmaf(static_cast<float>(local_g[i]), c_grad, static_cast<float>(local_d[i]) * c_data));
		data_v4[v_idx] = *reinterpret_cast<uint4*>(local_d);
	}
}
template <int channels_per_block = gpu_block_threads>
__global__ void convPointwiseFwdKernel(const uint32_t* __restrict__ input, const uint32_t* __restrict__ weights, const int8_t* __restrict__ input_scales, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	const int spatial_idx_out = blockIdx.z * gridDim.y + blockIdx.y;
	const int lane_id = threadIdx.x & 31;
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	const int spatial_idx_in = (spatial_idx_out >> (__ffs(Dimension_out) - 1)) * stride * Dimension_in + (spatial_idx_out & (Dimension_out - 1)) * stride;
	float acc = 0.0f;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
#if __CUDA_ARCH__ >= 800
	acc = tensor_cores_m16n8k32(input, input_scales, weights, weight_scales, InC, groups_per_in, u32_stride_in, (spatial_idx_in >> 4) << 4, (outc >> 3) << 3, lane_id >> 1, lane_id);
#else
	const int spatial_groups = spatial_idx_in * groups_per_in, outc_groups = outc * groups_per_in;
	const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
	const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
	uint32_t dot_shared[4];
	uint4 next_w = w_v4[0], next_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
	int8_t next_in_scale = input_scales[spatial_groups], next_w_scale = weight_scales[outc_groups];
	#pragma unroll 2
	for (int g = 0; g < groups_per_in; ++g)
	{
		const uint4 curr_w = next_w, curr_in_v4 = next_in_v4;
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		const int8_t curr_in_scale = next_in_scale, curr_w_scale = next_w_scale;
		int partial_sum = 0;
		if (g + 1 < groups_per_in)
		{
			next_w = w_v4[g + 1];
			next_in_v4 = lane_id == 0 ? input_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
			next_in_scale = input_scales[spatial_groups + g + 1];
			next_w_scale = weight_scales[outc_groups + g + 1];
		}
		#pragma unroll
		for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
		acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, acc);
	}
#endif
	acc += __half2float(biases[outc]);
	float local_sq = acc * acc;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	constexpr int warps = channels_per_block >> 5;
	__shared__ float smem_warps[warps], smem_block_total;
	const int warp_id = threadIdx.x >> 5;
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_block_total = block_sq;
	}
	__syncthreads();
	const float normed_acc = acc * __frsqrt_rn((smem_block_total / channels_per_block) + epsilon);
	const float activated_acc = normed_acc * __frcp_rn(1.0f + __expf(-normed_acc));
	float group_max = fabsf(activated_acc);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max = fmaxf(group_max, __shfl_xor_sync(0xFFFFFFFF, group_max, offset));
	const float out_scale = group_max < epsilon ? 1.0f : group_max;
	uint32_t val = quantize_scalar_to_int4(activated_acc, quant_int4_max * __frcp_rn(out_scale)) << ((lane_id & 7) << 2);
	#pragma unroll
	for (int i = 4; i > 0; i >>= 1) val |= __shfl_down_sync(0xFFFFFFFF, val, i);
	if ((lane_id & 7) == 0) output[spatial_idx_out * (OutC >> 3) + (outc >> 3)] = val;
	if (lane_id == 0) output_scales[spatial_idx_out * (OutC >> 5) + (outc >> 5)] = quantize_scale(out_scale);
}
template <int channels_per_block = gpu_block_threads>
__global__ void convFusedPointwiseFwdKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights1, const int8_t* __restrict__ weight_scales1, const __half* __restrict__ biases1, const uint32_t* __restrict__ weights2, const int8_t* __restrict__ weight_scales2, const __half* __restrict__ biases2, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int MidC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	const int spatial_idx_out = blockIdx.z * gridDim.y + blockIdx.y;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	constexpr int warps = channels_per_block >> 5;
	extern __shared__ uint32_t smem_middle[];
	__shared__ float smem_warps[warps], smem_block_total;
	__shared__ int8_t s_mid_scale;
	const int outc_1 = blockIdx.x * channels_per_block + threadIdx.x;
	float acc1 = 0.0f;
	if (outc_1 < MidC)
	{
		const int groups_per_in1 = InC >> 5, u32_stride_in1 = InC >> 3;
		#if __CUDA_ARCH__ >= 800
		acc1 = tensor_cores_m16n8k32(input, input_scales, weights1, weight_scales1, InC, groups_per_in1, u32_stride_in1, (spatial_idx_out >> 4) << 4, (outc_1 >> 3) << 3, lane_id >> 1, lane_id);
		#else
		const int spatial_idx_in = (spatial_idx_out >> (__ffs(Dimension_out) - 1)) * stride * Dimension_in + (spatial_idx_out & (Dimension_out - 1)) * stride;
		const int spatial_groups1 = spatial_idx_in * groups_per_in1, outc_groups1 = outc_1 * groups_per_in1;
		const uint4* __restrict__ w_v4_1 = reinterpret_cast<const uint4*>(weights1) + outc_groups1;
		const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in1);
		uint32_t dot_shared1[4];
		uint4 next_w1 = w_v4_1[0], next_in_v4_1 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t next_in_scale1 = input_scales[spatial_groups1], next_w_scale1 = weight_scales1[outc_groups1];
		#pragma unroll 2
		for (int g = 0; g < groups_per_in1; ++g)
		{
			const uint4 curr_w1 = next_w1, curr_in_v4_1 = next_in_v4_1;
			const auto& w_val_ptr1 = reinterpret_cast<const uint32_t(&)[4]>(curr_w1);
			const auto& shared_in1 = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4_1);
			const int8_t curr_in_scale1 = next_in_scale1, curr_w_scale1 = next_w_scale1;
			int partial_sum1 = 0;
			if (g + 1 < groups_per_in1)
			{
				next_w1 = w_v4_1[g + 1];
				next_in_v4_1 = lane_id == 0 ? input_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
				next_in_scale1 = input_scales[spatial_groups1 + g + 1];
				next_w_scale1 = weight_scales1[outc_groups1 + g + 1];
			}
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared1[i] = __shfl_sync(0xFFFFFFFF, shared_in1[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum1 += dot_int4(dot_shared1[i], w_val_ptr1[i]);
			acc1 = fmaf(dequantize_scale(curr_in_scale1), dequantize_scale(curr_w_scale1) * partial_sum1, acc1);
		}
		#endif
		acc1 += __half2float(biases1[outc_1]);
	}
	float local_sq1 = acc1 * acc1;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq1 += __shfl_xor_sync(0xFFFFFFFF, local_sq1, offset);
	if (lane_id == 0) smem_warps[warp_id] = local_sq1;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq1 = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq1 += __shfl_xor_sync(0xFFFFFFFF, block_sq1, offset);
		if (lane_id == 0) smem_block_total = block_sq1;
	}
	__syncthreads();
	const float normed_acc1 = acc1 * __frsqrt_rn((smem_block_total / channels_per_block) + epsilon);
	const float activated_acc1 = normed_acc1 * __frcp_rn(1.0f + __expf(-normed_acc1));
	float group_max1 = fabsf(activated_acc1);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max1 = fmaxf(group_max1, __shfl_xor_sync(0xFFFFFFFF, group_max1, offset));
	const float out_scale1 = group_max1 < epsilon ? 1.0f : group_max1;
	uint32_t val1 = quantize_scalar_to_int4(activated_acc1, quant_int4_max * __frcp_rn(out_scale1)) << ((lane_id & 7) << 2);
	#pragma unroll
	for (int i = 4; i > 0; i >>= 1) val1 |= __shfl_down_sync(0xFFFFFFFF, val1, i);
	if ((lane_id & 7) == 0 && outc_1 < MidC) smem_middle[outc_1 >> 3] = val1;
	if (threadIdx.x == 0) s_mid_scale = quantize_scale(out_scale1);
	__syncthreads();
	const int outc_2 = blockIdx.x * channels_per_block + threadIdx.x;
	float acc2 = 0.0f;
	if (outc_2 < OutC)
	{
		#if __CUDA_ARCH__ >= 800
		acc2 = tensor_cores_m16n8k32_smem(smem_middle, s_mid_scale, weights2, weight_scales2, MidC, (outc_2 >> 3) << 3, lane_id);
		#else
		const int groups_per_in2 = MidC >> 5;
		const int outc_groups2 = outc_2 * groups_per_in2;
		const uint4* __restrict__ w_v4_2 = reinterpret_cast<const uint4*>(weights2) + outc_groups2;
		const uint4* __restrict__ smem_v4_base = reinterpret_cast<const uint4*>(smem_middle);
		uint32_t dot_shared2[4];
		uint4 next_w2 = w_v4_2[0];
		uint4 next_in_v4_2 = lane_id == 0 ? smem_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t next_w_scale2 = weight_scales2[outc_groups2];
		#pragma unroll 2
		for (int g = 0; g < groups_per_in2; ++g)
		{
			const uint4 curr_w2 = next_w2, curr_in_v4_2 = next_in_v4_2;
			const auto& w_val_ptr2 = reinterpret_cast<const uint32_t(&)[4]>(curr_w2);
			const auto& shared_in2 = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4_2);
			const int8_t curr_w_scale2 = next_w_scale2;
			int partial_sum2 = 0;
			if (g + 1 < groups_per_in2)
			{
				next_w2 = w_v4_2[g + 1];
				next_in_v4_2 = lane_id == 0 ? smem_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
				next_w_scale2 = weight_scales2[outc_groups2 + g + 1];
			}
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared2[i] = __shfl_sync(0xFFFFFFFF, shared_in2[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum2 += dot_int4(dot_shared2[i], w_val_ptr2[i]);
			acc2 = fmaf(dequantize_scale(s_mid_scale), dequantize_scale(curr_w_scale2) * partial_sum2, acc2);
		}
		#endif
		acc2 += __half2float(biases2[outc_2]);
	}
	float local_sq2 = acc2 * acc2;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq2 += __shfl_xor_sync(0xFFFFFFFF, local_sq2, offset);
	if (lane_id == 0) smem_warps[warp_id] = local_sq2;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq2 = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq2 += __shfl_xor_sync(0xFFFFFFFF, block_sq2, offset);
		if (lane_id == 0) smem_block_total = block_sq2;
	}
	__syncthreads();
	const float normed_acc2 = acc2 * __frsqrt_rn((smem_block_total / channels_per_block) + epsilon);
	const float activated_acc2 = normed_acc2 * __frcp_rn(1.0f + __expf(-normed_acc2));
	float group_max2 = fabsf(activated_acc2);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max2 = fmaxf(group_max2, __shfl_xor_sync(0xFFFFFFFF, group_max2, offset));
	const float out_scale2 = group_max2 < epsilon ? 1.0f : group_max2;
	uint32_t val2 = quantize_scalar_to_int4(activated_acc2, quant_int4_max * __frcp_rn(out_scale2)) << ((lane_id & 7) << 2);
	#pragma unroll
	for (int i = 4; i > 0; i >>= 1) val2 |= __shfl_down_sync(0xFFFFFFFF, val2, i);
	if ((lane_id & 7) == 0 && outc_2 < OutC) output[spatial_idx_out * (OutC >> 3) + (outc_2 >> 3)] = val2;
	if (lane_id == 0 && outc_2 < OutC) output_scales[spatial_idx_out * (OutC >> 5) + (outc_2 >> 5)] = quantize_scale(out_scale2);
}
template <typename T = __half, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradInputKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, const T* __restrict__ grad_output, const T* __restrict__ master_weights, T* __restrict__ grad_input, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	const int spatial_idx_out = blockIdx.z * gridDim.y + blockIdx.y;
	const int log2_dim_out = __ffs(Dimension_out) - 1;
	const int spatial_idx_in = (spatial_idx_out >> log2_dim_out) * stride * Dimension_in + (spatial_idx_out & (Dimension_out - 1)) * stride;
	const int lane_id = threadIdx.x & 31;
	__shared__ float smem_dy_normed[channels_per_block];
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int spatial_groups = spatial_idx_in * groups_per_in, outc_groups = outc * groups_per_in;
	const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
	const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
	float fwd_acc = 0.0f;
	uint32_t dot_shared[4];
	uint4 next_w = w_v4[0], next_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
	int8_t next_in_scale = input_scales[spatial_groups];
	int8_t next_w_scale = weight_scales[outc_groups];
	#pragma unroll 2
	for (int g = 0; g < groups_per_in; ++g)
	{
		const uint4 curr_w = next_w;
		const uint4 curr_in_v4 = next_in_v4;
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		const int8_t curr_in_scale = next_in_scale;
		const int8_t curr_w_scale = next_w_scale;
		int partial_sum = 0;
		if (g + 1 < groups_per_in)
		{
			next_w = w_v4[g + 1];
			next_in_v4 = lane_id == 0 ? input_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
			next_in_scale = input_scales[spatial_groups + g + 1];
			next_w_scale = weight_scales[outc_groups + g + 1];
		}
		#pragma unroll
		for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
		fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
	}
	fwd_acc += __half2float(biases[outc]);
	float local_sq = fwd_acc * fwd_acc;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	constexpr int warps = channels_per_block >> 5;
	__shared__ float smem_warps[warps], smem_block_total;
	const int warp_id = threadIdx.x >> 5;
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_block_total = block_sq;
	}
	__syncthreads();
	const float inv_rms = __frsqrt_rn((smem_block_total / channels_per_block) + epsilon);
	smem_dy_normed[threadIdx.x] = silu_grad(fwd_acc * inv_rms, static_cast<float>(grad_output[spatial_idx_out * OutC + outc])) * inv_rms;
	__syncthreads();
	for (int inc_base = 0; inc_base < InC; inc_base += channels_per_block)
	{
		const int curr_inc = inc_base + threadIdx.x;
		if (curr_inc < InC)
		{
			float acc_input = 0.0f;
			#pragma unroll 8
			for (int i = 0; i < channels_per_block; ++i) acc_input = fmaf(smem_dy_normed[i], static_cast<float>(master_weights[blockIdx.x * channels_per_block + i * InC + curr_inc]), acc_input);
            atomicAdd(&grad_input[(((spatial_idx_out >> log2_dim_out) * stride << (__ffs(Dimension_in) - 1)) + (spatial_idx_out & (Dimension_out - 1)) * stride) * InC + curr_inc], static_cast<T>(acc_input));
		}
	}
}
template <typename T = __half, int block_size = 32, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradWeightKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, const T* __restrict__ grad_output, T* __restrict__ grad_weights, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	const int outc = blockIdx.x * block_size + threadIdx.x;
	const int inc_load = blockIdx.y * block_size + threadIdx.x;
	const int lane_id = threadIdx.x & 31;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int log2_dim_out = __ffs(Dimension_out) - 1, mask_dim_out = Dimension_out - 1;
	const int NHW_out = Dimension_out * Dimension_out, outc_groups = outc * groups_per_in;
	constexpr int warps_per_gn_group = channels_per_block / block_size, gn_groups_per_block = block_size * block_size / channels_per_block;
	__shared__ float smem_in[block_size][block_size + 1], smem_gn_warps[block_size], smem_gn_groups[gn_groups_per_block > 0 ? gn_groups_per_block : 1];
	__shared__ T smem_grad_out[block_size][block_size + 1];
	float sum = 0.0f;
	for (int p_base = 0; p_base < NHW_out; p_base += block_size)
	{
		const int p_in = p_base + threadIdx.y;
		const int spatial_idx_in = (p_in >> log2_dim_out) * stride * Dimension_in + (p_in & mask_dim_out) * stride;
		smem_in[threadIdx.y][threadIdx.x] = dequantize_scalar(load_u32(input, spatial_idx_in, u32_stride_in, inc_load >> 3), inc_load & 7, input_scales[spatial_idx_in * groups_per_in + (inc_load >> 5)]);
		const int p_grad = p_base + threadIdx.y;
		const int spatial_idx_in_fwd = (p_grad >> log2_dim_out) * stride * Dimension_in + (p_grad & mask_dim_out) * stride;
		const int spatial_groups_fwd = spatial_idx_in_fwd * groups_per_in;
		float fwd_acc = 0.0f;
		uint32_t dot_shared[4];
		const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
		const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in_fwd * u32_stride_in);
		uint4 next_w = w_v4[0];
		uint4 next_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t next_in_scale = input_scales[spatial_groups_fwd];
		int8_t next_w_scale = weight_scales[outc_groups];
		#pragma unroll 2
		for (int g = 0; g < groups_per_in; ++g)
		{
			const uint4 curr_w = next_w;
			const uint4 curr_in_v4 = next_in_v4;
			const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
			const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
			const int8_t curr_in_scale = next_in_scale;
			const int8_t curr_w_scale = next_w_scale;
			int partial_sum = 0;
			if (g + 1 < groups_per_in)
			{
				next_w = w_v4[g + 1];
				next_in_v4 = lane_id == 0 ? input_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
				next_in_scale = input_scales[spatial_groups_fwd + g + 1];
				next_w_scale = weight_scales[outc_groups + g + 1];
			}
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
			fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
		}
		fwd_acc += __half2float(biases[outc]);
		float local_sq = fwd_acc * fwd_acc;
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
		if (lane_id == 0) smem_gn_warps[threadIdx.y] = local_sq;
		__syncthreads();
		if (threadIdx.y == 0 && lane_id < blockDim.y / warps_per_gn_group)
		{
			float group_sum = 0.0f;
			const int group_start_warp = lane_id * warps_per_gn_group;
			#pragma unroll
			for (int i = 0; i < warps_per_gn_group; ++i) group_sum += smem_gn_warps[group_start_warp + i];
			smem_gn_groups[lane_id] = group_sum;
		}
		__syncthreads();
		const float inv_rms = __frsqrt_rn((smem_gn_groups[threadIdx.y / warps_per_gn_group] / channels_per_block) + epsilon);
		smem_grad_out[threadIdx.y][threadIdx.x] = static_cast<T>(silu_grad(fwd_acc * inv_rms, static_cast<float>(grad_output[p_grad * OutC + outc])) * inv_rms);
		T go[4];
		float gi[4];
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < block_size; k += 4)
		{
			#pragma unroll
			for (int i = 0; i < 4; i++) gi[i] = smem_in[threadIdx.y][k + i];
			#pragma unroll
			for (int i = 0; i < 4; i++) go[i] = smem_grad_out[k + i][threadIdx.x];
			#pragma unroll
			for (int i = 0; i < 4; i++) sum = fmaf(gi[i], static_cast<float>(go[i]), sum);
		}
		__syncthreads();
	}
	grad_weights[outc * InC + blockIdx.y * block_size + threadIdx.y] = static_cast<T>(sum);
}
template <typename T = __half, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradBiasKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, const T* __restrict__ grad_output, __half* __restrict__ grad_biases, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	const int lane_id = threadIdx.x & 31;
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	const int groups_per_in = InC >> 5;
	const int u32_stride_in = InC >> 3;
	const int outc_groups = outc * groups_per_in;
	const int log2_dim_out = __ffs(Dimension_out) - 1;
	const int mask_dim_out = Dimension_out - 1;
	const int NHW_out = Dimension_out * Dimension_out;
	constexpr int warps = channels_per_block >> 5;
	__shared__ float smem_warps[warps], smem_block_total;
	const int warp_id = threadIdx.x >> 5;
	float bias_grad_accumulator = 0.0f;
	for (int p = 0; p < NHW_out; ++p)
	{
		const int spatial_idx_in = (p >> log2_dim_out) * stride * Dimension_in + (p & mask_dim_out) * stride;
		const int spatial_groups = spatial_idx_in * groups_per_in;
		const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
		const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
		float fwd_acc = 0.0f;
		uint32_t dot_shared[4];
		uint4 next_w = w_v4[0];
		uint4 next_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t next_in_scale = input_scales[spatial_groups];
		int8_t next_w_scale = weight_scales[outc_groups];
		#pragma unroll 2
		for (int g = 0; g < groups_per_in; ++g)
		{
			const uint4 curr_w = next_w;
			const uint4 curr_in_v4 = next_in_v4;
			const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
			const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
			const int8_t curr_in_scale = next_in_scale;
			const int8_t curr_w_scale = next_w_scale;
			int partial_sum = 0;
			if (g + 1 < groups_per_in)
			{
				next_w = w_v4[g + 1];
				next_in_v4 = lane_id == 0 ? input_v4_base[g + 1] : make_uint4(0, 0, 0, 0);
				next_in_scale = input_scales[spatial_groups + g + 1];
				next_w_scale = weight_scales[outc_groups + g + 1];
			}
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
			fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
		}
		fwd_acc += __half2float(biases[outc]);
		float local_sq = fwd_acc * fwd_acc;
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
		if (lane_id == 0) smem_warps[warp_id] = local_sq;
		__syncthreads();
		if (warp_id == 0)
		{
			float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
			#pragma unroll
			for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
			if (lane_id == 0) smem_block_total = block_sq;
		}
		__syncthreads();
		const float inv_rms = __frsqrt_rn((smem_block_total / channels_per_block) + epsilon);
		bias_grad_accumulator = fmaf(silu_grad(fwd_acc * inv_rms, static_cast<float>(grad_output[p * OutC + outc])), inv_rms, bias_grad_accumulator);
	}
	grad_biases[outc] = __float2half(bias_grad_accumulator);
}
template <int kernel_size = 7, int threads_per_block = gpu_block_threads>
__global__ void convDepthwiseFwdKernel(const uint32_t* __restrict__ input, const uint32_t* __restrict__ weights, const int8_t* __restrict__ input_scales, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int Dimension_in, int Dimension_out, int stride)
{
	const int spatial_idx_out = blockIdx.z * gridDim.y + blockIdx.y;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	constexpr int warps_per_block = threads_per_block >> 5,
				  padding = kernel_size >> 1,
				  k_sq = kernel_size * kernel_size,
				  k_sq_padded = ((k_sq << 2) + 31) & ~31,
				  k_bound = (k_sq >> 2) << 2;
	const int group_idx = blockIdx.x * warps_per_block + warp_id;
	const int outc = (group_idx << 5) + lane_id;
	__shared__ uint32_t smem_all_warps[warps_per_block][k_sq_padded];
	uint32_t* smem_in = smem_all_warps[warp_id];
	const int h_in_start = (spatial_idx_out >> (__ffs(Dimension_out) - 1)) * stride - padding;
	const int w_in_start = (spatial_idx_out & (Dimension_out - 1)) * stride - padding;
	const int u32_stride_in = InC >> 3, groups_per_in = InC >> 5;
	uint4* smem_in_v4 = reinterpret_cast<uint4*>(smem_in);
	#pragma unroll
	for (int i = lane_id; i < (k_sq_padded >> 2); i += 32) smem_in_v4[i] = make_uint4(0, 0, 0, 0);
	__syncwarp();
	#pragma unroll
	for (int i = lane_id; i < (k_sq << 2); i += 32)
	{
		const int p_idx = i >> 2;
		const int cur_h = h_in_start + p_idx / kernel_size;
		const int cur_w = w_in_start + p_idx % kernel_size;
		if (cur_h >= 0 && cur_h < Dimension_in && cur_w >= 0 && cur_w < Dimension_in) smem_in[i] = load_u32(input, cur_h * Dimension_in + cur_w, u32_stride_in, (group_idx << 2) + (i & 3));
	}
	float acc = __half2float(biases[outc]), acc_arr[4] = {};
	const int u32_off = lane_id >> 3, shift = lane_id & 7;
	int k = 0;
	__syncwarp();
	for (; k < k_bound; k += 4)
	{
		#pragma unroll
		for (int sub_k = 0; sub_k < 4; ++sub_k)
		{
			const int curr_k = k + sub_k;
			const int curr_absolute_idx = curr_k * InC + outc;
			const uint32_t curr_w_packed = weights[curr_absolute_idx >> 3];
			const int8_t curr_w_scale = weight_scales[curr_absolute_idx >> 5];
			const int8_t curr_in_scale = input_scales[((h_in_start + curr_k / kernel_size) * Dimension_in + w_in_start + curr_k % kernel_size) * groups_per_in + group_idx];
			const int32_t smem_in_val = extract_int4(smem_in[(curr_k << 2) + u32_off], shift);
			acc_arr[sub_k] = smem_in_val == 0 ? acc_arr[sub_k] : fmaf(smem_in_val * dequantize_scale(curr_in_scale), dequantize_scalar(curr_w_packed, curr_absolute_idx & 7, curr_w_scale), acc_arr[sub_k]);
		}
	}
	for (; k < k_sq; ++k)
	{
		const int curr_absolute_idx = k * InC + outc;
		const uint32_t curr_w_packed = weights[curr_absolute_idx >> 3];
		const int8_t curr_w_scale = weight_scales[curr_absolute_idx >> 5];
		const int8_t curr_in_scale = input_scales[((h_in_start + k / kernel_size) * Dimension_in + w_in_start + k % kernel_size) * groups_per_in + group_idx];
		const int32_t smem_in_val = extract_int4(smem_in[(k << 2) + u32_off], shift);
		acc_arr[0] = smem_in_val == 0 ? acc_arr[0] : fmaf(smem_in_val * dequantize_scale(curr_in_scale), dequantize_scalar(curr_w_packed, curr_absolute_idx & 7, curr_w_scale), acc_arr[0]);
	}
	acc = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];
	float group_max = fabsf(acc);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max = fmaxf(group_max, __shfl_xor_sync(0xFFFFFFFF, group_max, offset));
	const float out_scale = group_max < epsilon ? 1.0f : group_max;
	uint32_t val = static_cast<uint32_t>(quantize_scalar_to_int4(acc, quant_int4_max * __frcp_rn(out_scale))) << (shift << 2);
	#pragma unroll
	for (int i = 1; i < 8; i <<= 1) val |= __shfl_down_sync(0xFFFFFFFF, val, i);
	if ((lane_id & 7) == 0) output[spatial_idx_out * u32_stride_in + (group_idx << 2) + (lane_id >> 3)] = val;
	if (lane_id == 0) output_scales[spatial_idx_out * groups_per_in + group_idx] = quantize_scale(out_scale);
}
template <typename T = __half, int kernel_size = 7>
__global__ void convDepthwiseBwdGradInputKernel(const T* __restrict__ grad_output, const T* __restrict__ master_weights, T* __restrict__ grad_input, int InC, int Dimension_in, int Dimension_out, int stride)
{
	const int spatial_idx_in = blockIdx.z * gridDim.y + blockIdx.y;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int warps_per_block = blockDim.x >> 5;
	const int group_idx = blockIdx.x * warps_per_block + warp_id;
	const int inc = (group_idx << 5) + lane_id;
	const int h_in = spatial_idx_in >> (__ffs(Dimension_in) - 1);
	const int w_in = spatial_idx_in & (Dimension_in - 1);
	constexpr int padding = kernel_size >> 1, k_sq = kernel_size * kernel_size;
	float acc = 0.0f;
	int next_h_out_t = h_in / kernel_size + padding, next_w_out_t = w_in % kernel_size + padding;
	float next_go = 0.0f, next_mw = static_cast<float>(master_weights[inc]);
	if (next_h_out_t % stride == 0 && next_w_out_t % stride == 0)
	{
		const int ho = next_h_out_t / stride;
		const int wo = next_w_out_t / stride;
		if (ho >= 0 && ho < Dimension_out && wo >= 0 && wo < Dimension_out) next_go = static_cast<float>(grad_output[(ho * Dimension_out + wo) * InC + inc]);
	}
	#pragma unroll 4
	for (int k = 0; k < k_sq; ++k)
	{
		const float curr_go = next_go, curr_mw = next_mw;
		if (k + 1 < k_sq)
		{
			next_h_out_t = h_in - (k + 1) / kernel_size + padding;
			next_w_out_t = w_in - (k + 1) % kernel_size + padding;
			next_mw = static_cast<float>(master_weights[(k + 1) * InC + inc]);
			next_go = 0.0f;
			if (next_h_out_t % stride == 0 && next_w_out_t % stride == 0)
			{
				const int ho = next_h_out_t / stride;
				const int wo = next_w_out_t / stride;
				if (ho >= 0 && ho < Dimension_out && wo >= 0 && wo < Dimension_out) next_go = static_cast<float>(grad_output[(ho * Dimension_out + wo) * InC + inc]);
			}
		}
		acc = fmaf(curr_go, curr_mw, acc);
	}
	grad_input[spatial_idx_in * InC + inc] = static_cast<T>(acc);
}
template <typename T = __half, int kernel_size = 7, int threads_per_block = gpu_block_threads, int chunk_size = 8>
__global__ void convDepthwiseBwdGradWeightKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const T* __restrict__ grad_output, T* __restrict__ grad_weights, int InC, int Dimension_in, int Dimension_out, int stride)
{
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	constexpr int warps_per_block = threads_per_block >> 5,
				  padding = kernel_size >> 1,
				  k_sq = kernel_size * kernel_size;
	const int group_idx = blockIdx.x * warps_per_block + warp_id;
	const int inc = (group_idx << 5) + lane_id;
	const int log2_dim_out = __ffs(Dimension_out) - 1;
	const int mask_dim_out = Dimension_out - 1;
	const int u32_stride_in = InC >> 3;
	const int groups_per_in = InC >> 5;
	const int in_idx_u32 = inc >> 3;
	const int shift = inc & 7;
	__shared__ uint32_t smem_all_windows[warps_per_block][k_sq];
	__shared__ float smem_all_scales[warps_per_block][k_sq], smem_acc[threads_per_block][chunk_size];
	uint32_t* s_window_in = smem_all_windows[warp_id];
	float* s_window_scales = smem_all_scales[warp_id];
	for (int chunk_start = 0; chunk_start < k_sq; chunk_start += chunk_size)
	{
		#pragma unroll
		for (int c = 0; c < chunk_size; ++c) smem_acc[threadIdx.x][c] = 0.0f;
		for (int p = 0; p < Dimension_out * Dimension_out; ++p)
		{
			const float go = static_cast<float>(grad_output[p * InC + inc]);
			const int h_in_base = (p >> log2_dim_out) * stride - padding;
			const int w_in_base = (p & mask_dim_out) * stride - padding;
			#pragma unroll 2
			for (int k = lane_id; k < k_sq; k += 32)
			{
				const int h_in = h_in_base + k / kernel_size;
				const int w_in = w_in_base + k % kernel_size;
				if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
				{
					const int spatial_idx_in = h_in * Dimension_in + w_in;
					s_window_in[k] = load_u32(input, spatial_idx_in, u32_stride_in, in_idx_u32);
					s_window_scales[k] = dequantize_scale(input_scales[spatial_idx_in * groups_per_in + group_idx]);
				}
				else
				{
					s_window_in[k] = 0;
					s_window_scales[k] = 0.0f;
				}
			}
			__syncwarp();
			#pragma unroll
			for (int c = 0; c < chunk_size; ++c)
			{
				const int k = chunk_start + c;
				if (k < k_sq) smem_acc[threadIdx.x][c] = fmaf(extract_int4(s_window_in[k], shift) * s_window_scales[k], go, smem_acc[threadIdx.x][c]);
			}
			__syncwarp();
		}
		#pragma unroll
		for (int c = 0; c < chunk_size; ++c)
		{
			const int k = chunk_start + c;
			if (k < k_sq) grad_weights[k * InC + inc] = static_cast<T>(smem_acc[threadIdx.x][c]);
		}
	}
}
template <typename T = __half>
__global__ void convDepthwiseBwdGradBiasKernel(const T* __restrict__ grad_output, __half* __restrict__ grad_biases, int InC, int NHW)
{
	const int lane_id = threadIdx.x & 31;
	const int group_idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	const int outc = (group_idx << 5) + lane_id;
	float sum_arr[4] = {};
	float next_go[4];
	#pragma unroll
	for (int i = 0; i < 4; ++i) next_go[i] = i < NHW ? static_cast<float>(grad_output[i * InC + outc]) : 0.0f;
	for (int p = 0; p < NHW; p += 4)
	{
		float curr_go[4];
		#pragma unroll
		for (int i = 0; i < 4; ++i) curr_go[i] = next_go[i];
		if (p + 4 < NHW)
		{
			#pragma unroll
			for (int i = 0; i < 4; ++i) next_go[i] = p + 4 + i < NHW ? static_cast<float>(grad_output[(p + 4 + i) * InC + outc]) : 0.0f;
		}
		#pragma unroll
		for (int i = 0; i < 4; i++) sum_arr[i] += curr_go[i];
	}
	const float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
	const float neighbor_sum = __shfl_xor_sync(0xFFFFFFFF, sum, 1);
	if ((lane_id & 1) == 0) *reinterpret_cast<__half2*>(grad_biases + outc) = __floats2half2_rn(sum, neighbor_sum);
}
template <int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 4) residualAddFwdKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ residual, const int8_t* __restrict__ residual_scales, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int channels, int dimension, float survival_prob_inv)
{
	const int s_idx = blockIdx.y * dimension + blockIdx.x;
	constexpr int warps_per_block = threads_per_block >> 5;
	extern __shared__ float smem_pool[];
	float* smem_results = smem_pool;
	__shared__ float s_final_max, warp_red[warps_per_block];
	const int lane_id = threadIdx.x & 31;
	const int num_u32 = channels >> 3, num_groups = channels >> 5;
	float l_max = 0.0f;
	const uint4* __restrict__ input_v4_ptr = reinterpret_cast<const uint4*>(input + s_idx * num_u32);
	const uint4* __restrict__ residual_v4_ptr = reinterpret_cast<const uint4*>(residual + s_idx * num_u32);
	const float res_scale_expt_res = f_residual * survival_prob_inv;
	for (int v_idx = threadIdx.x; v_idx < num_groups; v_idx += threads_per_block)
	{
		const float s_res = dequantize_scale(residual_scales[s_idx * num_groups + v_idx]);
		const float s_in = dequantize_scale(input_scales[s_idx * num_groups + v_idx]);
		const uint4 val_v4 = input_v4_ptr[v_idx];
		const uint4 res_v4 = residual_v4_ptr[v_idx];
		const auto& val_v4_ptr = reinterpret_cast<const uint32_t(&)[4]>(val_v4);
		const auto& res_v4_ptr = reinterpret_cast<const uint32_t(&)[4]>(res_v4);
		const int base_c_idx = v_idx << 5;
		const float res_scale = s_res * res_scale_expt_res;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
			const int sub_c_idx = base_c_idx + (i << 3);
			#pragma unroll
			for (int pair_idx = 0; pair_idx < 4; ++pair_idx)
			{
				const float2 pair_in = extract_ints4(val_v4_ptr[i], pair_idx);
				const float2 pair_res = extract_ints4(res_v4_ptr[i], pair_idx);
				const float res0 = fmaf(pair_in.x, s_in, pair_res.x * res_scale);
				const float res1 = fmaf(pair_in.y, s_in, pair_res.y * res_scale);
				const int write_idx = sub_c_idx + (pair_idx << 1);
				smem_results[write_idx] = res0;
				smem_results[write_idx + 1] = res1;
				l_max = fmaxf(l_max, fmaxf(fabsf(res0), fabsf(res1)));
			}
		}
	}
	__syncthreads();
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFF, l_max, offset));
	if (lane_id == 0) warp_red[threadIdx.x >> 5] = l_max;
	__syncthreads();
	if (threadIdx.x < 32)
	{
		float val = threadIdx.x < warps_per_block ? warp_red[threadIdx.x] : 0.0f;
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
		if (threadIdx.x == 0) s_final_max = val;
	}
	__syncthreads();
	const float out_scale = s_final_max < epsilon ? 1.0f : s_final_max;
	const float inv_out_scale = quant_int4_max * __frcp_rn(out_scale);
	uint4* __restrict__ output_v4_ptr = reinterpret_cast<uint4*>(output + s_idx * num_u32);
	for (int v_idx = threadIdx.x; v_idx < num_groups; v_idx += threads_per_block)
	{
		uint4 out_v4;
		auto& out_v4_ptr = reinterpret_cast<uint32_t(&)[4]>(out_v4);
		#pragma unroll
		for (int i = 0; i < 4; ++i) out_v4_ptr[i] = pack_8_elements_to_u32(inv_out_scale, smem_results + (((v_idx << 2) + i) << 3));
		output_v4_ptr[v_idx] = out_v4;
	}
	for (int g_idx = threadIdx.x; g_idx < num_groups; g_idx += threads_per_block) output_scales[s_idx * num_groups + g_idx] = quantize_scale(out_scale);
}
template <typename T = __half, int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 4) residualAddBwdKernel(const T* __restrict__ grad_next, T* __restrict__ grad_res, int channels, int dimension, float survival_prob_inv)
{
	constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(T);
	const float combined_weight = survival_prob_inv * f_residual;
	const int offset = (blockIdx.y * dimension + blockIdx.x) * channels;
	const uint4* __restrict__ gn_ptr_v4 = reinterpret_cast<const uint4*>(grad_next + offset);
	uint4* __restrict__ gr_ptr_v4 = reinterpret_cast<uint4*>(grad_res + offset);
	const int size_v4 = channels / elements_per_uint4;
	#pragma unroll 2
	for (int v_idx = threadIdx.x; v_idx < size_v4; v_idx += threads_per_block)
	{
		T local_gn[elements_per_uint4], local_gr[elements_per_uint4];
		*reinterpret_cast<uint4*>(local_gn) = gn_ptr_v4[v_idx];
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; ++i) local_gr[i] = static_cast<T>(static_cast<float>(local_gn[i]) * combined_weight);
		gr_ptr_v4[v_idx] = *reinterpret_cast<uint4*>(local_gr);
	}
}
template <int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 2) mambaSSMFwdKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const __half* __restrict__ biases, float* __restrict__ h_states, float* __restrict__ trace_A_re, float* __restrict__ trace_A_im, float* __restrict__ trace_B, float* __restrict__ trace_Delta, __half* __restrict__ output, int OutC)
{
	constexpr int num_warps = threads_per_block >> 5,
				  off_W_C_re = mamba_d_inner + (mamba_d_inner << 5),
				  off_W_C_im = off_W_C_re + mamba_d_inner * mamba_d_state,
				  off_A_re = off_W_C_im + mamba_d_inner * mamba_d_state,
				  off_A_im = off_A_re + mamba_d_inner * mamba_d_state,
				  off_D = off_A_im + mamba_d_inner * mamba_d_state,
				  off_W_out = off_D + mamba_d_inner,
				  u32_stride_in = final_channels >> 3,
				  mamba_cols = mamba_d_state >> 5;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int block_ch_idx = (blockIdx.y * num_warps + warp_id) << 5;
	const int ch = block_ch_idx + lane_id;
	const int in_ch_idx = ch & (final_channels - 1);
	const int oc_per_block = OutC / gridDim.x;
	const int block_oc_start = blockIdx.x * oc_per_block;
	const int block_oc_end = block_oc_start + oc_per_block;
	__shared__ __half s_mem[mamba_d_inner];
	float hr_l[mamba_cols], hi_l[mamba_cols], trA_re[mamba_cols], trA_im[mamba_cols], trB[mamba_cols], trDelta[mamba_cols], cached_raw_A_re[mamba_cols], cached_raw_A_im[mamba_cols], cached_W_C_re[mamba_cols], cached_W_C_im[mamba_cols];
	float W_delta_l = 0.0f, D_skip_l = 0.0f;
	uint32_t B_row_packed[4];
	int8_t B_single_scale = 0;
	const int base_row_h = ch * (mamba_d_state << 1);
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		hr_l[i] = h_states[base_row_h + lane_id + (i << 6)];
		hi_l[i] = h_states[base_row_h + lane_id + (i << 6) + 32];
		trA_re[i] = 0.0f;
		trA_im[i] = 0.0f;
		trB[i] = 0.0f;
		trDelta[i] = 0.0f;
	}
	W_delta_l = dequantize_scalar(weights[ch >> 3], ch & 7, weight_scales[ch >> 5]);
	D_skip_l = dequantize_scalar(weights[(off_D + ch) >> 3], (off_D + ch) & 7, weight_scales[(off_D + ch) >> 5]);
	const int mimo_block_id = block_ch_idx / mamba_mimo_group_size;
	const int base_W_B = (mamba_d_inner >> 3) + mimo_block_id * (mamba_mimo_matrix_elements >> 3) + (lane_id << 2);
	#pragma unroll
	for (int b = 0; b < 4; ++b) B_row_packed[b] = weights[base_W_B + b];
	B_single_scale = weight_scales[(mamba_d_inner + mimo_block_id * mamba_mimo_matrix_elements + lane_id * mamba_mimo_group_size) >> 5];
	#pragma unroll
	for (int c = 0; c < mamba_cols; ++c)
	{
		const int w_off = ch * mamba_d_state + lane_id + (c << 5);
		cached_raw_A_re[c] = dequantize_scalar(weights[(off_A_re + w_off) >> 3], (off_A_re + w_off) & 7, weight_scales[(off_A_re + w_off) >> 5]);
		cached_raw_A_im[c] = dequantize_scalar(weights[(off_A_im + w_off) >> 3], (off_A_im + w_off) & 7, weight_scales[(off_A_im + w_off) >> 5]);
		cached_W_C_re[c] = dequantize_scalar(weights[(off_W_C_re + w_off) >> 3], (off_W_C_re + w_off) & 7, weight_scales[(off_W_C_re + w_off) >> 5]);
		cached_W_C_im[c] = dequantize_scalar(weights[(off_W_C_im + w_off) >> 3], (off_W_C_im + w_off) & 7, weight_scales[(off_W_C_im + w_off) >> 5]);
	}
	const float x_local = dequantize_scalar(load_u32(input, 0, u32_stride_in, in_ch_idx >> 3), in_ch_idx & 7, input_scales[in_ch_idx >> 5]);
	float B_x_mimo = 0.0f, y_acc = 0.0f;
	#pragma unroll
	for (int window = 0; window < 32; window += 4) 
	{
		float x_chunk[4];
		#pragma unroll
		for (int i = 0; i < 4; ++i) x_chunk[i] = __shfl_sync(0xFFFFFFFF, x_local, window + i);
		#pragma unroll
		for (int i = 0; i < 4; ++i) B_x_mimo = fmaf(dequantize_scalar(B_row_packed[(window + i) >> 3], (window + i) & 7, B_single_scale), x_chunk[i], B_x_mimo);
	}
	const float xlwdl = x_local * W_delta_l, dt = xlwdl + __logf(1.0f + __expf(-fabsf(xlwdl))), dt_half_neg = -dt * 0.5f;
	#pragma unroll
	for (int c = 0; c < mamba_cols; ++c)
	{
		const float are = -__expf(cached_raw_A_re[c]);
		const float dim = dt_half_neg * cached_raw_A_im[c];
		const float dre = fmaf(dt_half_neg, are, 1.0f);
		const float inv_s = __frcp_rn(fmaf(dre, dre, fmaf(dim, dim, epsilon)));
		const float ire = dre * inv_s;
		const float iim = -dim * inv_s;
		const float nim = -1.0f * dim;
		const float nre = fmaf(dt * 0.5f, are, 1.0f);
		const float abr = fmaf(nre, ire, -nim * iim);
		const float abi = fmaf(nre, iim, nim * ire);
		const float dt_B_x = dt * B_x_mimo;
		const float hr_old = hr_l[c];
		hr_l[c] = fmaf(abr, hr_old, fmaf(-abi, hi_l[c], dt_B_x * ire));
		hi_l[c] = fmaf(abr, hi_l[c], fmaf(abi, hr_old, dt_B_x * iim));
		trA_re[c] = fmaf(abr, trA_re[c], hr_old);
		trA_im[c] = fmaf(abr, trA_im[c], hi_l[c]);
		trB[c] = fmaf(abr, trB[c], dt * x_local);
		trDelta[c] = fmaf(abr, trDelta[c], B_x_mimo * ire + hr_old * are * -0.5f);
		y_acc = fmaf(hr_l[c], cached_W_C_re[c], fmaf(-hi_l[c], cached_W_C_im[c], y_acc));
	}
	s_mem[ch] = __float2half(fmaf(D_skip_l, x_local, y_acc));
	__syncthreads();
	for (int b_oc = block_oc_start; b_oc < block_oc_end; b_oc += num_warps)
	{
		const int oc = b_oc + warp_id;
		float sum_out = 0.0f;
		if (oc < block_oc_end)
		{
			const int w_base = off_W_out + oc * mamba_d_inner;
			float local_s_arr[4] = {};
			uint32_t w_raw[4];
			int8_t ws_raw[4];
			for (int ic = lane_id; ic < mamba_d_inner; ic += 128)
			{
				#pragma unroll
				for (int i = 0; i < 4; i++)
				{
					const int idx = w_base + ic + (i << 5);
					w_raw[i] = weights[idx >> 3];
					ws_raw[i] = weight_scales[idx >> 5];
				}
				#pragma unroll
				for (int i = 0; i < 4; i++)
				{
					const int idx_ic = ic + (i << 5);
					local_s_arr[i] = fmaf(__half2float(s_mem[idx_ic]), dequantize_scalar(w_raw[i], (w_base + idx_ic) & 7, ws_raw[i]), local_s_arr[i]);
				}
			}
			sum_out = local_s_arr[0] + local_s_arr[1] + local_s_arr[2] + local_s_arr[3];
		}
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) sum_out += __shfl_down_sync(0xFFFFFFFF, sum_out, offset);
		if (lane_id == 0 && oc < block_oc_end)
		{
			const float b = __half2float(biases[oc]);
			output[oc] = __float2half((sum_out + b) * __frcp_rn(1.0f + __expf(-(sum_out + b))));
		}
	}
	__syncthreads();
	if (blockIdx.x != 0) return;
	const int base_h = ch * (mamba_d_state << 1), base_tr = ch * mamba_d_state;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_tr + lane_id + (i << 5);
		h_states[base_h + lane_id + (i << 6)] = hr_l[i];
		h_states[base_h + lane_id + (i << 6) + 32] = hi_l[i];
		trace_A_re[idx] = trA_re[i];
		trace_A_im[idx] = trA_im[i];
		trace_B[idx] = trB[i];
		trace_Delta[idx] = trDelta[i];
	}
}
template <typename T = __half, typename U = __half, int threads_per_block = gpu_block_threads>
__global__ void mambaSSMApplyTracesKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const U* __restrict__ grad_output, const __half* __restrict__ output, const float* __restrict__ trace_A_re, const float* __restrict__ trace_A_im, const float* __restrict__ trace_B, const float* __restrict__ trace_Delta, const T* __restrict__ master_weights, T* __restrict__ grad_weights, __half* __restrict__ grad_biases, T* __restrict__ grad_input, int OutC)
{
	constexpr int off_W_C_re = mamba_d_inner + (mamba_d_inner << 5),
				  off_W_C_im = off_W_C_re + mamba_d_inner * mamba_d_state,
				  off_A_re = off_W_C_im + mamba_d_inner * mamba_d_state,
				  off_A_im = off_A_re + mamba_d_inner * mamba_d_state,
				  off_D = off_A_im + mamba_d_inner * mamba_d_state,
				  off_W_out = off_D + mamba_d_inner,
				  mamba_cols = mamba_d_state >> 5,
				  u32_stride_in = final_channels >> 3;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int num_warps = threads_per_block >> 5;
	const int block_ch_idx = (blockIdx.x * num_warps + warp_id) << 5;
	const int ch_idx = block_ch_idx + lane_id;
	const int in_ch_idx = ch_idx & (final_channels - 1);
	float dW_delta_acc = 0.0f, dD_acc = 0.0f, dAre_acc[mamba_cols], dAim_acc[mamba_cols], dBre_acc[mamba_cols];
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		dAre_acc[i] = 0.0f;
		dAim_acc[i] = 0.0f;
		dBre_acc[i] = 0.0f;
	}
	const float D_skip = static_cast<float>(master_weights[off_D + ch_idx]), W_delta = static_cast<float>(master_weights[ch_idx]);
	float grad_y_ssm = 0.0f;
	for (int oc = 0; oc < OutC; ++oc)
	{
		const float o = __half2float(output[oc]);
		const float g_act = static_cast<float>(grad_output[oc]) * o * (1.0f - o);
		grad_y_ssm = fmaf(g_act, static_cast<float>(master_weights[off_W_out + oc * mamba_d_inner + ch_idx]), grad_y_ssm);
		if (blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(&grad_biases[oc], __float2half(g_act));
	}
	const int base_tr = ch_idx * mamba_d_state;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_tr + lane_id + (i << 5);
		dAre_acc[i] = fmaf(grad_y_ssm, trace_A_re[idx], dAre_acc[i]);
		dAim_acc[i] = fmaf(grad_y_ssm, trace_A_im[idx], dAim_acc[i]);
		dBre_acc[i] = fmaf(grad_y_ssm, trace_B[idx], dBre_acc[i]);
	}
	const float current_trace_delta = trace_Delta[base_tr + lane_id];
	const float x_t = dequantize_scalar(load_u32(input, 0, u32_stride_in, in_ch_idx >> 3), in_ch_idx & 7, input_scales[in_ch_idx >> 5]);
	dW_delta_acc = fmaf(grad_y_ssm, current_trace_delta, dW_delta_acc);
	dD_acc = fmaf(grad_y_ssm, x_t, dD_acc);
	atomicAdd(grad_input + (ch_idx & (final_channels - 1)), static_cast<T>(fmaf(grad_y_ssm * current_trace_delta * __frcp_rn(1.0f + __expf(-x_t * W_delta)), W_delta, grad_y_ssm * D_skip)));
	atomicAdd(&grad_weights[ch_idx], static_cast<T>(dW_delta_acc));
	atomicAdd(&grad_weights[off_D + ch_idx], static_cast<T>(dD_acc));
	const int base_row_offset = ch_idx * mamba_d_state;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int w_idx = base_row_offset + lane_id + (i << 5);
		atomicAdd(&grad_weights[off_A_re + w_idx], static_cast<T>(dAre_acc[i]));
		atomicAdd(&grad_weights[off_A_im + w_idx], static_cast<T>(dAim_acc[i]));
		atomicAdd(&grad_weights[mamba_d_inner + block_ch_idx / mamba_mimo_group_size * mamba_mimo_matrix_elements + lane_id * mamba_mimo_group_size + i], static_cast<T>(dBre_acc[i]));
	}
}
template <int dummy = 0>
__global__ void resizeKernel(const __half* __restrict__ src, __half* __restrict__ dst, int srcH, int srcW, int dstDim, int channels)
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (blockIdx.z >= channels || y >= dstDim || x >= dstDim) return;
	const float scale = static_cast<float>(srcW) / dstDim;
	const float y_scaled = fmaf(y + 0.5f, scale, -0.5f);
	const float fy = floorf(y_scaled);
	const int y1 = max(0, static_cast<int>(fy));
	const float wy = y_scaled - fy;
	const __half* s_ptr = src + blockIdx.z * srcW * srcH;
	const int row1 = y1 * srcW;
	const int row2 = min(y1 + 1, srcH - 1) * srcW;
	const float x_scaled_0 = fmaf(x + 0.5f, scale, -0.5f);
	const float x_scaled_1 = fmaf(x + 1.5f, scale, -0.5f);
	const float fx0 = floorf(x_scaled_0);
	const float fx1 = floorf(x_scaled_1);
	const float wx0 = x_scaled_0 - fx0;
	const float wx1 = x_scaled_1 - fx1;
	const int x1_0 = max(0, static_cast<int>(fx0));
	const int x1_1 = max(0, static_cast<int>(fx1));
	const int x2_0 = min(x1_0 + 1, srcW - 1);
	const int x2_1 = min(x1_1 + 1, srcW - 1);
	const float v11_0 = __half2float(s_ptr[row1 + x1_0]);
	const float v11_1 = __half2float(s_ptr[row1 + x1_1]);
	const float v21_0 = __half2float(s_ptr[row2 + x1_0]);
	const float v21_1 = __half2float(s_ptr[row2 + x1_1]);
	const float top0 = fmaf(wx0, __half2float(s_ptr[row1 + x2_0]) - v11_0, v11_0);
	const float top1 = fmaf(wx1, __half2float(s_ptr[row1 + x2_1]) - v11_1, v11_1);
	const float res0 = fmaf(wy, fmaf(wx0, __half2float(s_ptr[row2 + x2_0]) - v21_0, v21_0) - top0, top0);
	const float res1 = fmaf(wy, fmaf(wx1, __half2float(s_ptr[row2 + x2_1]) - v21_1, v21_1) - top1, top1);
	if (x + 1 < dstDim) *reinterpret_cast<__half2*>(dst + blockIdx.z * dstDim * dstDim + y * dstDim + x) = __floats2half2_rn(res0, res1);
	else dst[blockIdx.z * dstDim * dstDim + y * dstDim + x] = __float2half(res0);
}
template <int TILE_DIM = 32>
__global__ void NCHWtoNHWCKernel(const __half* __restrict__ src, __half* __restrict__ dst, int dimension, int channels)
{
	__shared__ __half tile[TILE_DIM][TILE_DIM + 1];
	const int spatial_size = dimension * dimension;
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	if (x < spatial_size && y < channels) tile[threadIdx.y][threadIdx.x] = src[y * spatial_size + x];
	__syncthreads();
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;
	if (x < channels && y < spatial_size) dst[y * channels + x] = tile[threadIdx.x][threadIdx.y];
}
template <int dummy = 0>
__global__ void setActionsKernel(const __half* __restrict__ arr64, action_type* __restrict__ actions, int actionCount)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool is_set = false;
	if (idx < actionCount)
	{
		const float val = __half2float(arr64[idx]);
		is_set = getFastRandomFloatBoundedUniform(generateUniqueSeed(static_cast<int>(val * fmaximum) ^ idx), 1.0f) < val;
	}
	const uint32_t warp_mask = __ballot_sync(__activemask(), is_set);
	if (idx < actionCount && (idx & 31) == 0) atomicOr(reinterpret_cast<unsigned long long*>(&actions[idx >> 6]), static_cast<unsigned long long>(warp_mask) << (idx & 32));
}
template <int dummy = 0>
__global__ void __launch_bounds__(gpu_block_threads, 4) huberKernel(const __half* __restrict__ pred, const __half* __restrict__ real, float* __restrict__ out_loss, int num_elements, float delta)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int element_idx = idx << 3;
	float h[8] = {};
	if (element_idx + 7 < num_elements)
	{
		const uint4 pred_v4 = reinterpret_cast<const uint4*>(pred)[idx];
		const uint4 real_v4 = reinterpret_cast<const uint4*>(real)[idx];
		const auto& p_h2 = reinterpret_cast<const __half(&)[8]>(pred_v4);
		const auto& r_h2 = reinterpret_cast<const __half(&)[8]>(real_v4);
		float diff[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) diff[i] = __half2float(p_h2[i]) - __half2float(r_h2[i]);
		#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const float abs_d = fabsf(diff[i]);
			h[i] = abs_d <= delta ? 0.5f * diff[i] * diff[i] : delta * fmaf(-0.5f, delta, abs_d);
		}
	}
	else
	{
		#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int idx_local = i + element_idx;
			if (idx_local < num_elements)
			{
				const float diff = __half2float(pred[idx_local]) - __half2float(real[idx_local]);
				const float abs_d = fabsf(diff);
				h[i] = abs_d <= delta ? 0.5f * diff * diff : delta * fmaf(-0.5f, delta, abs_d);
			}
		}
	}
	float thread_sum = 0.0f;
	#pragma unroll
	for (int i = 0; i < 4; i++) thread_sum += h[i << 1] + h[(i << 1) + 1];
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, offset);
	if ((threadIdx.x & 31) == 0) atomicAdd(out_loss, thread_sum);
}
template <typename T = __half>
__global__ void __launch_bounds__(gpu_block_threads, 4) huberGradOutputKernel(const __half* __restrict__ pred, const __half* __restrict__ real, T* __restrict__ grad_output, int spatial_size, int channels, float score_factor, float delta)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= channels) return;
	const float factor = score_factor / spatial_size; 
	float sum[4] = {};
	#pragma unroll 4
	for (int s = 0; s < spatial_size; s += 4)
	{
		float diff[4];
		#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			const int offset = (s + i) * channels + idx;
			diff[i] = __half2float(pred[offset]) - __half2float(real[offset]);
		}
		#pragma unroll
		for (int i = 0; i < 4; i++) sum[i] += fabsf(diff[i]) <= delta ? diff[i] : copysignf(delta, diff[i]);
	}
	float thread_sum = 0.0f;
	#pragma unroll
	for (int i = 0; i < 2; i++) thread_sum += sum[i << 1] + sum[(i << 1) + 1];
	if (thread_sum != 0.0f) atomicAdd(&grad_output[idx], static_cast<T>(thread_sum * factor));
}
__host__ __device__ __forceinline__ constexpr int constexpr_clz(unsigned int x)
{
	if (x == 0) return 32;
	int n = 0;
	if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
	if ((x & 0xFF000000) == 0) { n += 8; x <<= 8; }
	if ((x & 0xF0000000) == 0) { n += 4; x <<= 4; }
	if ((x & 0xC0000000) == 0) { n += 2; x <<= 2; }
	if ((x & 0x80000000) == 0) { n += 1; }
	return n;
}
template <int resolution = img_resolution>
__global__ void fftRowKernel(const __half* __restrict__ input, __half2* __restrict__ output)
{
	__shared__ __half2 shared_data[resolution + 1];
	constexpr int log2_n = 31 - constexpr_clz(resolution), total_butterflies = resolution >> 1;
	const int row_offset_in = blockIdx.x * (resolution << 1);
	const int row_offset_out = blockIdx.x * resolution;
	for (int idx = threadIdx.x; idx < resolution; idx += gpu_block_threads) shared_data[__brev(idx) >> (32 - log2_n)] = __halves2half2(input[row_offset_in + (idx << 1)], input[row_offset_in + (idx << 1) + 1]);
	__syncthreads();
	#pragma unroll 2
	for (int s = 1; s <= log2_n; ++s)
	{
		const int m = 1 << s, half_s = m >> 1;
		#pragma unroll
		for (int b_offset = 0; b_offset < total_butterflies; b_offset += gpu_block_threads)
		{
			const int b_idx = b_offset + threadIdx.x;
			__half2 t = h2_zero(), u = h2_zero();
			int u_idx = 0, v_idx = 0;
			if (b_idx < total_butterflies)
			{
				const int local_idx = b_idx % half_s;
				u_idx = b_idx / half_s * m + local_idx;
				v_idx = u_idx + half_s;
				float s_val, c_val;
				__sincosf(-6.2831853f * local_idx / m, &s_val, &c_val);
				u = shared_data[u_idx];
				t = complex_mul_half2(shared_data[v_idx], __floats2half2_rn(c_val, s_val));
			}
			__syncthreads();
			if (b_idx < total_butterflies)
			{
				shared_data[u_idx] = __hadd2(u, t);
				shared_data[v_idx] = __hsub2(u, t);
			}
			__syncthreads();
		}
	}
	for (int idx = threadIdx.x; idx < resolution; idx += gpu_block_threads) output[row_offset_out + idx] = shared_data[idx];
}
template <int resolution = img_resolution>
__global__ void fftColumnKernel(const __half2* __restrict__ input, __half* __restrict__ output)
{
	__shared__ __half2 shared_data[resolution + 1];
	constexpr int log2_n = 31 - constexpr_clz(resolution), total_butterflies = resolution >> 1;
	constexpr float norm = 1.0f / (resolution * resolution);
	for (int idx = threadIdx.x; idx < resolution; idx += gpu_block_threads) shared_data[__brev(idx) >> (32 - log2_n)] = input[idx * resolution + blockIdx.x];
	__syncthreads();
	#pragma unroll 2
	for (int s = 1; s <= log2_n; ++s)
	{
		const int m = 1 << s, half_s = m >> 1;
		#pragma unroll
		for (int b_offset = 0; b_offset < total_butterflies; b_offset += gpu_block_threads)
		{
			const int b_idx = b_offset + threadIdx.x;
			__half2 t = h2_zero(), u = h2_zero();
			int u_idx = 0, v_idx = 0;
			if (b_idx < total_butterflies)
			{
				const int local_idx = b_idx % half_s;
				u_idx = b_idx / half_s * m + local_idx;
				v_idx = u_idx + half_s;
				float s_val, c_val;
				__sincosf(-6.2831853f * local_idx / m, &s_val, &c_val);
				u = shared_data[u_idx];
				t = complex_mul_half2(shared_data[v_idx], __floats2half2_rn(c_val, s_val));
			}
			__syncthreads();
			if (b_idx < total_butterflies)
			{
				shared_data[u_idx] = __hadd2(u, t);
				shared_data[v_idx] = __hsub2(u, t);
			}
			__syncthreads();
		}
	}
	for (int row_idx = threadIdx.x; row_idx < resolution; row_idx += gpu_block_threads)
	{
		const __half2 res = shared_data[row_idx];
		const float re = __half2float(res.x);
		const float im = __half2float(res.y);
		const int offset_out = row_idx * resolution + blockIdx.x;
		output[offset_out] = __float2half(__fsqrt_rn(fmaf(re, re, im * im)) * norm);
		output[offset_out + resolution * resolution] = __float2half(atan2f(im, re));
	}
}
template <int dummy = 0>
__global__ void gapKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, __half* __restrict__ pooled_features, int spatial_size, int InC)
{
	const int base_ch_idx = (blockIdx.x * blockDim.x + threadIdx.x) << 5;
	if (base_ch_idx >= InC) return;
	const int groups_per_in = InC >> 5;
	const int scale_group_idx = base_ch_idx >> 5;
	float sum[32];
	#pragma unroll
	for (int c = 0; c < 32; ++c) sum[c] = 0.0f;
	const uint4* __restrict__ input_v4 = reinterpret_cast<const uint4*>(input + (base_ch_idx >> 3));
	for (int i = 0; i < spatial_size; ++i) 
	{
		uint4 packed_v4 = input_v4[i * groups_per_in];
		const auto& packed_words = reinterpret_cast<const uint32_t(&)[4]>(packed_v4);
		const int8_t s_quant = input_scales[i * groups_per_in + scale_group_idx];
		#pragma unroll
		for (int w = 0; w < 4; ++w)
		{
			#pragma unroll
			for (int e = 0; e < 8; ++e) sum[(w << 3) + e] += dequantize_scalar(packed_words[w], e, s_quant);
		}
	}
	#pragma unroll
	for (int c = 0; c < 32; ++c) pooled_features[base_ch_idx + c] = __float2half(sum[c] / spatial_size);
}
template <int dummy = 0>
__global__ void criticFwd1Kernel(const __half* __restrict__ hidden_pool, const __half* __restrict__ weights1, const __half* __restrict__ weights_lambda, __half* __restrict__ hidden_state, float* __restrict__ trace_w1, float* __restrict__ trace_lambda, int out_channels, int in_channels)
{
	const int idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (idx >= out_channels) return;
	const int lane_id = threadIdx.x & 31;
	const __half2* w1_v2 = reinterpret_cast<const __half2*>(weights1);
	const __half2* wl_v2 = reinterpret_cast<const __half2*>(weights_lambda);
	float2 acc_val = make_float2(0.0f, 0.0f), acc_lam = make_float2(0.0f, 0.0f);
	for (int c = lane_id; c < in_channels; c += 32)
	{
		const float x = __half2float(hidden_pool[c]);
		const __half2 w1_val = w1_v2[c * out_channels + idx];
		const __half2 lam_val = wl_v2[idx * in_channels + c];
		acc_val.x = fmaf(x, __half2float(w1_val.x), acc_val.x);
		acc_val.y = fmaf(x, __half2float(w1_val.y), acc_val.y);
		acc_lam.x = fmaf(x, __half2float(lam_val.x), acc_lam.x);
		acc_lam.y = fmaf(x, __half2float(lam_val.y), acc_lam.y);
	}
	#pragma unroll
	for (int i = 16; i > 0; i >>= 1) 
	{
		acc_val.x += __shfl_down_sync(0xFFFFFFFF, acc_val.x, i);
		acc_val.y += __shfl_down_sync(0xFFFFFFFF, acc_val.y, i);
		acc_lam.x += __shfl_down_sync(0xFFFFFFFF, acc_lam.x, i);
		acc_lam.y += __shfl_down_sync(0xFFFFFFFF, acc_lam.y, i);
	}
	acc_val.x = __shfl_sync(0xFFFFFFFF, acc_val.x, 0);
	acc_val.y = __shfl_sync(0xFFFFFFFF, acc_val.y, 0);
	acc_lam.x = __shfl_sync(0xFFFFFFFF, acc_lam.x, 0);
	acc_lam.y = __shfl_sync(0xFFFFFFFF, acc_lam.y, 0);
	__half2* h_state_v2 = reinterpret_cast<__half2*>(hidden_state);
	const __half2 h_old = h_state_v2[idx];
	float sigm_re, sigm_im, lambda_soft, lam_mod_inv;
	if (lane_id == 0)
	{
		sigm_re = __frcp_rn(1.0f + expf(-acc_val.x));
		sigm_im = __frcp_rn(1.0f + expf(-acc_val.y));
		const float lam_mod = sqrtf(fmaf(acc_lam.x, acc_lam.x, fmaf(acc_lam.y, acc_lam.y, epsilon)));
		lambda_soft = __frcp_rn(1.0f + expf(-lam_mod));
		lam_mod_inv = __frcp_rn(lam_mod);
	}
	sigm_re = __shfl_sync(0xFFFFFFFF, sigm_re, 0);
	sigm_im = __shfl_sync(0xFFFFFFFF, sigm_im, 0);
	lambda_soft = __shfl_sync(0xFFFFFFFF, lambda_soft, 0);
	lam_mod_inv = __shfl_sync(0xFFFFFFFF, lam_mod_inv, 0);
	const float dh_dact = 1.0f - lambda_soft;
	const float2 act_curr = make_float2(acc_val.x * sigm_re, acc_val.y * sigm_im);
	const float2 h_old_f = __half22float2(h_old);
	if (lane_id == 0) h_state_v2[idx] = __floats2half2_rn(fmaf(h_old_f.x, lambda_soft, dh_dact * act_curr.x), fmaf(h_old_f.y, lambda_soft, dh_dact * act_curr.y));
	const float act_grad_re = dh_dact * sigm_re * fmaf(acc_val.x, 1.0f - sigm_re, 1.0f);
	const float act_grad_im = dh_dact * sigm_im * fmaf(acc_val.y, 1.0f - sigm_im, 1.0f);
	const float dlambda = (h_old_f.x - act_curr.x + h_old_f.y - act_curr.y) * lambda_soft * dh_dact * lam_mod_inv;
	float2* t_w1_v2 = reinterpret_cast<float2*>(trace_w1);
	float2* t_lam_v2 = reinterpret_cast<float2*>(trace_lambda);
	for (int c = lane_id; c < in_channels; c += 32)
	{
		const float x_input = __half2float(hidden_pool[c]);
		const int w1_idx = c * out_channels + idx, wl_idx = idx * in_channels + c;
		float2 t_w1 = t_w1_v2[w1_idx];
		t_w1.x = fmaf(t_w1.x, lambda_soft, act_grad_re * x_input);
		t_w1.y = fmaf(t_w1.y, lambda_soft, act_grad_im * x_input);
		t_w1_v2[w1_idx] = t_w1;
		float2 t_lam = t_lam_v2[wl_idx];
		t_lam.x = fmaf(t_lam.x, lambda_soft, dlambda * x_input * acc_lam.x);
		t_lam.y = fmaf(t_lam.y, lambda_soft, dlambda * x_input * acc_lam.y);
		t_lam_v2[wl_idx] = t_lam;
	}
}
template <int dummy = 0>
__global__ void criticFwd2Kernel(const __half* __restrict__ hidden_state, const __half* __restrict__ weights2, float* __restrict__ output_score, int out_channels)
{
	__shared__ float warp_sums[32];
	float local_sum = 0.0f;
	const __half2* h_state_v2 = reinterpret_cast<const __half2*>(hidden_state);
	const __half2* w2_v2 = reinterpret_cast<const __half2*>(weights2);
	for (int i = threadIdx.x; i < out_channels; i += blockDim.x) 
	{
		const float2 f = __half22float2(h_state_v2[i]);
		const float2 w2 = __half22float2(w2_v2[i]);
		local_sum = fmaf(f.x, w2.x, fmaf(f.y, w2.y, local_sum));
	}
	#pragma unroll
	for (int i = 16; i > 0; i >>= 1) local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, i);
	const int lane_id = threadIdx.x & 31, warpId = threadIdx.x >> 5, num_warps = blockDim.x >> 5;
	if (lane_id == 0) warp_sums[warpId] = local_sum;
	__syncthreads();
	if (warpId == 0) 
	{
		float final_warp_sum = lane_id < num_warps ? warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for (int i = 16; i > 0; i >>= 1) final_warp_sum += __shfl_down_sync(0xFFFFFFFF, final_warp_sum, i);
		if (threadIdx.x == 0) *output_score = final_warp_sum;
	}
}
template <int dummy = 0>
__global__ void criticApplyTracesKernel(const __half* __restrict__ w2, const float* __restrict__ trace_w1, const float* __restrict__ trace_lambda, __half* __restrict__ grad_weights1, __half* __restrict__ grad_lambda, float pred_score, float real_score, int out_channels, int in_channels)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= out_channels * in_channels) return;
	const float2* t_w1_v2 = reinterpret_cast<const float2*>(trace_w1);
	const float2* t_lam_v2 = reinterpret_cast<const float2*>(trace_lambda);
	const __half2* w2_v2 = reinterpret_cast<const __half2*>(w2);
	__half2* g_w1_v2 = reinterpret_cast<__half2*>(grad_weights1);
	__half2* g_lam_v2 = reinterpret_cast<__half2*>(grad_lambda);
	const float2 t_w1 = t_w1_v2[idx];
	const float2 t_lam = t_lam_v2[idx];
	const float grad_out = pred_score - real_score;
	const float2 w2_node = __half22float2(w2_v2[idx & (out_channels - 1)]);
	const float2 w2_lam_node = __half22float2(w2_v2[idx / in_channels]);
	g_w1_v2[idx] = __floats2half2_rn(grad_out * w2_node.x * t_w1.x, grad_out * w2_node.y * t_w1.y);
	g_lam_v2[idx] = __floats2half2_rn(grad_out * w2_lam_node.x * t_lam.x, grad_out * w2_lam_node.y * t_lam.y);
}
template <int dummy = 0>
__global__ void criticOutputGradKernel(const __half* __restrict__ hidden_state, __half* __restrict__ grad_w2, float pred_score, float real_score, int out_channels)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= out_channels) return;
	const float grad_out = pred_score - real_score;
	const __half2* h_state_v2 = reinterpret_cast<const __half2*>(hidden_state);
	__half2* g_w2_v2 = reinterpret_cast<__half2*>(grad_w2);
	const float2 f = __half22float2(h_state_v2[idx]);
	g_w2_v2[idx] = __floats2half2_rn(grad_out * f.x, grad_out * f.y);
}
