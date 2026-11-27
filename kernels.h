#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
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

__host__ __device__ __forceinline__ __half2_raw h2_zero() { return __half2_raw{0x0000, 0x0000}; }
__host__ __device__ __forceinline__ __half2_raw h2_one() { return __half2_raw{0x3C00, 0x3C00}; }

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
__host__ __device__ __forceinline__ float getFastRandomFloatBoundedUniform(uint32_t seed, float n) { return getFastRandomModifier(seed) * (1.0f / 0x1p32f) * n; }
__host__ __device__ __forceinline__ float getFastRandomFloatBoundedNormal(uint32_t seed, float n) { return -n + getFastRandomFloatBoundedUniform(seed, n) * 2.0f; }
template <typename T>
inline void generateNormalRandoms(universal_vector<T>& arr_it, int size, float offset, cudaStream_t stream, int seed = 1234)
{
	T* raw_ptr = arr_it.data();
	auto functor = [raw_ptr, size, offset, seed] __host__ __device__ (int idx) -> void { raw_ptr[idx] = static_cast<T>(getFastRandomFloatBoundedNormal(generateUniqueSeed(idx * seed + size), offset)); };
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
__device__ __forceinline__ float sigmoid(float x) { return __frcp_rn(1.0f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float silu_grad(float x, float dy)
{
	const float sig = sigmoid(x);
	return dy * sig * fmaf(x, 1.0f - sig, 1.0f);
}
__device__ __forceinline__ float sigmoid_grad(float x, float dy)
{
	const float sig = sigmoid(x);
	return dy * sig * (1.0f - sig);
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

template <typename T, bool disable_stochastic = false>
__global__ void __launch_bounds__(gpu_block_threads, 4) quantizeKernel(const T* __restrict__ input, uint32_t* __restrict__ output, int8_t* __restrict__ scales, int total_elements, int stochastic_add_idx)
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
	if constexpr (!disable_stochastic)
	{
		const uint32_t thread_seed = generateUniqueSeed(idx + stochastic_add_idx);
		#pragma unroll
		for (int i = 0; i < 32; ++i) out_words[i >> 3] |= (quantize_scalar_to_int4_stochastic(static_cast<float>(local_data[i]), inv_scale, thread_seed ^ i) << ((i & 7) << 2));
	}
	else
	{
		#pragma unroll
		for (int i = 0; i < 32; ++i) out_words[i >> 3] |= (quantize_scalar_to_int4(static_cast<float>(local_data[i]), inv_scale) << ((i & 7) << 2));
	}
	reinterpret_cast<uint4*>(output)[idx] = *reinterpret_cast<uint4*>(out_words);
}
template <typename T, bool disable_stochastic = true, int tile_dim = 32>
__global__ void __launch_bounds__(gpu_block_threads, 4) quantizeNCHWtoNHWCKernel(const T* __restrict__ input, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int dimension, int channels, int seed)
{
	constexpr int spatial_per_warp = tile_dim / (gpu_block_threads >> 5);
	const int base_ch = blockIdx.z << 5;
	if (base_ch >= channels || blockIdx.y >= dimension) return;
	__shared__ T tile[tile_dim][tile_dim + 1];
	const int spatial_size = dimension * dimension;
	const int global_row = blockIdx.y;
	const int tile_base_x = blockIdx.x << 5;
	#pragma unroll
	for (int i = 0; i < (tile_dim * tile_dim) / gpu_block_threads; ++i)
	{
		const int linear_idx = threadIdx.x + i * gpu_block_threads;
		const int ch_off = linear_idx >> 5;
		const int sp_x = linear_idx & 31;
		const int global_x = tile_base_x + sp_x;
		tile[sp_x][ch_off] = global_x < dimension && base_ch + ch_off < channels ? input[(base_ch + ch_off) * spatial_size + global_row * dimension + global_x] : static_cast<T>(0);
	}
	const int warp_id = threadIdx.x >> 5;
	const int lane_id = threadIdx.x & 31;
	const int lane_id_v8 = lane_id & 7;
	__syncthreads();
	#pragma unroll
	for (int s = 0; s < spatial_per_warp; ++s)
	{
		const int sp_x_in_tile = warp_id * spatial_per_warp + s;
		const int global_x = tile_base_x + sp_x_in_tile;
		const int global_sp = global_row * dimension + global_x;
		const bool x_valid = global_x < dimension;
		const float my_val = x_valid ? static_cast<float>(tile[sp_x_in_tile][lane_id]) : 0.0f;
		float abs_val = fabsf(my_val);
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, offset));
		const float scale = abs_val < epsilon ? 1.0f : abs_val;
		const float inv_scale = quant_int4_max * __frcp_rn(scale);
		int32_t q_val;
		if constexpr (!disable_stochastic) q_val = quantize_scalar_to_int4_stochastic(my_val, inv_scale, generateUniqueSeed(spatial_size * channels + global_sp * channels + base_ch + lane_id + seed));
		else q_val = quantize_scalar_to_int4(my_val, inv_scale);
		uint32_t packed = static_cast<uint32_t>(q_val & 0xF) << (lane_id_v8 << 2);
		#pragma unroll
		for (int sh = 4; sh > 0; sh >>= 1) packed |= __shfl_down_sync(0xFFFFFFFF, packed, sh);
		if (x_valid)
        {
			const int ch_base_idx = global_sp * (channels >> 3);
			if (lane_id_v8 == 0) output[ch_base_idx + (blockIdx.z << 2) + (lane_id >> 3)] = packed;
			if (lane_id == 0) output_scales[(ch_base_idx >> 2) + blockIdx.z] = quantize_scale(scale);
        }
	}
}
template <int dummy = 0>
__global__ void __launch_bounds__(gpu_block_threads, 4) updateParamsKernel(float* __restrict__ data, const float* __restrict__ grad, int size, float grad_norm_factor, float learning_rate, float penalty)
{
	constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(float);
	const float c_grad = -learning_rate * grad_norm_factor;
	const float c_data = fmaf(-learning_rate, penalty, 1.0f);
	const uint4* __restrict__ grad_v4 = reinterpret_cast<const uint4*>(grad);
	uint4* __restrict__ data_v4 = reinterpret_cast<uint4*>(data);
	const int size_v4 = size / elements_per_uint4;
	const int stride = blockDim.x * gridDim.x;
	#pragma unroll 4
	for (int v_idx = blockIdx.x * blockDim.x + threadIdx.x; v_idx < size_v4; v_idx += stride)
	{
		float local_d[elements_per_uint4], local_g[elements_per_uint4];
		*reinterpret_cast<uint4*>(local_d) = data_v4[v_idx];
		*reinterpret_cast<uint4*>(local_g) = grad_v4[v_idx];
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; ++i) local_d[i] = fmaf(local_g[i], c_grad, local_d[i] * c_data);
		data_v4[v_idx] = *reinterpret_cast<uint4*>(local_d);
	}
}
template <bool disable_activation = false, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseFwdKernel(const uint32_t* __restrict__ input, const uint32_t* __restrict__ weights, const int8_t* __restrict__ input_scales, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int OutC)
{
	constexpr int warps = channels_per_block >> 5;
	const int outc = blockIdx.y * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int spatial_groups = blockIdx.x * groups_per_in, outc_groups = outc * groups_per_in;
	const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
	const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + blockIdx.x * u32_stride_in);
	const int lane_id = threadIdx.x & 31;
	uint4 curr_w = w_v4[0], curr_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
	int8_t curr_in_scale = input_scales[spatial_groups], curr_w_scale = weight_scales[outc_groups];
	float activated_acc = biases[outc];
	float acc = 0.0f;
	for (int g = 1; g < groups_per_in; ++g)
	{
		uint4 next_w = w_v4[g], next_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
		const int8_t next_in_scale = input_scales[spatial_groups + g], next_w_scale = weight_scales[outc_groups + g];
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		int partial_sum = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
		acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, acc);
		curr_w = next_w;
		curr_in_v4 = next_in_v4;
		curr_in_scale = next_in_scale;
		curr_w_scale = next_w_scale;
	}
	const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
	const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
	int partial_sum = 0;
	#pragma unroll
	for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
	acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, acc);
	float local_sq = acc * acc;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	const int warp_id = threadIdx.x >> 5;
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	activated_acc = fmaf(acc, __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon)), activated_acc);
	if constexpr (!disable_activation) activated_acc = silu(activated_acc);
	float group_max = fabsf(activated_acc);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max = fmaxf(group_max, __shfl_xor_sync(0xFFFFFFFF, group_max, offset));
	const float out_scale = group_max < epsilon ? 1.0f : group_max;
	const int lane_id_v8 = lane_id & 7;
	uint32_t val = quantize_scalar_to_int4(activated_acc, quant_int4_max * __frcp_rn(out_scale)) << (lane_id_v8 << 2);
	#pragma unroll
	for (int i = 4; i > 0; i >>= 1) val |= __shfl_down_sync(0xFFFFFFFF, val, i);
	const int spatial_idx_out_base = blockIdx.x * (OutC >> 3);
	if (lane_id_v8 == 0) output[spatial_idx_out_base + (outc >> 3)] = val;
	if (lane_id == 0) output_scales[(spatial_idx_out_base >> 2) + (outc >> 5)] = quantize_scale(out_scale);
}
template <bool disable_activation = false, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradInputKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, const float* __restrict__ master_weights, float* __restrict__ grad_input, int InC, int OutC)
{
	constexpr int warps = channels_per_block >> 5;
	const int block_base_outc = blockIdx.y * channels_per_block;
	const int outc = block_base_outc + threadIdx.x;
	__shared__ float smem_warps[warps], smem_dy_normed[channels_per_block];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int spatial_groups = blockIdx.x * groups_per_in, outc_groups = outc * groups_per_in;
	const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
	const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + blockIdx.x * u32_stride_in);
	const int lane_id = threadIdx.x & 31;
	uint4 curr_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0), curr_w_v4 = w_v4[0];
	int8_t curr_in_scale = input_scales[spatial_groups], curr_w_scale = weight_scales[outc_groups];
	float fwd_acc = 0.0f;
	uint32_t dot_shared[4];
	for (int g = 1; g < groups_per_in; ++g)
	{
		uint4 next_w_v4 = w_v4[g], next_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
		const int8_t next_in_scale = input_scales[spatial_groups + g], next_w_scale = weight_scales[outc_groups + g];
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w_v4);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		int partial_sum = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
		fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
		curr_w_v4 = next_w_v4;
		curr_in_v4 = next_in_v4;
		curr_in_scale = next_in_scale;
		curr_w_scale = next_w_scale;
	}
	const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w_v4);
	const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
	int partial_sum = 0;
	#pragma unroll
	for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
	#pragma unroll
	for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
	[[maybe_unused]] const float bias = biases[outc];
	fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
	float local_sq = fwd_acc * fwd_acc;
	const float dy = grad_output[blockIdx.x * OutC + outc];
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	const int warp_id = threadIdx.x >> 5;
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	const float inv_rms = __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon));
	if constexpr (!disable_activation) smem_dy_normed[threadIdx.x] = silu_grad(fmaf(fwd_acc, inv_rms, bias), dy) * inv_rms;
	else smem_dy_normed[threadIdx.x] = dy * inv_rms;
	__syncthreads();
	const int block_base_dim = blockIdx.x * InC;
	for (int inc_base = 0; inc_base < InC; inc_base += channels_per_block)
	{
		const int curr_inc = inc_base + threadIdx.x;
		if (curr_inc < InC)
		{
			float acc_input = 0.0f;
			#pragma unroll 4
			for (int i = 0; i < channels_per_block; ++i) acc_input = fmaf(smem_dy_normed[i], master_weights[block_base_outc + i * InC + curr_inc], acc_input);
			atomicAdd(&grad_input[block_base_dim + curr_inc], acc_input);
		}
	}
}
template <bool disable_activation = false, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradWeightKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, float* __restrict__ grad_weights, int InC, int OutC, int NHW)
{
	const int outc = blockIdx.x * dim_block_pw_bwd_gw_size + threadIdx.x;
	const int inc_load = blockIdx.y * dim_block_pw_bwd_gw_size + threadIdx.x;
	const int lane_id = threadIdx.x & 31;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int outc_groups = outc * groups_per_in;
	constexpr int warps_per_gn_group = channels_per_block / dim_block_pw_bwd_gw_size;
	constexpr int gn_groups_per_block = dim_block_pw_bwd_gw_size * dim_block_pw_bwd_gw_size / channels_per_block;
	__shared__ float smem_in[dim_block_pw_bwd_gw_size][dim_block_pw_bwd_gw_size + 1];
	__shared__ float smem_gn_warps[dim_block_pw_bwd_gw_size];
	__shared__ float smem_gn_groups[gn_groups_per_block > 0 ? gn_groups_per_block : 1];
	__shared__ float smem_grad_out[dim_block_pw_bwd_gw_size][dim_block_pw_bwd_gw_size + 1];
	float sum[4] = {};
	[[maybe_unused]] const float bias = biases[outc];
	const float min_inv_chunk_out = __frcp_rn(static_cast<float>(min(OutC, channels_per_block)));
	for (int p_base = 0; p_base < NHW; p_base += dim_block_pw_bwd_gw_size)
	{
		const int p = p_base + threadIdx.y, spatial_groups_fwd = p * groups_per_in;
		smem_in[threadIdx.y][threadIdx.x] = dequantize_scalar(input[p * u32_stride_in + (inc_load >> 3)], inc_load & 7, input_scales[spatial_groups_fwd + (inc_load >> 5)]);
		const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
		const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + p * u32_stride_in);
		uint4 curr_w = w_v4[0], curr_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t curr_in_scale = input_scales[spatial_groups_fwd], curr_w_scale = weight_scales[outc_groups];
		float fwd_acc = 0.0f;
		uint32_t dot_shared[4];
		for (int g = 1; g < groups_per_in; ++g)
		{
			uint4 next_w = w_v4[g], next_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
			const int8_t next_in_scale = input_scales[spatial_groups_fwd + g], next_w_scale = weight_scales[outc_groups + g];
			const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
			const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
			int partial_sum = 0;
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
			fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
			curr_w = next_w;
			curr_in_v4 = next_in_v4;
			curr_in_scale = next_in_scale;
			curr_w_scale = next_w_scale;
		}
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		int partial_sum = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
		fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
		float local_sq = fwd_acc * fwd_acc;
		const float dy = grad_output[p * OutC + outc];
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
		const float inv_rms = __frsqrt_rn(fmaf(smem_gn_groups[threadIdx.y / warps_per_gn_group], min_inv_chunk_out, epsilon));
		if constexpr (!disable_activation) smem_grad_out[threadIdx.y][threadIdx.x] = silu_grad(fmaf(fwd_acc, inv_rms, bias), dy) * inv_rms;
		else smem_grad_out[threadIdx.y][threadIdx.x] = dy * inv_rms;
		float go[4], gi[4];
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < dim_block_pw_bwd_gw_size; k += 4)
		{
			#pragma unroll
			for (int i = 0; i < 4; i++) gi[i] = smem_in[threadIdx.y][k + i];
			#pragma unroll
			for (int i = 0; i < 4; i++) go[i] = smem_grad_out[k + i][threadIdx.x];
			#pragma unroll
			for (int i = 0; i < 4; i++) sum[i] = fmaf(gi[i], go[i], sum[i]);
		}
		__syncthreads();
	}
	grad_weights[outc * InC + blockIdx.y * dim_block_pw_bwd_gw_size + threadIdx.y] = sum[0] + sum[1] + sum[2] + sum[3];
}
template <bool disable_activation = false, int channels_per_block = gpu_block_threads>
__global__ void convPointwiseBwdGradBiasKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, float* __restrict__ grad_biases, int InC, int OutC, int NHW)
{
	constexpr int warps = channels_per_block >> 5;
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int lane_id = threadIdx.x & 31;
	const int groups_per_in = InC >> 5, u32_stride_in = InC >> 3;
	const int outc_groups = outc * groups_per_in;
	float bias_grad_accumulator = 0.0f;
	[[maybe_unused]] const float bias = biases[outc];
	const float min_inv_chunk_out = __frcp_rn(static_cast<float>(min(OutC, channels_per_block)));
	for (int p = 0; p < NHW; ++p)
	{
		const int spatial_groups = p * groups_per_in;
		const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups;
		const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + p * u32_stride_in);
		uint4 curr_w = w_v4[0], curr_in_v4 = lane_id == 0 ? input_v4_base[0] : make_uint4(0, 0, 0, 0);
		int8_t curr_in_scale = input_scales[spatial_groups], curr_w_scale = weight_scales[outc_groups];
		float fwd_acc = 0.0f;
		uint32_t dot_shared[4];
		for (int g = 1; g < groups_per_in; ++g)
		{
			uint4 next_w = w_v4[g], next_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
			const int8_t next_in_scale = input_scales[spatial_groups + g], next_w_scale = weight_scales[outc_groups + g];
			const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
			const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
			int partial_sum = 0;
			#pragma unroll
			for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
			#pragma unroll
			for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
			fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
			curr_w = next_w;
			curr_in_v4 = next_in_v4;
			curr_in_scale = next_in_scale;
			curr_w_scale = next_w_scale;
		}
		const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
		const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
		int partial_sum = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i) dot_shared[i] = __shfl_sync(0xFFFFFFFF, shared_in[i], 0);
		#pragma unroll
		for (int i = 0; i < 4; ++i) partial_sum += dot_int4(dot_shared[i], w_val_ptr[i]);
		fwd_acc = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc);
		float local_sq = fwd_acc * fwd_acc;
		const float dy = grad_output[p * OutC + outc];
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
		const int warp_id = threadIdx.x >> 5;
		if (lane_id == 0) smem_warps[warp_id] = local_sq;
		__syncthreads();
		if (warp_id == 0)
		{
			float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
			#pragma unroll
			for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
			if (lane_id == 0) smem_warps[0] = block_sq;
		}
		__syncthreads();
		const float inv_rms = __frsqrt_rn(fmaf(smem_warps[0], min_inv_chunk_out, epsilon));
		if constexpr (!disable_activation) bias_grad_accumulator += silu_grad(fmaf(fwd_acc, inv_rms, bias), dy);
		else bias_grad_accumulator += dy;
	}
	grad_biases[outc] = bias_grad_accumulator;
}
template <bool disable_activation = true, int kernel_size = 4, int channels_per_block = gpu_block_threads>
__global__ void convDenseFwdKernel(const uint32_t* __restrict__ input, const uint32_t* __restrict__ weights, const int8_t* __restrict__ input_scales, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int k_sq = kernel_size * kernel_size, warps = channels_per_block >> 5;
	const int outc = blockIdx.y * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int padding = kernel_size == stride ? 0 : (kernel_size - 1) >> 1;
	const int h_in_start = blockIdx.x / Dimension_out * stride - padding;
	const int w_in_start = (blockIdx.x & (Dimension_out - 1)) * stride - padding;
	const int groups_per_in = InC >> 5;
	const int u32_stride_in = InC >> 3;
	const int lane_id = threadIdx.x & 31;
	const int outc_groups = outc * k_sq * groups_per_in;
	float acc_arr[4] = {};
	#pragma unroll
	for (int k = 0; k < k_sq; ++k)
	{
		const int h_in = h_in_start + k / kernel_size;
		const int w_in = w_in_start + k % kernel_size;
		if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
		{
			const int spatial_idx_in = h_in * Dimension_in + w_in;
			const int spatial_groups = spatial_idx_in * groups_per_in;
			const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
			const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups + k * groups_per_in;
			for (int g = 0; g < groups_per_in; ++g)
			{
				uint4 curr_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
				uint4 curr_w = w_v4[g];
				const int8_t curr_in_scale = input_scales[spatial_groups + g];
				const int8_t curr_w_scale = weight_scales[outc_groups + k * groups_per_in + g];
				const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
				const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
				int partial_sum = 0;
				#pragma unroll
				for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
				acc_arr[k & 3] = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, acc_arr[k & 3]);
			}
		}
	}
	float activated_acc = biases[outc];
	const float acc = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];
	float local_sq = acc * acc;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	const int warp_id = threadIdx.x >> 5;
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	activated_acc = fmaf(acc, __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon)), activated_acc);
	if constexpr (!disable_activation) activated_acc = silu(activated_acc);
	float group_max = fabsf(activated_acc);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max = fmaxf(group_max, __shfl_xor_sync(0xFFFFFFFF, group_max, offset));
	const float out_scale = group_max < epsilon ? 1.0f : group_max;
	const int lane_id_v8 = lane_id & 7;
	uint32_t val = quantize_scalar_to_int4(activated_acc, quant_int4_max * __frcp_rn(out_scale)) << (lane_id_v8 << 2);
	#pragma unroll
	for (int i = 4; i > 0; i >>= 1) val |= __shfl_down_sync(0xFFFFFFFF, val, i);
	const int spatial_idx_out_base = blockIdx.x * (OutC >> 3);
	if (lane_id_v8 == 0) output[spatial_idx_out_base + (outc >> 3)] = val;
	if (lane_id == 0) output_scales[(spatial_idx_out_base >> 2) + (outc >> 5)] = quantize_scale(out_scale);
}
template <bool disable_activation = true, int kernel_size = 4, int channels_per_block = gpu_block_threads>
__global__ void convDenseBwdGradInputKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, const float* __restrict__ master_weights, float* __restrict__ grad_input, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int k_sq = kernel_size * kernel_size, warps = channels_per_block >> 5;
	const int outc = blockIdx.y * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps], smem_dy_normed[channels_per_block];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int padding = kernel_size == stride ? 0 : (kernel_size - 1) >> 1;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int h_in_start = blockIdx.x / Dimension_out * stride - padding;
	const int w_in_start = (blockIdx.x & (Dimension_out - 1)) * stride - padding;
	const int groups_per_in = InC >> 5;
	const int u32_stride_in = InC >> 3;
	const int outc_groups = outc * k_sq * groups_per_in;
	float fwd_acc_arr[4] = {};
	#pragma unroll
	for (int k = 0; k < k_sq; ++k)
	{
		const int h_in = h_in_start + k / kernel_size;
		const int w_in = w_in_start + k % kernel_size;
		if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
		{
			const int spatial_idx_in = h_in * Dimension_in + w_in;
			const int spatial_groups = spatial_idx_in * groups_per_in;
			const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
			const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups + k * groups_per_in;
			#pragma unroll 4
			for (int g = 0; g < groups_per_in; ++g)
			{
				uint4 curr_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
				uint4 curr_w = w_v4[g];
				const int8_t curr_in_scale = input_scales[spatial_groups + g];
				const int8_t curr_w_scale = weight_scales[outc_groups + k * groups_per_in + g];
				const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
				const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
				int partial_sum = 0;
				#pragma unroll
				for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
				fwd_acc_arr[k & 3] = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc_arr[k & 3]);
			}
		}
	}
	[[maybe_unused]] const float bias = biases[outc];
	const float fwd_acc = fwd_acc_arr[0] + fwd_acc_arr[1] + fwd_acc_arr[2] + fwd_acc_arr[3];
	float local_sq = fwd_acc * fwd_acc;
	const float dy = grad_output[blockIdx.x * OutC + outc];
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	const float inv_rms = __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon));
	if constexpr (!disable_activation) smem_dy_normed[threadIdx.x] = silu_grad(fmaf(fwd_acc, inv_rms, bias), dy) * inv_rms;
	else smem_dy_normed[threadIdx.x] = dy * inv_rms;
	__syncthreads();
	const int block_base_outc = blockIdx.y * channels_per_block;
	for (int inc_base = 0; inc_base < InC; inc_base += channels_per_block)
	{
		const int curr_inc = inc_base + threadIdx.x;
		if (curr_inc < InC)
		{
			#pragma unroll
			for (int k = 0; k < k_sq; ++k)
			{
				const int h_in = h_in_start + k / kernel_size;
				const int w_in = w_in_start + k % kernel_size;
				if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
				{
					float acc_input = 0.0f;
					#pragma unroll 2
					for (int i = 0; i < channels_per_block; ++i)
					{
						const int target_outc = block_base_outc + i;
						if (target_outc < OutC) acc_input = fmaf(smem_dy_normed[i], master_weights[(target_outc * k_sq + k) * InC + curr_inc], acc_input);
					}
					atomicAdd(&grad_input[(h_in * Dimension_in + w_in) * InC + curr_inc], acc_input);
                }
			}
		}
	}
}
template <bool disable_activation = true, int kernel_size = 4, int channels_per_block = gpu_block_threads>
__global__ void convDenseBwdGradWeightKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, float* __restrict__ grad_weights, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int k_sq = kernel_size * kernel_size, warps = channels_per_block >> 5;
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC || blockIdx.y >= InC) return;
	const int padding = kernel_size == stride ? 0 : (kernel_size - 1) >> 1;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int NHW_out = Dimension_out * Dimension_out;
	const int groups_per_in = InC >> 5;
	const int u32_stride_in = InC >> 3;
	const int outc_groups = outc * k_sq * groups_per_in;
	[[maybe_unused]] const float bias = biases[outc];
	float sum_grad_w[k_sq] = {};
	const float min_inv_chunk_out = __frcp_rn(static_cast<float>(min(OutC, channels_per_block)));
	for (int p = 0; p < NHW_out; ++p)
	{
		const int h_in_start = p / Dimension_out * stride - padding;
		const int w_in_start = (p & (Dimension_out - 1)) * stride - padding;
		float fwd_acc_arr[4] = {};
		#pragma unroll
		for (int k = 0; k < k_sq; ++k)
		{
			const int h_in = h_in_start + k / kernel_size;
			const int w_in = w_in_start + k % kernel_size;
			if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
			{
				const int spatial_idx_in = h_in * Dimension_in + w_in;
				const int spatial_groups = spatial_idx_in * groups_per_in;
				const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
				const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups + k * groups_per_in;
				#pragma unroll 4
				for (int g = 0; g < groups_per_in; ++g)
				{
					uint4 curr_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
					uint4 curr_w = w_v4[g];
					const int8_t curr_in_scale = input_scales[spatial_groups + g];
					const int8_t curr_w_scale = weight_scales[outc_groups + k * groups_per_in + g];
					const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
					const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
					int partial_sum = 0;
					#pragma unroll
					for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
					fwd_acc_arr[k & 3] = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc_arr[k & 3]);
				}
			}
		}
		const float fwd_acc = fwd_acc_arr[0] + fwd_acc_arr[1] + fwd_acc_arr[2] + fwd_acc_arr[3];
		float local_sq = fwd_acc * fwd_acc;
		const float dy = grad_output[p * OutC + outc];
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
		if (lane_id == 0) smem_warps[warp_id] = local_sq;
		__syncthreads();
		if (warp_id == 0)
		{
			float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
			#pragma unroll
			for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
			if (lane_id == 0) smem_warps[0] = block_sq;
		}
		__syncthreads();
		const float inv_rms = __frsqrt_rn(fmaf(smem_warps[0], min_inv_chunk_out, epsilon));
		float dy_normed = 0.0f;
		if constexpr (!disable_activation) dy_normed = silu_grad(fmaf(fwd_acc, inv_rms, bias), dy) * inv_rms;
		else dy_normed = dy * inv_rms;
		#pragma unroll
		for (int k = 0; k < k_sq; ++k)
		{
			const int h_in = h_in_start + k / kernel_size;
			const int w_in = w_in_start + k % kernel_size;
			if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
			{
				const int spatial_idx_in = h_in * Dimension_in + w_in;
				sum_grad_w[k] = fmaf(dequantize_scalar(input[spatial_idx_in * u32_stride_in + (blockIdx.y >> 3)], blockIdx.y & 7, input_scales[spatial_idx_in * groups_per_in + (blockIdx.y >> 5)]), dy_normed, sum_grad_w[k]);
			}
		}
	}
	#pragma unroll
	for (int k = 0; k < k_sq; ++k) grad_weights[(outc * k_sq + k) * InC + blockIdx.y] = sum_grad_w[k];
}
template <bool disable_activation = true, int kernel_size = 4, int channels_per_block = gpu_block_threads>
__global__ void convDenseBwdGradBiasKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, const float* __restrict__ grad_output, float* __restrict__ grad_biases, int InC, int OutC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int k_sq = kernel_size * kernel_size, warps = channels_per_block >> 5;
	const int outc = blockIdx.x * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (outc >= OutC) return;
	const int padding = kernel_size == stride ? 0 : (kernel_size - 1) >> 1;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int NHW_out = Dimension_out * Dimension_out;
	const int groups_per_in = InC >> 5;
	const int u32_stride_in = InC >> 3;
	const int outc_groups = outc * k_sq * groups_per_in;
	[[maybe_unused]] const float bias = biases[outc];
	float bias_grad_accumulator = 0.0f;
	const float min_inv_chunk_out = __frcp_rn(static_cast<float>(min(OutC, channels_per_block)));
	for (int p = 0; p < NHW_out; ++p)
	{
		const int h_in_start = p / Dimension_out * stride - padding;
		const int w_in_start = (p & (Dimension_out - 1)) * stride - padding;
		float fwd_acc_arr[4] = {};
		#pragma unroll
		for (int k = 0; k < k_sq; ++k)
		{
			const int h_in = h_in_start + k / kernel_size;
			const int w_in = w_in_start + k % kernel_size;
			if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
			{
				const int spatial_idx_in = h_in * Dimension_in + w_in;
				const int spatial_groups = spatial_idx_in * groups_per_in;
				const uint4* __restrict__ input_v4_base = reinterpret_cast<const uint4*>(input + spatial_idx_in * u32_stride_in);
				const uint4* __restrict__ w_v4 = reinterpret_cast<const uint4*>(weights) + outc_groups + k * groups_per_in;
				#pragma unroll 4
				for (int g = 0; g < groups_per_in; ++g)
				{
					uint4 curr_in_v4 = lane_id == 0 ? input_v4_base[g] : make_uint4(0, 0, 0, 0);
					uint4 curr_w = w_v4[g];
					const int8_t curr_in_scale = input_scales[spatial_groups + g];
					const int8_t curr_w_scale = weight_scales[outc_groups + k * groups_per_in + g];
					const auto& w_val_ptr = reinterpret_cast<const uint32_t(&)[4]>(curr_w);
					const auto& shared_in = reinterpret_cast<const uint32_t(&)[4]>(curr_in_v4);
					int partial_sum = 0;
					#pragma unroll
					for (int i = 0; i < 4; ++i) partial_sum += dot_int4(__shfl_sync(0xFFFFFFFF, shared_in[i], 0), w_val_ptr[i]);
					fwd_acc_arr[k & 3] = fmaf(dequantize_scale(curr_in_scale), dequantize_scale(curr_w_scale) * partial_sum, fwd_acc_arr[k & 3]);
				}
			}
		}
		const float fwd_acc = fwd_acc_arr[0] + fwd_acc_arr[1] + fwd_acc_arr[2] + fwd_acc_arr[3];
		float local_sq = fwd_acc * fwd_acc;
		const float dy = grad_output[p * OutC + outc];
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
		if (lane_id == 0) smem_warps[warp_id] = local_sq;
		__syncthreads();
		if (warp_id == 0)
		{
			float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
			#pragma unroll
			for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
			if (lane_id == 0) smem_warps[0] = block_sq;
		}
		__syncthreads();
		const float inv_rms = __frsqrt_rn(fmaf(smem_warps[0], min_inv_chunk_out, epsilon));
		if constexpr (!disable_activation) bias_grad_accumulator += silu_grad(fmaf(fwd_acc, inv_rms, bias), dy);
		else bias_grad_accumulator += dy;
	}
	grad_biases[outc] = bias_grad_accumulator;
}
template <int kernel_size = 7, int threads_per_block = gpu_block_threads>
__global__ void convDepthwiseFwdKernel(const uint32_t* __restrict__ input, const uint32_t* __restrict__ weights, const int8_t* __restrict__ input_scales, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int InC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int warps = threads_per_block >> 5, padding = kernel_size >> 1, k_sq = kernel_size * kernel_size;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int group_idx = blockIdx.y * warps + warp_id;
	const int outc = (group_idx << 5) + lane_id;
	if (outc >= InC) return;
	float acc = biases[outc];
	const int h_in_start = blockIdx.x / Dimension_out * stride - padding;
	const int w_in_start = (blockIdx.x & (Dimension_out - 1)) * stride - padding;
	const int u32_stride_in = InC >> 3, groups_per_in = InC >> 5;
	const int u32_off = lane_id >> 3, shift = lane_id & 7;
	float acc_arr[4] = {};
	#pragma unroll
	for (int k = 0; k < k_sq; ++k)
	{
		const int cur_h = h_in_start + k / kernel_size;
		const int cur_w = w_in_start + k % kernel_size;
		const bool is_valid = cur_h >= 0 && cur_h < Dimension_in && cur_w >= 0 && cur_w < Dimension_in;
		const int spatial_idx_in = (is_valid ? cur_h : 0) * Dimension_in + (is_valid ? cur_w : 0);
		const uint32_t input_packed = is_valid ? input[spatial_idx_in * u32_stride_in + (group_idx << 2) + u32_off] : 0;
		const int8_t curr_in_scale = is_valid ? input_scales[spatial_idx_in * groups_per_in + group_idx] : 0;
		const int curr_absolute_idx = k * InC + outc;
		const uint32_t curr_w_packed = weights[curr_absolute_idx >> 3];
		const int8_t curr_w_scale = weight_scales[curr_absolute_idx >> 5];
		acc_arr[k & 3] = fmaf(dequantize_scalar(input_packed, shift, curr_in_scale), dequantize_scalar(curr_w_packed, curr_absolute_idx & 7, curr_w_scale), acc_arr[k & 3]);
	}
	acc += acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];
	float group_max = fabsf(acc);
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) group_max = fmaxf(group_max, __shfl_xor_sync(0xFFFFFFFF, group_max, offset));
	const float out_scale = group_max < epsilon ? 1.0f : group_max;
	uint32_t val = static_cast<uint32_t>(quantize_scalar_to_int4(acc, quant_int4_max * __frcp_rn(out_scale))) << (shift << 2);
	#pragma unroll
	for (int i = 1; i < 8; i <<= 1) val |= __shfl_down_sync(0xFFFFFFFF, val, i);
	if (shift == 0) output[blockIdx.x * u32_stride_in + (group_idx << 2) + (lane_id >> 3)] = val;
	if (lane_id == 0) output_scales[blockIdx.x * groups_per_in + group_idx] = quantize_scale(out_scale);
}
template <int kernel_size = 7>
__global__ void convDepthwiseBwdGradInputKernel(const float* __restrict__ grad_output, const float* __restrict__ master_weights, float* __restrict__ grad_input, int InC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int padding = kernel_size >> 1, k_sq = kernel_size * kernel_size;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int warps = blockDim.x >> 5;
	const int group_idx = blockIdx.y * warps + warp_id;
	const int inc = (group_idx << 5) + lane_id;
	if (inc >= InC) return;
	const int h_in = blockIdx.x / Dimension_in;
	const int w_in = blockIdx.x & (Dimension_in - 1);
	float acc = 0.0f, curr_go = 0.0f, curr_mw = master_weights[inc], next_go, next_mw;
	const int h_out_t = h_in + padding;
	const int w_out_t = w_in + padding;
	if (h_out_t % stride == 0 && w_out_t % stride == 0)
	{
		const int ho = h_out_t / stride;
		const int wo = w_out_t / stride;
		if (ho >= 0 && ho < Dimension_out && wo >= 0 && wo < Dimension_out) curr_go = grad_output[(ho * Dimension_out + wo) * InC + inc];
	}
	for (int k = 1; k < k_sq; ++k)
	{
		const int next_h_out_t = h_in - k / kernel_size + padding;
		const int next_w_out_t = w_in - k % kernel_size + padding;
		next_mw = master_weights[k * InC + inc];
		next_go = 0.0f;
		if (next_h_out_t % stride == 0 && next_w_out_t % stride == 0)
		{
			const int ho = next_h_out_t / stride;
			const int wo = next_w_out_t / stride;
			if (ho >= 0 && ho < Dimension_out && wo >= 0 && wo < Dimension_out) next_go = grad_output[(ho * Dimension_out + wo) * InC + inc];
		}
		acc = fmaf(curr_go, curr_mw, acc);
		curr_go = next_go;
		curr_mw = next_mw;
	}
	acc = fmaf(curr_go, curr_mw, acc);
	grad_input[blockIdx.x * InC + inc] = acc;
}
template <int kernel_size = 7, int threads_per_block = gpu_block_threads, int chunk_size = 8>
__global__ void convDepthwiseBwdGradWeightKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const float* __restrict__ grad_output, float* __restrict__ grad_weights, int InC, int Dimension_in, int Dimension_out, int stride)
{
	constexpr int warps = threads_per_block >> 5, padding = kernel_size >> 1, k_sq = kernel_size * kernel_size;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int group_idx = blockIdx.x * warps + warp_id;
	const int inc = (group_idx << 5) + lane_id;
	const int log2_dim_out = __ffs(Dimension_out) - 1;
	const int mask_dim_out = Dimension_out - 1;
	const int u32_stride_in = InC >> 3;
	const int groups_per_in = InC >> 5;
	const int in_idx_u32 = inc >> 3;
	const int shift = inc & 7;
	const int NHW_out = Dimension_out * Dimension_out;
	for (int chunk_start = 0; chunk_start < k_sq; chunk_start += chunk_size)
	{
		float register_acc[chunk_size] = {};
		for (int p = 0; p < NHW_out; ++p)
		{
			const float go = grad_output[p * InC + inc];
			const int h_in_base = (p >> log2_dim_out) * stride - padding;
			const int w_in_base = (p & mask_dim_out) * stride - padding;
			#pragma unroll
			for (int c = 0; c < chunk_size; ++c)
			{
				const int k = chunk_start + c;
				if (k < k_sq)
				{
					const int h_in = h_in_base + k / kernel_size;
					const int w_in = w_in_base + k % kernel_size;
					if (h_in >= 0 && h_in < Dimension_in && w_in >= 0 && w_in < Dimension_in)
					{
						const int spatial_idx_in = h_in * Dimension_in + w_in;
						const int8_t s_quant = input_scales[spatial_idx_in * groups_per_in + group_idx];
						const uint32_t input_packed = input[spatial_idx_in * u32_stride_in + in_idx_u32];
						register_acc[c] = fmaf(dequantize_scalar(input_packed, shift, s_quant), go, register_acc[c]);
					}
				}
			}
		}
		#pragma unroll
		for (int c = 0; c < chunk_size; ++c)
		{
			const int k = chunk_start + c;
			if (k < k_sq) grad_weights[k * InC + inc] = register_acc[c];
		}
	}
}
template <int dummy = 0>
__global__ void convDepthwiseBwdGradBiasKernel(const float* __restrict__ grad_output, float* __restrict__ grad_biases, int InC, int NHW)
{
	const int lane_id = threadIdx.x & 31;
	const int group_idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	const int outc = (group_idx << 5) + lane_id;
	float sum_arr[4] = {}, curr_go[4];
	#pragma unroll
	for (int p = 0; p < 4; ++p) curr_go[p] = p < NHW ? grad_output[p * InC + outc] : 0.0f;
	float next_go[4];
	for (int p = 4; p < NHW; p += 4)
	{
		#pragma unroll
		for (int i = 0; i < 4; ++i) next_go[i] = grad_output[(p + i) * InC + outc];
		#pragma unroll
		for (int i = 0; i < 4; ++i) sum_arr[i] += curr_go[i];
		#pragma unroll
		for (int i = 0; i < 4; ++i) curr_go[i] = next_go[i];
	}
	#pragma unroll
	for (int i = 0; i < 4; ++i) sum_arr[i] += curr_go[i];
	const float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
	const float neighbor_sum = __shfl_xor_sync(0xFFFFFFFF, sum, 1);
	if ((lane_id & 1) == 0)
	{
		grad_biases[outc] = sum;
		grad_biases[outc + 1] = neighbor_sum;
	}
}
template <int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 4) residualAddFwdKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ residual, const int8_t* __restrict__ residual_scales, uint32_t* __restrict__ output, int8_t* __restrict__ output_scales, int channels, int dimension, float survival_prob_inv, bool is_skipped, float residual_factor)
{
	constexpr int warps = threads_per_block >> 5;
	const int s_idx = blockIdx.y * dimension + blockIdx.x;
	__shared__ float smem_warps[warps];
	extern __shared__ float pool_smem[];
	float* smem_results = pool_smem;
	const int lane_id = threadIdx.x & 31;
	const int num_u32 = channels >> 3, num_groups = channels >> 5;
	const uint4* __restrict__ input_v4_ptr = reinterpret_cast<const uint4*>(input + s_idx * num_u32);
	const uint4* __restrict__ residual_v4_ptr = reinterpret_cast<const uint4*>(residual + s_idx * num_u32);
	const float in_scale_mult = residual_factor * survival_prob_inv;
	float l_max_arr[4] = {};
	for (int v_idx = threadIdx.x; v_idx < num_groups; v_idx += threads_per_block)
	{
		const float s_res = dequantize_scale(residual_scales[s_idx * num_groups + v_idx]);
		const float s_in = is_skipped ? 0.0f : dequantize_scale(input_scales[s_idx * num_groups + v_idx]) * in_scale_mult;
		const uint4 val_v4 = is_skipped ? make_uint4(0, 0, 0, 0) : input_v4_ptr[v_idx];
		const uint4 res_v4 = residual_v4_ptr[v_idx];
		const auto& val_v4_ptr = reinterpret_cast<const uint32_t(&)[4]>(val_v4);
		const auto& res_v4_ptr = reinterpret_cast<const uint32_t(&)[4]>(res_v4);
		const int base_c_idx = v_idx << 5;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
			const int sub_c_idx = base_c_idx + (i << 3);
			#pragma unroll
			for (int pair_idx = 0; pair_idx < 4; ++pair_idx)
			{
				const float2 pair_in = extract_ints4(val_v4_ptr[i], pair_idx);
				const float2 pair_res = extract_ints4(res_v4_ptr[i], pair_idx);
				const float res0 = fmaf(pair_in.x, s_in, pair_res.x * s_res);
				const float res1 = fmaf(pair_in.y, s_in, pair_res.y * s_res);
				const int write_idx = sub_c_idx + (pair_idx << 1);
				smem_results[write_idx] = res0;
				smem_results[write_idx + 1] = res1;
				l_max_arr[pair_idx] = fmaxf(l_max_arr[pair_idx], fmaxf(fabsf(res0), fabsf(res1)));
			}
		}
	}
	float l_max = fmaxf(fmaxf(l_max_arr[0], l_max_arr[1]), fmaxf(l_max_arr[2], l_max_arr[3]));
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFF, l_max, offset));
	if (lane_id == 0) smem_warps[threadIdx.x >> 5] = l_max;
	__syncthreads();
	if (threadIdx.x < 32)
	{
		float val = threadIdx.x < warps ? smem_warps[threadIdx.x] : 0.0f;
		#pragma unroll
		for (int offset = 16; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
		if (threadIdx.x == 0) smem_warps[0] = val;
	}
	__syncthreads();
	const float out_scale = smem_warps[0] < epsilon ? 1.0f : smem_warps[0];
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
template <int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 4) residualAddBwdKernel(float* __restrict__ grad_next, float* __restrict__ grad_res, int channels, int dimension, float survival_prob_inv, float residual_factor)
{
	constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(float);
	const float combined_weight = survival_prob_inv * residual_factor;
	const int offset = (blockIdx.y * dimension + blockIdx.x) * channels;
	uint4* __restrict__ gn_ptr_v4 = reinterpret_cast<uint4*>(grad_next + offset);
	uint4* __restrict__ gr_ptr_v4 = reinterpret_cast<uint4*>(grad_res + offset);
	const int size_v4 = channels / elements_per_uint4;
	#pragma unroll 2
	for (int v_idx = threadIdx.x; v_idx < size_v4; v_idx += threads_per_block)
	{
		float local_gn[elements_per_uint4], local_gr[elements_per_uint4];
		*reinterpret_cast<uint4*>(local_gn) = gn_ptr_v4[v_idx];
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; ++i)
		{
			local_gr[i] = local_gn[i];
			local_gn[i] = local_gn[i] * combined_weight;
		}
		gr_ptr_v4[v_idx] = *reinterpret_cast<uint4*>(local_gr);
		gn_ptr_v4[v_idx] = *reinterpret_cast<uint4*>(local_gn);
	}
}
template <bool update_state = true, int threads_per_block = gpu_block_threads>
__global__ void __launch_bounds__(threads_per_block, 2) mambaSSMFwdStateKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, float2* __restrict__ h_states, float2* __restrict__ trace_A, float2* __restrict__ trace_B, float2* __restrict__ trace_Delta, float* __restrict__ y_ssm_out, int InC, int d_inner)
{
	constexpr int num_warps = threads_per_block >> 5, mamba_cols = mamba_d_state >> 5;
	const int off_W_C_re = d_inner + (d_inner << 5),
			  off_W_C_im = off_W_C_re + d_inner * mamba_d_state,
			  off_A_re = off_W_C_im + d_inner * mamba_d_state,
			  off_A_im = off_A_re + d_inner * mamba_d_state,
			  off_D = off_A_im + d_inner * mamba_d_state;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int block_ch_idx = (blockIdx.x * num_warps + warp_id) << 5;
	const int ch = block_ch_idx + lane_id;
	const int in_ch_idx = ch & (InC - 1);
	float2 h_l[mamba_cols];
	[[maybe_unused]] float2 trA[mamba_cols], trB[mamba_cols], trDelta[mamba_cols];
	const int base_row_tr = ch * mamba_d_state, base_row_tr_lane_id = base_row_tr + lane_id;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_row_tr_lane_id + (i << 5);
		h_l[i] = h_states[idx];
		if constexpr (update_state)
		{
			trA[i] = trace_A[idx];
			trB[i] = trace_B[idx];
			trDelta[i] = trace_Delta[idx];
		}
	}
	float2 cached_raw_A[mamba_cols], cached_W_C[mamba_cols];
	uint32_t raw_weights_ac[4];
	int8_t raw_weight_scales_ac[4];
	#pragma unroll
	for (int c = 0; c < mamba_cols; ++c)
	{
		const int w_off = base_row_tr_lane_id + (c << 5);
		const int off_a_r = off_A_re + w_off, off_a_i = off_A_im + w_off, off_c_r = off_W_C_re + w_off, off_c_i = off_W_C_im + w_off;
		raw_weights_ac[0] = weights[off_a_r >> 3]; raw_weight_scales_ac[0] = weight_scales[off_a_r >> 5];
		raw_weights_ac[1] = weights[off_a_i >> 3]; raw_weight_scales_ac[1] = weight_scales[off_a_i >> 5];
		raw_weights_ac[2] = weights[off_c_r >> 3]; raw_weight_scales_ac[2] = weight_scales[off_c_r >> 5];
		raw_weights_ac[3] = weights[off_c_i >> 3]; raw_weight_scales_ac[3] = weight_scales[off_c_i >> 5];
		cached_raw_A[c] = make_float2(dequantize_scalar(raw_weights_ac[0], off_a_r & 7, raw_weight_scales_ac[0]), dequantize_scalar(raw_weights_ac[1], off_a_i & 7, raw_weight_scales_ac[1]));
		cached_W_C[c] = make_float2(dequantize_scalar(raw_weights_ac[2], off_c_r & 7, raw_weight_scales_ac[2]), dequantize_scalar(raw_weights_ac[3], off_c_i & 7, raw_weight_scales_ac[3]));
	}
	const int8_t input_scale = input_scales[in_ch_idx >> 5];
	const int mimo_block_id = block_ch_idx / mamba_mimo_group_size;
	const int8_t B_single_scale = weight_scales[(d_inner + mimo_block_id * mamba_mimo_matrix_elements + lane_id * mamba_mimo_group_size) >> 5];
	uint32_t B_row_packed[4];
	*reinterpret_cast<uint4*>(B_row_packed) = *reinterpret_cast<const uint4*>(&weights[(d_inner >> 3) + mimo_block_id * (mamba_mimo_matrix_elements >> 3) + (lane_id << 2)]);
	const uint32_t weight_u32 = weights[ch >> 3];
	const int8_t weight_scale = weight_scales[ch >> 5];
	const uint4 x_v4 = lane_id == 0 ? *reinterpret_cast<const uint4*>(&input[in_ch_idx >> 3]) : make_uint4(0, 0, 0, 0);
	const auto& x_v4_ptr = reinterpret_cast<const uint32_t(&)[4]>(x_v4);
	uint32_t x_packed[4];
	int dot_mimo = 0;
	#pragma unroll
	for (int i = 0; i < 4; ++i) x_packed[i] = __shfl_sync(0xFFFFFFFF, x_v4_ptr[i], 0);
	#pragma unroll
	for (int i = 0; i < 4; ++i) dot_mimo += dot_int4(x_packed[i], B_row_packed[i]);
	const float x_local = dequantize_scalar(x_packed[(lane_id >> 3) & 3], in_ch_idx & 7, input_scale);
	const float B_x_mimo = dot_mimo * (dequantize_scale(input_scale) * dequantize_scale(B_single_scale));
	const float xlwdl = x_local * dequantize_scalar(weight_u32, ch & 7, weight_scale);
	const float dt = xlwdl + __logf(1.0f + __expf(-fabsf(xlwdl)));
	const float dt_half = dt * 0.5f;
	const int off_D_ch = off_D + ch;
	const uint32_t weight_u32_d = weights[off_D_ch >> 3];
	const int8_t weight_scale_d = weight_scales[off_D_ch >> 5];
	float y_acc_arr[mamba_cols];
	#pragma unroll
	for (int c = 0; c < mamba_cols; ++c)
	{
		const float are = -__expf(cached_raw_A[c].x);
		const float dim = -dt_half * cached_raw_A[c].y;
		const float dre = fmaf(-dt_half, are, 1.0f);
		const float inv_s = __frcp_rn(fmaf(dre, dre, fmaf(dim, dim, epsilon)));
		const float ire = dre * inv_s;
		const float iim = -dim * inv_s;
		const float nre = fmaf(dt_half, are, 1.0f);
		const float abr = fmaf(nre, ire, dim * iim);
		const float abi = fmaf(nre, iim, -dim * ire);
		if constexpr (update_state)
		{
			const float dt_B_x = dt * B_x_mimo;
			const float hr_old = h_l[c].x, hi_old = h_l[c].y;
			h_l[c] = make_float2(fmaf(abr, hr_old, fmaf(-abi, hi_old, dt_B_x * ire)), fmaf(abr, hi_old, fmaf(abi, hr_old, dt_B_x * iim)));
			const float i2_re = fmaf(ire, ire, -iim * iim);
			const float i2_im = 2.0f * ire * iim;
			const float za_re = fmaf(dt_half, B_x_mimo, hr_old);
			const float dt_i2_za_re = dt * fmaf(i2_re, za_re, -i2_im * hi_old);
			const float dt_i2_za_im = dt * fmaf(i2_re, hi_old, i2_im * za_re);
			const float zdt_re = fmaf(are, hr_old, fmaf(-cached_raw_A[c].y, hi_old, B_x_mimo));
			const float zdt_im = fmaf(are, hi_old, cached_raw_A[c].y * hr_old);
			const float d_dt_re = fmaf(i2_re, zdt_re, -i2_im * zdt_im);
			const float d_dt_im = fmaf(i2_re, zdt_im, i2_im * zdt_re);
			trA[c] = make_float2(fmaf(abr, trA[c].x, fmaf(-abi, trA[c].y, dt_i2_za_re)), fmaf(abr, trA[c].y, fmaf(abi, trA[c].x, dt_i2_za_im)));
			trB[c] = make_float2(fmaf(abr, trB[c].x, fmaf(-abi, trB[c].y, dt * ire)), fmaf(abr, trB[c].y, fmaf(abi, trB[c].x, dt * iim)));
			trDelta[c] = make_float2(fmaf(abr, trDelta[c].x, fmaf(-abi, trDelta[c].y, d_dt_re)), fmaf(abr, trDelta[c].y, fmaf(abi, trDelta[c].x, d_dt_im)));
		}
		y_acc_arr[c] = fmaf(h_l[c].x, cached_W_C[c].x, -h_l[c].y * cached_W_C[c].y);
	}
	float y_acc = 0.0f;
	#pragma unroll
	for (int c = 0; c < mamba_cols; ++c) y_acc += y_acc_arr[c];
	y_ssm_out[ch] = fmaf(dequantize_scalar(weight_u32_d, off_D_ch & 7, weight_scale_d), x_local, y_acc);
	if constexpr (update_state)
	{
		#pragma unroll
		for (int i = 0; i < mamba_cols; ++i)
		{
			const int idx = base_row_tr_lane_id + (i << 5);
			h_states[idx] = h_l[i];
			trace_A[idx] = trA[i];
			trace_B[idx] = trB[i];
			trace_Delta[idx] = trDelta[i];
		}
	}
}
template <int activation_type = 1, int channels_per_block = gpu_block_threads>
__global__ void __launch_bounds__(channels_per_block, 4) mambaSSMFwdOutProjKernel(const float* __restrict__ y_ssm, const uint32_t* __restrict__ weights, const int8_t* __restrict__ weight_scales, const float* __restrict__ biases, float* __restrict__ output, int OutC, int d_inner)
{
	constexpr int warps = channels_per_block >> 5;
	const int off_W_out = d_inner + (d_inner << 5) + 4 * d_inner * mamba_d_state + d_inner;
	const int oc = blockIdx.x * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int w_base = off_W_out + oc * d_inner;
	const float4* __restrict__ y_v4 = reinterpret_cast<const float4*>(y_ssm);
	float acc = 0.0f;
	for (int ic_base = 0; ic_base < d_inner; ic_base += 128)
	{
		const int y_idx = (ic_base >> 2) + lane_id;
		const float4 y_reg = ic_base + (lane_id << 2) < d_inner ? y_v4[y_idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		const auto& y_arr = reinterpret_cast<const float(&)[4]>(y_reg);
		float y_arr_local[8] = {};
		#pragma unroll
		for (int g = 0; g < 4; ++g)
		{
			const int curr_ic = ic_base + (g << 5);
			if (curr_ic >= d_inner) break;
			uint4 w_curr = make_uint4(0, 0, 0, 0);
			int8_t w_scale = 0;
			if (oc < OutC)
			{
				const int w_offset = w_base + curr_ic;
				w_curr = *reinterpret_cast<const uint4*>(weights + (w_offset >> 3));
				w_scale = weight_scales[w_offset >> 5];
			}
			const auto& w_packed = reinterpret_cast<const uint32_t(&)[4]>(w_curr);
			float grp_sum[4] = {};
			const int src_lane_base = g << 3;
			#pragma unroll
			for (int w = 0; w < 4; ++w)
			{
				const uint32_t wp = w_packed[w];
				const int src_lane = src_lane_base + (w << 1);
				#pragma unroll
				for (int i = 0; i < 4; ++i)
				{
					y_arr_local[i] = __shfl_sync(0xFFFFFFFF, y_arr[i], src_lane);
					y_arr_local[i + 4] = __shfl_sync(0xFFFFFFFF, y_arr[i], src_lane + 1);
				}
				#pragma unroll
				for (int i = 0; i < 8; ++i) grp_sum[i & 3] = fmaf(y_arr_local[i], static_cast<float>(extract_int4(wp, i)), grp_sum[i & 3]);
			}
			acc = fmaf(grp_sum[0] + grp_sum[1] + grp_sum[2] + grp_sum[3], dequantize_scale(w_scale), acc);
		}
	}
	if (oc >= OutC) return;
	float local_sq = acc * acc;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	const float inv_rms = OutC < 2 ? 1.0f : __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon));
	const float val = fmaf(acc, inv_rms, biases[oc]);
	if constexpr (activation_type == 0) output[oc] = val;
    else if constexpr (activation_type == 1) output[oc] = silu(val);
    else output[oc] = sigmoid(val);
}
template <int threads_per_block = gpu_block_threads>
__global__ void mambaSSMBwdOutProjKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const float2* __restrict__ h_states, const float* __restrict__ master_weights, float* __restrict__ raw_out_buf, int InC, int OutC, int d_inner)
{
	constexpr int num_warps = threads_per_block >> 5, mamba_cols = mamba_d_state >> 5;
	const int off_W_C_re = d_inner + (d_inner << 5),
			  off_W_C_im = off_W_C_re + d_inner * mamba_d_state,
			  off_A_re = off_W_C_im + d_inner * mamba_d_state,
			  off_A_im = off_A_re + d_inner * mamba_d_state,
			  off_D = off_A_im + d_inner * mamba_d_state,
			  off_W_out = off_D + d_inner;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int block_ch_idx = (blockIdx.y * num_warps + warp_id) << 5;
	const int ch_idx = block_ch_idx + lane_id;
	const int in_ch_idx = ch_idx & (InC - 1);
	__shared__ float s_y_ssm[num_warps][33];
	float y_ssm_ch = master_weights[off_D + ch_idx] * dequantize_scalar(input[in_ch_idx >> 3], in_ch_idx & 7, input_scales[in_ch_idx >> 5]);
	const int base_tr_lane_id = ch_idx * mamba_d_state + lane_id;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_tr_lane_id + (i << 5);
		const float2 h_curr = h_states[idx];
		const float w_c_re = master_weights[off_W_C_re + idx];
		const float w_c_im = master_weights[off_W_C_im + idx];
		y_ssm_ch = fmaf(h_curr.x, w_c_re, y_ssm_ch);
		y_ssm_ch = fmaf(-h_curr.y, w_c_im, y_ssm_ch);
	}
	s_y_ssm[warp_id][lane_id] = y_ssm_ch;
	const int oc = (blockIdx.x << 5) + (warp_id << 2) + (lane_id >> 3);
	const int lane_id_v8 = lane_id & 7;
	const int block_ic_base = blockIdx.y * threads_per_block;
	const int w_idx_base = off_W_out + oc * d_inner + block_ic_base + (lane_id_v8 << 5);
	float sum_out = 0.0f;
	__syncthreads();
	#pragma unroll
	for (int i = 0; i < 32; ++i) sum_out = fmaf(s_y_ssm[lane_id_v8][i], master_weights[w_idx_base + i], sum_out);
	#pragma unroll
	for (int offset = 4; offset > 0; offset >>= 1) sum_out += __shfl_sync(0xFFFFFFFF, sum_out, offset);
	if (lane_id_v8 == 0) atomicAdd(&raw_out_buf[oc], sum_out);
}
template <int activation_type = 1, int channels_per_block = gpu_block_threads>
__global__ void mambaSSMBwdNormActKernel(float* __restrict__ raw_out_to_g_normed, const float* __restrict__ biases, const float* __restrict__ grad_output, float* __restrict__ grad_biases, int OutC)
{
	constexpr int warps = channels_per_block >> 5;
	const int oc = blockIdx.x * channels_per_block + threadIdx.x;
	__shared__ float smem_warps[warps];
	if (threadIdx.x < warps) smem_warps[threadIdx.x] = 0.0f;
	if (oc >= OutC) return;
	const float raw_val = raw_out_to_g_normed[oc];
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	float local_sq = raw_val * raw_val;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1) local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
	if (lane_id == 0) smem_warps[warp_id] = local_sq;
	__syncthreads();
	if (warp_id == 0)
	{
		float block_sq = lane_id < warps ? smem_warps[lane_id] : 0.0f;
		#pragma unroll
		for (int offset = warps >> 1; offset > 0; offset >>= 1) block_sq += __shfl_xor_sync(0xFFFFFFFF, block_sq, offset);
		if (lane_id == 0) smem_warps[0] = block_sq;
	}
	__syncthreads();
	const float inv_rms = OutC < 2 ? 1.0f : __frsqrt_rn(fmaf(smem_warps[0], __frcp_rn(static_cast<float>(min(OutC, channels_per_block))), epsilon));
	const float u = fmaf(raw_val, inv_rms, biases[oc]);
	const float dy = grad_output[oc];
	float g_act = 0.0f;
	if constexpr (activation_type == 0) g_act = dy;
    else if constexpr (activation_type == 1) g_act = silu_grad(u, dy);
    else g_act = sigmoid_grad(u, dy);
	grad_biases[oc] = g_act;
	raw_out_to_g_normed[oc] = g_act * inv_rms;
}
template <int threads_per_block = gpu_block_threads>
__global__ void mambaSSMApplyTracesKernel(const uint32_t* __restrict__ input, const int8_t* __restrict__ input_scales, const float* __restrict__ g_normed_buf, const float2* __restrict__ h_states, const float2* __restrict__ trace_A, const float2* __restrict__ trace_B, const float2* __restrict__ trace_Delta, const float* __restrict__ master_weights, float* __restrict__ grad_weights, float* __restrict__ grad_input, int InC, int OutC, int d_inner)
{
	constexpr int num_warps = threads_per_block >> 5, mamba_cols = mamba_d_state >> 5;
	const int off_W_C_re = d_inner + (d_inner << 5),
			  off_W_C_im = off_W_C_re + d_inner * mamba_d_state,
			  off_A_re = off_W_C_im + d_inner * mamba_d_state,
			  off_A_im = off_A_re + d_inner * mamba_d_state,
			  off_D = off_A_im + d_inner * mamba_d_state,
			  off_W_out = off_D + d_inner;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = threadIdx.x >> 5;
	const int block_ch_idx = (blockIdx.x * num_warps + warp_id) << 5;
	const int ch_idx = block_ch_idx + lane_id;
	const float D_skip = master_weights[off_D + ch_idx];
	const float W_delta = master_weights[ch_idx];
	const int in_ch_idx = ch_idx & (InC - 1);
	const uint32_t raw_input_xt = input[in_ch_idx >> 3];
	const int8_t raw_input_scales_xt = input_scales[in_ch_idx >> 5];
	const int base_tr_lane_id = ch_idx * mamba_d_state + lane_id;
	const float x_t = dequantize_scalar(raw_input_xt, in_ch_idx & 7, raw_input_scales_xt);
	float y_ssm_ch = D_skip * x_t;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_tr_lane_id + (i << 5);
		const float2 h_curr = h_states[idx];
		const float w_c_re = master_weights[off_W_C_re + idx];
		const float w_c_im = master_weights[off_W_C_im + idx];
		y_ssm_ch = fmaf(h_curr.x, w_c_re, y_ssm_ch);
		y_ssm_ch = fmaf(-h_curr.y, w_c_im, y_ssm_ch);
	}
	float grad_y_ssm = 0.0f;
	for (int oc = 0; oc < OutC; ++oc)
	{
		const float g_normed = g_normed_buf[oc];
		const int w_out_idx = off_W_out + oc * d_inner + ch_idx;
		const float w_out = master_weights[w_out_idx];
		grad_weights[w_out_idx] = g_normed * y_ssm_ch;
		grad_y_ssm = fmaf(g_normed, w_out, grad_y_ssm);
	}
	const int mimo_group_base = d_inner + block_ch_idx / mamba_mimo_group_size * mamba_mimo_matrix_elements;
	const int bre_base_id = mimo_group_base + lane_id * mamba_mimo_group_size;
	float g_dt_total = 0.0f, g_B_mimo_total = 0.0f;
	#pragma unroll
	for (int i = 0; i < mamba_cols; ++i)
	{
		const int idx = base_tr_lane_id + (i << 5);
		const float wc_re_raw = master_weights[off_W_C_re + idx];
		const float wc_im_raw = master_weights[off_W_C_im + idx];
		const float are_raw = master_weights[off_A_re + idx];
		const float2 h_curr = h_states[idx];
		const float2 tr_a = trace_A[idx];
		const float2 tr_b = trace_B[idx];
		const float2 tr_dt = trace_Delta[idx];
		const float gh_re = grad_y_ssm * wc_re_raw;
		const float gh_im = -grad_y_ssm * wc_im_raw;
		const float are = -__expf(are_raw);
		atomicAdd(&grad_weights[off_W_C_re + idx], grad_y_ssm * h_curr.x);
		atomicAdd(&grad_weights[off_W_C_im + idx], -grad_y_ssm * h_curr.y);
		atomicAdd(&grad_weights[off_A_re + idx], fmaf(gh_re, tr_a.x, gh_im * tr_a.y) * are);
		atomicAdd(&grad_weights[off_A_im + idx], fmaf(gh_im, tr_a.x, -gh_re * tr_a.y));
		g_B_mimo_total = fmaf(gh_re, tr_b.x, fmaf(gh_im, tr_b.y, g_B_mimo_total));
		g_dt_total = fmaf(gh_re, tr_dt.x, fmaf(gh_im, tr_dt.y, g_dt_total));
	}
	const int lane_in_mimo_g = mimo_group_base + (lane_id & (mamba_mimo_group_size - 1));
	const int group_base_lane = lane_id & ~(mamba_mimo_group_size - 1);
	const int lane_mimo_block_ch = block_ch_idx + group_base_lane;
	float g_x_B_arr[4] = {};
	#pragma unroll
	for (int k = 0; k < mamba_mimo_group_size; ++k)
	{
		const int src_in_ch = (lane_mimo_block_ch + k) & (InC - 1);
		const uint32_t raw_input_mimo = input[src_in_ch >> 3];
		const int8_t raw_input_scales_mimo = input_scales[src_in_ch >> 5];
		const float B_val_raw = master_weights[lane_in_mimo_g + (group_base_lane + k) * mamba_mimo_group_size];
		atomicAdd(&grad_weights[bre_base_id + k], g_B_mimo_total * dequantize_scalar(raw_input_mimo, src_in_ch & 7, raw_input_scales_mimo));
		g_x_B_arr[k & 3] = fmaf(__shfl_sync(0xFFFFFFFF, g_B_mimo_total, group_base_lane + k), B_val_raw, g_x_B_arr[k & 3]);
	}
	const float g_x_B = g_x_B_arr[0] + g_x_B_arr[1] + g_x_B_arr[2] + g_x_B_arr[3];
	const float g_pre_dt = g_dt_total * sigmoid(x_t * W_delta);
	atomicAdd(&grad_weights[ch_idx], g_pre_dt * x_t);
	atomicAdd(&grad_weights[off_D + ch_idx], grad_y_ssm * x_t);
	atomicAdd(&grad_input[in_ch_idx], fmaf(g_pre_dt, W_delta, fmaf(grad_y_ssm, D_skip, g_x_B)));
}
template <typename T>
__global__ void resizeKernel(const T* __restrict__ src, T* __restrict__ dst, int srcH, int srcW, int dstDim, int channels)
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (blockIdx.z >= channels || y >= dstDim || x >= dstDim) return;
	const float scale = static_cast<float>(srcW) / dstDim;
	const float y_scaled = fmaf(y + 0.5f, scale, -0.5f);
	const float fy = floorf(y_scaled);
	const int y1 = max(0, static_cast<int>(fy));
	const float wy = y_scaled - fy;
	const T* s_ptr = src + blockIdx.z * srcW * srcH;
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
	const float v11_0 = static_cast<float>(s_ptr[row1 + x1_0]);
	const float v11_1 = static_cast<float>(s_ptr[row1 + x1_1]);
	const float v21_0 = static_cast<float>(s_ptr[row2 + x1_0]);
	const float v21_1 = static_cast<float>(s_ptr[row2 + x1_1]);
	const float top0 = fmaf(wx0, static_cast<float>(s_ptr[row1 + x2_0]) - v11_0, v11_0);
	const float top1 = fmaf(wx1, static_cast<float>(s_ptr[row1 + x2_1]) - v11_1, v11_1);
	const float res0 = fmaf(wy, fmaf(wx0, static_cast<float>(s_ptr[row2 + x2_0]) - v21_0, v21_0) - top0, top0);
	const float res1 = fmaf(wy, fmaf(wx1, static_cast<float>(s_ptr[row2 + x2_1]) - v21_1, v21_1) - top1, top1);
	const int dst_idx_base = blockIdx.z * dstDim * dstDim + y * dstDim + x;
	if (x + 1 < dstDim)
	{
		dst[dst_idx_base] = static_cast<T>(res0);
		dst[dst_idx_base + 1] = static_cast<T>(res1);
	}
	else dst[dst_idx_base] = static_cast<T>(res0);
}
template <typename T, int tile_dim = 32>
__global__ void NCHWtoNHWCKernel(const T* __restrict__ src, T* __restrict__ dst, int dimension, int channels)
{
	__shared__ T tile[tile_dim][tile_dim + 1];
	const int spatial_size = dimension * dimension;
	int x = blockIdx.x * tile_dim + threadIdx.x;
	int y = blockIdx.y * tile_dim + threadIdx.y;
	if (x < spatial_size && y < channels) tile[threadIdx.y][threadIdx.x] = src[y * spatial_size + x];
	__syncthreads();
	x = blockIdx.y * tile_dim + threadIdx.x;
	y = blockIdx.x * tile_dim + threadIdx.y;
	if (x < channels && y < spatial_size) dst[y * channels + x] = tile[threadIdx.x][threadIdx.y];
}
template <typename T>
__global__ void setActionsKernel(const T* __restrict__ arr64, action_type* __restrict__ actions, int actionCount)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool is_set = false;
	if (idx < actionCount)
	{
		const float val = static_cast<float>(arr64[idx]);
		is_set = getFastRandomFloatBoundedUniform(generateUniqueSeed(static_cast<int>(val * fmaximum) ^ idx), 1.0f) < val;
	}
	const uint32_t warp_mask = __ballot_sync(__activemask(), is_set);
	if (idx < actionCount && (idx & 31) == 0) atomicOr(reinterpret_cast<unsigned long long*>(&actions[idx >> 6]), static_cast<unsigned long long>(warp_mask) << (idx & 32));
}
template <typename T>
__global__ void huberKernel(const T* __restrict__ pred, const T* __restrict__ real, float* __restrict__ out_loss, int num_elements, float delta)
{
	constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(T);
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int element_idx = idx << 3;
	float h[elements_per_uint4] = {};
	if (element_idx + 7 < num_elements)
	{
		const uint4 pred_v4 = reinterpret_cast<const uint4*>(pred)[idx];
		const uint4 real_v4 = reinterpret_cast<const uint4*>(real)[idx];
		const auto& p_h2 = reinterpret_cast<const T(&)[elements_per_uint4]>(pred_v4);
		const auto& r_h2 = reinterpret_cast<const T(&)[elements_per_uint4]>(real_v4);
		float diff[elements_per_uint4];
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; i++) diff[i] = static_cast<float>(p_h2[i]) - static_cast<float>(r_h2[i]);
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; i++)
		{
			const float abs_d = fabsf(diff[i]);
			h[i] = abs_d <= delta ? 0.5f * diff[i] * diff[i] : delta * fmaf(-0.5f, delta, abs_d);
		}
	}
	else
	{
		#pragma unroll
		for (int i = 0; i < elements_per_uint4; i++)
		{
			const int idx_local = i + element_idx;
			if (idx_local < num_elements)
			{
				const float diff = static_cast<float>(pred[idx_local]) - static_cast<float>(real[idx_local]);
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
template <typename T>
__global__ void huberGradOutputKernel(const T* __restrict__ pred, const T* __restrict__ real, float* __restrict__ grad_output, int spatial_size, int channels, float score_factor, float delta)
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
			diff[i] = static_cast<float>(pred[offset]) - static_cast<float>(real[offset]);
		}
		#pragma unroll
		for (int i = 0; i < 4; i++) sum[i] += fabsf(diff[i]) <= delta ? diff[i] : copysignf(delta, diff[i]);
	}
	float thread_sum = 0.0f;
	#pragma unroll
	for (int i = 0; i < 2; i++) thread_sum += sum[i << 1] + sum[(i << 1) + 1];
	if (thread_sum != 0.0f) atomicAdd(&grad_output[idx], thread_sum * factor);
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
