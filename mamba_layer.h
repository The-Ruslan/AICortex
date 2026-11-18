#pragma once
#include "kernels.h"

template <typename T>
struct MambaWeightsOP
{
	T* ptr;
	__host__ __device__ void operator()(size_t i) const { ptr[i] = static_cast<T>(-12.1f - 0.4f * (i & (mamba_d_state - 1)) / (mamba_d_state - 1)); }
};

class MambaLayer
{
private:
	const int out_channels, real_size_weight, num_groups, activation_type;
	cudaStream_t stream_forward, stream_backward;
	const float learning_rate, f_penalty;
	universal_vector<uint32_t> weights;
	universal_vector<int8_t> weight_scales;
	universal_vector<float> master_weights, biases, grad_weights, grad_biases;
	universal_vector<float> mamba_output, y_ssm;
	universal_vector<float2> h_states, trace_A, trace_B, trace_Delta;
public:
	auto& get_output() { return mamba_output; }
	void saveToFile(std::string pathFile) const
	{
		std::filesystem::path file_path(pathFile);
		std::filesystem::path dir = file_path.parent_path();
		if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
		std::ofstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for writing: " + pathFile);
		universal_vector<float> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		biases_tmp.copy(biases, 0, out_channels, 0, out_channels, 0);
		file.write(reinterpret_cast<const char*>(master_weights.data()), real_size_weight * sizeof(float));
		file.write(reinterpret_cast<const char*>(biases_tmp.data()), out_channels * sizeof(float));
		file.close();
		checkCudaError(cudaStreamSynchronize(0));
	}
	void loadFromFile(std::string pathFile)
	{
		std::ifstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for reading: " + pathFile);
		universal_vector<float> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		file.read(reinterpret_cast<char*>(master_weights.data()), real_size_weight * sizeof(float));
		file.read(reinterpret_cast<char*>(biases_tmp.data()), out_channels * sizeof(float));
		file.close();
		biases.copy(biases_tmp, 0, out_channels, 0, out_channels, 0);
		LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
		checkCudaError(cudaStreamSynchronize(0));
	}
	MambaLayer(int output_channels, cudaStream_t stream_fwd, int* total_size_grad_weights, int* total_size_grad_biases,
			   cudaStream_t stream_bwd, float learningRate, float penalty,
			   size_t* total_mem_device, size_t* total_mem_host, size_t* total_learnable_data, size_t* total_learnable_data_count, int type_activation) :
		out_channels(output_channels),
		real_size_weight(mamba_d_inner * (2 + mamba_mimo_group_size + (mamba_d_state << 2) + out_channels)),
		num_groups(real_size_weight >> 5),
		activation_type(type_activation),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd),
		learning_rate(learningRate),
		f_penalty(penalty)
	{
		*total_size_grad_weights = *total_size_grad_weights < real_size_weight ? real_size_weight : *total_size_grad_weights;
		*total_size_grad_biases = *total_size_grad_biases < out_channels ? out_channels : *total_size_grad_biases;
		*total_mem_device = *total_mem_device + align16(num_groups * sizeof(uint32_t) << 2) + align16(num_groups * sizeof(int8_t)) + align16(out_channels * sizeof(float)) + align16(out_channels * sizeof(float)) + align16(mamba_h_states_size * sizeof(float2)) + align16(mamba_d_in_sta * sizeof(float2)) * 3 + align16(mamba_d_inner * sizeof(float));
		*total_mem_host = *total_mem_host + align16(real_size_weight * sizeof(float));
		if (total_learnable_data) *total_learnable_data = *total_learnable_data + real_size_weight + out_channels;
		if (total_learnable_data_count) *total_learnable_data_count = *total_learnable_data_count + 2;
	}
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& current_dev_side_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, bool is_in_inference, float*** meta_adapted_data, size_t** meta_adapted_offsets, int total_size_grad_weights, int total_size_grad_biases)
	{
		size_t w_bytes = align16(out_channels * sizeof(float));
		mamba_output.resize(reinterpret_cast<float*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(num_groups * sizeof(uint32_t) << 2);
		weights.resize(reinterpret_cast<uint32_t*>(current_dev_main_ptr), num_groups << 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(num_groups * sizeof(int8_t));
		weight_scales.resize(reinterpret_cast<int8_t*>(current_dev_main_ptr), num_groups, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(out_channels * sizeof(float));
		biases.resize(reinterpret_cast<float*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(float));
		master_weights.resize(reinterpret_cast<float*>(current_host_main_ptr), real_size_weight, MemoryType::PinnedHost);
		current_host_main_ptr += w_bytes;
		w_bytes = align16(mamba_h_states_size * sizeof(float2));
		h_states.resize(reinterpret_cast<float2*>(current_dev_main_ptr), mamba_h_states_size, make_float2(0.0f, 0.0f), MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(mamba_d_inner * sizeof(float));
		y_ssm.resize(reinterpret_cast<float*>(current_dev_main_ptr), mamba_d_inner, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		if (!is_in_inference)
		{
			grad_weights.resize(reinterpret_cast<float*>(dev_side_ptr_weights), total_size_grad_weights, MemoryType::Device);
			grad_biases.resize(reinterpret_cast<float*>(dev_side_ptr_biases), total_size_grad_biases, MemoryType::Device);
			w_bytes = align16(mamba_d_in_sta * sizeof(float2));
			trace_A.resize(reinterpret_cast<float2*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			trace_B.resize(reinterpret_cast<float2*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			trace_Delta.resize(reinterpret_cast<float2*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			**meta_adapted_data = master_weights.data();
			(*meta_adapted_data) += 1;
			**meta_adapted_data = biases.data();
			(*meta_adapted_data) += 1;
			**meta_adapted_offsets = real_size_weight;
			(*meta_adapted_offsets) += 1;
			**meta_adapted_offsets = out_channels;
			(*meta_adapted_offsets) += 1;
		}
		if (isLoaded) loadFromFile(dir_path);
		else
		{
			generateNormalRandoms(master_weights, real_size_weight, std::sqrtf(2.0f / mamba_d_inner), 0, real_size_weight);
			biases.fill(0, biases.size(), 0, 0);
			for(int b = 0; b < (mamba_d_inner + mamba_mimo_group_size - 1) / mamba_mimo_group_size; ++b)
			{
				for(int row = 0; row < mamba_mimo_group_size; ++row)
				{
					for(int col = 0; col < mamba_mimo_group_size; ++col)
					{
						const size_t global_idx = mamba_d_inner + b * mamba_mimo_matrix_elements + row * mamba_mimo_group_size + col;
						if (global_idx >= real_size_weight) break;
						master_weights[global_idx] = fmaf(getFastRandomFloatBoundedUniform(global_idx, std::sqrtf(1.0f / mamba_mimo_group_size)), 0.1f, static_cast<float>(row == col));
					}
				}
			}
			constexpr size_t off_A_re = mamba_d_inner + (mamba_d_inner << 5) + mamba_d_in_sta + mamba_d_in_sta, off_A_im = off_A_re + mamba_d_in_sta;
			master_weights.for_each_n(mamba_d_in_sta, MambaWeightsOP<float>{master_weights.data() + off_A_re}, 0);
			master_weights.fill(off_A_im, off_A_im + mamba_d_in_sta, 0.001f, 0);
			LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
		}
	}
	void forward(const uint32_t* input, const int8_t* input_scales)
	{
		LAUNCH_KERNEL(mambaSSMFwdStateKernel<>, mamba_d_inner / gpu_block_threads, gpu_block_threads, 0, stream_forward, input, input_scales, weights.data(), weight_scales.data(), h_states.data(), trace_A.data(), trace_B.data(), trace_Delta.data(), y_ssm.data());
		const int act_grid = (out_channels + gpu_block_threads - 1) / gpu_block_threads;
		if (activation_type == 0) LAUNCH_KERNEL(mambaSSMFwdOutProjKernel<0>, act_grid, gpu_block_threads, 0, stream_forward, y_ssm.data(), weights.data(), weight_scales.data(), biases.data(), mamba_output.data(), out_channels);
		else if (activation_type == 1) LAUNCH_KERNEL(mambaSSMFwdOutProjKernel<1>, act_grid, gpu_block_threads, 0, stream_forward, y_ssm.data(), weights.data(), weight_scales.data(), biases.data(), mamba_output.data(), out_channels);
		else LAUNCH_KERNEL(mambaSSMFwdOutProjKernel<2>, act_grid, gpu_block_threads, 0, stream_forward, y_ssm.data(), weights.data(), weight_scales.data(), biases.data(), mamba_output.data(), out_channels);
	}
	void backward(const float* grad_output, const uint32_t* input, const int8_t* input_scales, universal_vector<float>& grad_input)
	{
		grad_weights.fill(0, real_size_weight, 0, stream_backward);
		grad_biases.fill(0, out_channels, 0, stream_backward);
		grad_input.fill(0, grad_input.size(), 0, stream_backward);
		mamba_output.fill(0, out_channels, 0, stream_backward);
		dim3 proj_grid(out_channels >> 5, mamba_d_inner / gpu_block_threads, 1);
		LAUNCH_KERNEL(mambaSSMBwdOutProjKernel<>, proj_grid, gpu_block_threads, 0, stream_backward, input, input_scales, h_states.data(), master_weights.data(), mamba_output.data(), out_channels);
		const int act_grid = (out_channels + gpu_block_threads - 1) / gpu_block_threads;
		if (activation_type == 0) LAUNCH_KERNEL(mambaSSMBwdNormActKernel<0>, act_grid, gpu_block_threads, 0, stream_backward, mamba_output.data(), biases.data(), grad_output, grad_biases.data(), out_channels);
		else if (activation_type == 1) LAUNCH_KERNEL(mambaSSMBwdNormActKernel<1>, act_grid, gpu_block_threads, 0, stream_backward, mamba_output.data(), biases.data(), grad_output, grad_biases.data(), out_channels);
		else LAUNCH_KERNEL(mambaSSMBwdNormActKernel<2>, act_grid, gpu_block_threads, 0, stream_backward, mamba_output.data(), biases.data(), grad_output, grad_biases.data(), out_channels);
		LAUNCH_KERNEL(mambaSSMApplyTracesKernel<>, mamba_d_inner / gpu_block_threads, gpu_block_threads, 0, stream_backward, input, input_scales, mamba_output.data(), h_states.data(), trace_A.data(), trace_B.data(), trace_Delta.data(), master_weights.data(), grad_weights.data(), grad_input.data(), out_channels);
		const float sum_sq_w = grad_weights.transform_reduce<float>(SquareOp<float>{}, real_size_weight, stream_backward);
		const float sum_sq_b = grad_biases.transform_reduce<float>(SquareOp<float>{}, out_channels, stream_backward);
		auto clip_op = [] (float sum_sq) -> float
		{
			const float norm = sqrtf(sum_sq + epsilon);
			return norm > 1.0f ? 1.0f / norm : 1.0f;
		};
		LAUNCH_KERNEL(updateParamsKernel<>, (real_size_weight + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, master_weights.data(), grad_weights.data(), real_size_weight, clip_op(sum_sq_w), learning_rate, f_penalty);
		LAUNCH_KERNEL(updateParamsKernel<>, (out_channels + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, biases.data(), grad_biases.data(), out_channels, clip_op(sum_sq_b), learning_rate, 0.0f);
		LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
	}
	void reset(cudaStream_t stream_lnk = 0)
	{
		h_states.fill(0, h_states.size(), make_float2(0.0f, 0.0f), stream_lnk);
		trace_A.fill(0, trace_A.size(), make_float2(0.0f, 0.0f), stream_lnk);
		trace_B.fill(0, trace_B.size(), make_float2(0.0f, 0.0f), stream_lnk);
		trace_Delta.fill(0, trace_Delta.size(), make_float2(0.0f, 0.0f), stream_lnk);
	}
};
