//mb.h
#pragma once
#include "kernels.h"

template<typename T = __half>
struct MambaWeightsOP
{
	T* ptr;
	__host__ __device__ void operator()(size_t i) const { ptr[i] = static_cast<T>(-12.1f - 0.4f * (i & (mamba_d_state - 1)) / (mamba_d_state - 1)); }
};

class MambaBlock
{
private:
	const int out_channels, real_size_weight, num_groups;
	universal_vector<uint32_t> weights;
	universal_vector<int8_t> weight_scales;
	universal_vector<__half> biases, grad_biases, mamba_output;
	universal_vector<type_gradients> master_weights, grad_weights;
	universal_vector<float> h_states, h_states_temp_buf, trace_A_re, trace_A_im, trace_B, trace_Delta;
	cudaStream_t stream_forward, stream_backward;
	int *total_size_grad_weights, *total_size_grad_biases, frame_counter = 0, *frame_counter_max = nullptr;
	bool trigger_memory_update = false;
	const float learning_rate, f_penalty;
public:
	void saveToFile(std::string pathFile) const
	{
		std::filesystem::path file_path(pathFile);
		std::filesystem::path dir = file_path.parent_path();
		if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
		std::ofstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for writing: " + pathFile);
		universal_vector<__half> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		biases_tmp.copy(biases, 0, out_channels, 0, out_channels, 0);
		file.write(reinterpret_cast<const char*>(master_weights.data()), real_size_weight * sizeof(std::remove_pointer_t<decltype(master_weights.data())>));
		file.write(reinterpret_cast<const char*>(biases_tmp.data()), out_channels * sizeof(__half));
		file.close();
		checkCudaError(cudaStreamSynchronize(0));
	}
	void loadFromFile(std::string pathFile)
	{
		std::ifstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for reading: " + pathFile);
		universal_vector<__half> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		file.read(reinterpret_cast<char*>(master_weights.data()), real_size_weight * sizeof(std::remove_pointer_t<decltype(master_weights.data())>));
		file.read(reinterpret_cast<char*>(biases_tmp.data()), out_channels * sizeof(__half));
		file.close();
		biases.copy(biases_tmp, 0, out_channels, 0, out_channels, 0);
		LAUNCH_KERNEL((quantizeKernel<std::remove_pointer_t<decltype(master_weights.data())>>), std::max(1, (num_groups + gpu_block_threads - 1) / gpu_block_threads), gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight);
		checkCudaError(cudaStreamSynchronize(0));
	}
	MambaBlock(int output_channels, cudaStream_t stream_fwd, int* total_size_g_weights, int* total_size_g_biases,
			   cudaStream_t stream_bwd, int* frame_counter_max_ptr, float learningRate, float penalty,
			   size_t* total_mem_device, size_t* total_mem_host, size_t* total_learnable_data, size_t* total_learnable_data_count) :
		out_channels(output_channels),
		real_size_weight(mamba_d_inner * (34 + (mamba_d_state << 2) + out_channels)),
		num_groups(real_size_weight >> 5),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd),
		total_size_grad_weights(total_size_g_weights),
		total_size_grad_biases(total_size_g_biases),
		frame_counter_max(frame_counter_max_ptr),
		learning_rate(learningRate),
		f_penalty(penalty)
	{
		*total_size_grad_weights = *total_size_grad_weights < real_size_weight ? real_size_weight : *total_size_grad_weights;
		*total_size_grad_biases = *total_size_grad_biases < out_channels ? out_channels : *total_size_grad_biases;
		*total_mem_device = *total_mem_device + align16(num_groups * sizeof(uint32_t) << 2) + align16(num_groups * sizeof(int8_t)) + align16(out_channels * sizeof(__half)) * 2 + align16(mamba_h_states_size * sizeof(float)) * 2 + align16(mamba_d_in_sta * sizeof(float)) * 4;
		*total_mem_host = *total_mem_host + align16(real_size_weight * sizeof(std::remove_pointer_t<decltype(master_weights.data())>));
		*total_learnable_data = *total_learnable_data + real_size_weight + out_channels;
		*total_learnable_data_count = *total_learnable_data_count + 2;
	}
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& current_dev_side_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, bool is_in_inference, void**& meta_adapted_data, size_t*& meta_adapted_offsets)
	{
		size_t w_bytes = align16(out_channels * sizeof(std::remove_pointer_t<decltype(mamba_output.data())>));
		mamba_output.resize(reinterpret_cast<std::remove_pointer_t<decltype(mamba_output.data())>*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(num_groups * sizeof(uint32_t) << 2);
		weights.resize(reinterpret_cast<uint32_t*>(current_dev_main_ptr), num_groups << 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(num_groups * sizeof(int8_t));
		weight_scales.resize(reinterpret_cast<int8_t*>(current_dev_main_ptr), num_groups, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(out_channels * sizeof(__half));
		biases.resize(reinterpret_cast<__half*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(std::remove_pointer_t<decltype(master_weights.data())>));
		master_weights.resize(reinterpret_cast<std::remove_pointer_t<decltype(master_weights.data())>*>(current_host_main_ptr), real_size_weight, MemoryType::PinnedHost);
		current_host_main_ptr += w_bytes;
		w_bytes = align16(mamba_h_states_size * sizeof(float));
		h_states.resize(reinterpret_cast<float*>(current_dev_main_ptr), mamba_h_states_size, 0.0f, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		h_states_temp_buf.resize(reinterpret_cast<float*>(current_dev_main_ptr), mamba_h_states_size, 0.0f, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		if (!is_in_inference)
		{
			grad_weights.resize(reinterpret_cast<std::remove_pointer_t<decltype(grad_weights.data())>*>(dev_side_ptr_weights), *total_size_grad_weights, MemoryType::Device);
			grad_biases.resize(reinterpret_cast<__half*>(dev_side_ptr_biases), *total_size_grad_biases, MemoryType::Device);
			w_bytes = align16(mamba_d_in_sta * sizeof(float));
			trace_A_re.resize(reinterpret_cast<float*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			trace_A_im.resize(reinterpret_cast<float*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			trace_B.resize(reinterpret_cast<float*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			trace_Delta.resize(reinterpret_cast<float*>(current_dev_side_ptr), mamba_d_in_sta, MemoryType::Device);
			current_dev_side_ptr += w_bytes;
			*meta_adapted_data = static_cast<void*>(master_weights.data());
			meta_adapted_data += 1;
			*meta_adapted_data = static_cast<void*>(biases.data());
			meta_adapted_data += 1;
			*meta_adapted_offsets = master_weights.size();
			meta_adapted_offsets += 1;
			*meta_adapted_offsets = biases.size();
			meta_adapted_offsets += 1;
		}
		if (isLoaded) loadFromFile(dir_path);
		else
		{
			generateNormalRandoms(master_weights, real_size_weight, std::sqrtf(2.0f / (final_channels + out_channels)), 0, 1234);
			generateNormalRandoms(biases, out_channels, std::sqrtf(2.0f / out_channels), 0, 5678);
			for(int b = 0; b < (mamba_d_inner + mamba_mimo_group_size - 1) / mamba_mimo_group_size; ++b)
			{
				for(int row = 0; row < mamba_mimo_group_size; ++row)
				{
					for(int col = 0; col < mamba_mimo_group_size; ++col)
					{
						const size_t global_idx = mamba_d_inner + b * mamba_mimo_matrix_elements + row * mamba_mimo_group_size + col;
						if (global_idx >= real_size_weight) break;
						master_weights[global_idx] = static_cast<std::remove_pointer_t<decltype(master_weights.data())>>(fmaf(getFastRandomFloatBoundedUniform(global_idx, std::sqrtf(1.0f / mamba_mimo_group_size)), 0.1f, static_cast<float>(row == col)));
					}
				}
			}
			constexpr size_t off_A_re = mamba_d_inner + (mamba_d_inner << 5) + mamba_d_in_sta + mamba_d_in_sta, off_A_im = off_A_re + mamba_d_in_sta;
			master_weights.for_each_n(mamba_d_in_sta, MambaWeightsOP<std::remove_pointer_t<decltype(master_weights.data())>>{master_weights.data() + off_A_re}, 0);
			master_weights.fill(off_A_im, off_A_im + mamba_d_in_sta, 0.001f, 0);
			LAUNCH_KERNEL((quantizeKernel<std::remove_pointer_t<decltype(master_weights.data())>>), std::max(1, (num_groups + gpu_block_threads - 1) / gpu_block_threads), gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight);
		}
	}
	void forward(const universal_vector<uint32_t>& input, const universal_vector<int8_t>& input_scales)
	{
		if (frame_counter_max != nullptr)
		{
			frame_counter++;
			if (frame_counter >= (*frame_counter_max >> 1))
			{
				trigger_memory_update = true;
				frame_counter = 0;
			}
			else trigger_memory_update = false;
		}
		else trigger_memory_update = true;
		float* target_h_states_ptr = trigger_memory_update ? h_states.data() : h_states_temp_buf.data();
		constexpr int num_warps = gpu_block_threads >> 5;
		dim3 grid_dim(out_channels, (((mamba_d_inner + 31) >> 5) + num_warps - 1) / num_warps, 1);
		LAUNCH_KERNEL(mambaSSMFwdKernel<>, grid_dim, gpu_block_threads, 0, stream_forward, input.data(), input_scales.data(), weights.data(), weight_scales.data(), biases.data(), target_h_states_ptr, trace_A_re.data(), trace_A_im.data(), trace_B.data(), trace_Delta.data(), mamba_output.data(), out_channels);
	}
	template<typename T = __half>
	void backward(universal_vector<T>& grad_output, const universal_vector<uint32_t>& input, const universal_vector<int8_t>& input_scales, universal_vector<type_gradients>& grad_input)
	{
		grad_weights.fill(0, real_size_weight, 0, stream_backward);
		grad_biases.fill(0, out_channels, 0, stream_backward);
		dim3 grid_dim((mamba_d_inner + gpu_block_threads - 1) / gpu_block_threads, 1, 1);
		LAUNCH_KERNEL((mambaSSMApplyTracesKernel<std::remove_pointer_t<decltype(grad_input.data())>, std::remove_pointer_t<decltype(grad_output.data())>>), grid_dim, gpu_block_threads, 0, stream_backward, input.data(), input_scales.data(), grad_output.data(), mamba_output.data(), trace_A_re.data(), trace_A_im.data(), trace_B.data(), trace_Delta.data(), master_weights.data(), grad_weights.data(), grad_biases.data(), grad_input.data(), out_channels);
		const float sum_sq_w = grad_weights.transform_reduce<float>(SquareOp<std::remove_pointer_t<decltype(grad_weights.data())>>{}, real_size_weight, stream_backward);
		const float sum_sq_b = grad_biases.transform_reduce<float>(SquareOp<std::remove_pointer_t<decltype(grad_biases.data())>>{}, out_channels, stream_backward);
		auto clip_op = [] (float sum_sq) -> float
		{
			const float norm = sqrtf(sum_sq + epsilon);
			return norm > 1.0f ? 1.0f / norm : 1.0f;
		};
		LAUNCH_KERNEL((updateParamsKernel<std::remove_pointer_t<decltype(master_weights.data())>>), std::max(1, (real_size_weight + gpu_block_threads - 1) / gpu_block_threads), gpu_block_threads, 0, stream_backward, master_weights.data(), grad_weights.data(), real_size_weight, clip_op(sum_sq_w), learning_rate, f_penalty);
		LAUNCH_KERNEL((updateParamsKernel<std::remove_pointer_t<decltype(biases.data())>>), std::max(1, (out_channels + gpu_block_threads - 1) / gpu_block_threads), gpu_block_threads, 0, stream_backward, biases.data(), grad_biases.data(), out_channels, clip_op(sum_sq_b), learning_rate, 0.0f);
		LAUNCH_KERNEL((quantizeKernel<std::remove_pointer_t<decltype(master_weights.data())>>), std::max(1, (num_groups + gpu_block_threads - 1) / gpu_block_threads), gpu_block_threads, 0, stream_backward, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight);
	}
	void reset(cudaStream_t stream_lnk = 0)
	{
		h_states.fill(0, h_states.size(), 0, stream_lnk);
		trace_A_re.fill(0, trace_A_re.size(), 0, stream_lnk);
		trace_A_im.fill(0, trace_A_im.size(), 0, stream_lnk);
		trace_B.fill(0, trace_B.size(), 0, stream_lnk);
		trace_Delta.fill(0, trace_Delta.size(), 0, stream_lnk);
	}
};
