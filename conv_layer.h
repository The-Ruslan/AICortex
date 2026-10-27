#pragma once
#include "kernels.h"

class ConvLayer
{
private:
	const int in_channels, out_channels, kernel_size, stride, dimension_in, dimension_out, real_size_weight, num_groups;
	universal_vector<uint32_t> weights;
	universal_vector<int8_t> weight_scales;
	universal_vector<type_learnable_parameters> master_weights, biases, grad_weights, grad_biases;
	cudaStream_t stream_forward, stream_backward;
	const float learning_rate, f_penalty;
	const bool disable_activation = false;
public:
	void saveToFile(std::string pathFile) const
	{
		std::filesystem::path file_path(pathFile);
		std::filesystem::path dir = file_path.parent_path();
		if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
		std::ofstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for writing: " + pathFile);
		universal_vector<type_learnable_parameters> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		biases_tmp.copy(biases, 0, out_channels, 0, out_channels, 0);
		file.write(reinterpret_cast<const char*>(master_weights.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.write(reinterpret_cast<const char*>(biases_tmp.data()), out_channels * sizeof(type_learnable_parameters));
		file.close();
		checkCudaError(cudaStreamSynchronize(0));
	}
	void loadFromFile(std::string pathFile)
	{
		std::ifstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for reading: " + pathFile);
		universal_vector<type_learnable_parameters> biases_tmp(out_channels, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		file.read(reinterpret_cast<char*>(master_weights.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.read(reinterpret_cast<char*>(biases_tmp.data()), out_channels * sizeof(type_learnable_parameters));
		file.close();
		biases.copy(biases_tmp, 0, out_channels, 0, out_channels, 0);
		LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
		checkCudaError(cudaStreamSynchronize(0));
	}
	ConvLayer(int inChannels, int outChannels, int kernelSize, int str_ide, int in_dimension, int out_dimension, cudaStream_t stream_fwd,
			  int* total_size_grad_weights, int* total_size_grad_biases, cudaStream_t stream_bwd, float learningRate, float penalty,
			  size_t* total_mem_main_device, size_t* total_mem_main_host, size_t* total_learnable_data, size_t* total_learnable_data_count, bool is_activation_disabled = false) :
		in_channels(inChannels), out_channels(outChannels), kernel_size(kernelSize), stride(str_ide), dimension_in(in_dimension), dimension_out(out_dimension),
		real_size_weight(in_channels * kernel_size * kernel_size * (kernel_size > 1 ? 1 : out_channels)),
		num_groups(real_size_weight >> 5),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd),
		learning_rate(learningRate),
		f_penalty(penalty),
		disable_activation(is_activation_disabled)
	{
		*total_size_grad_weights = *total_size_grad_weights < real_size_weight ? real_size_weight : *total_size_grad_weights;
		*total_size_grad_biases = *total_size_grad_biases < out_channels ? out_channels : *total_size_grad_biases;
		*total_mem_main_device = *total_mem_main_device + align16(num_groups * sizeof(uint32_t) << 2) + align16(num_groups * sizeof(int8_t)) + align16(out_channels * sizeof(type_learnable_parameters));
		*total_mem_main_host = *total_mem_main_host + align16(real_size_weight * sizeof(type_learnable_parameters));
		*total_learnable_data = *total_learnable_data + real_size_weight + out_channels;
		*total_learnable_data_count = *total_learnable_data_count + 2;
    }
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, bool is_in_inference, void**& meta_adapted_data, size_t*& meta_adapted_offsets, int total_size_grad_weights, int total_size_grad_biases)
	{
		size_t w_bytes = align16(num_groups * sizeof(uint32_t) << 2);
		weights.resize(reinterpret_cast<uint32_t*>(current_dev_main_ptr), num_groups << 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(num_groups * sizeof(int8_t));
		weight_scales.resize(reinterpret_cast<int8_t*>(current_dev_main_ptr), num_groups, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(out_channels * sizeof(type_learnable_parameters));
		biases.resize(reinterpret_cast<type_learnable_parameters*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(type_learnable_parameters));
		master_weights.resize(reinterpret_cast<type_learnable_parameters*>(current_host_main_ptr), real_size_weight, MemoryType::PinnedHost);
		current_host_main_ptr += w_bytes;
		if (!is_in_inference)
		{
			grad_weights.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_weights), total_size_grad_weights, MemoryType::Device);
			grad_biases.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_biases), total_size_grad_biases, MemoryType::Device);
			*meta_adapted_data = static_cast<void*>(master_weights.data());
			meta_adapted_data += 1;
			*meta_adapted_data = static_cast<void*>(biases.data());
			meta_adapted_data += 1;
			*meta_adapted_offsets = real_size_weight;
			meta_adapted_offsets += 1;
			*meta_adapted_offsets = out_channels;
			meta_adapted_offsets += 1;
		}
		if (isLoaded) loadFromFile(dir_path);
		else
		{
			float offset = 0.0f;
			if (kernel_size > 1) offset = std::sqrtf(1.0f / (kernel_size * kernel_size));
			else offset = std::sqrtf(2.0f / (in_channels + (disable_activation ? out_channels : 0)));
			generateNormalRandoms(master_weights, real_size_weight, offset, 0, 1234);
			LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, 0, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
			biases.fill(0, out_channels, 0, 0);
		}
	}
	void forward(const uint32_t* input, const int8_t* input_scales, universal_vector<uint32_t>& output, universal_vector<int8_t>& output_scales)
	{
		if (kernel_size > 1)
		{
			dim3 grid((in_channels + gpu_block_threads - 1) / gpu_block_threads, dimension_out, dimension_out);
			#define LAUNCH_CONV_FWD_KERNEL(K_SIZE) \
				LAUNCH_KERNEL((convDepthwiseFwdKernel<type_learnable_parameters, K_SIZE>), grid, gpu_block_threads, 0, stream_forward, input, weights.data(), input_scales, weight_scales.data(), biases.data(), output.data(), output_scales.data(), in_channels, dimension_in, dimension_out, stride)
			switch (kernel_size)
			{
				case 3:  LAUNCH_CONV_FWD_KERNEL(3);  break;
				case 5:  LAUNCH_CONV_FWD_KERNEL(5);  break;
				case 7:  LAUNCH_CONV_FWD_KERNEL(7);  break;
				case 9:  LAUNCH_CONV_FWD_KERNEL(9);  break;
				case 11: LAUNCH_CONV_FWD_KERNEL(11); break;
				case 13: LAUNCH_CONV_FWD_KERNEL(13); break;
				case 15: LAUNCH_CONV_FWD_KERNEL(15); break;
				default: throw std::runtime_error("Unsupported kernel size: " + std::to_string(kernel_size)); break;
			}
			#undef LAUNCH_CONV_FWD_KERNEL
		}
		else
		{
			dim3 grid((out_channels + gpu_block_threads - 1) / gpu_block_threads, dimension_out, dimension_out);
			if (disable_activation) LAUNCH_KERNEL((convPointwiseFwdKernel<type_learnable_parameters, true>), grid, gpu_block_threads, 0, stream_forward, input, weights.data(), input_scales, weight_scales.data(), biases.data(), output.data(), output_scales.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			else LAUNCH_KERNEL((convPointwiseFwdKernel<type_learnable_parameters, false>), grid, gpu_block_threads, 0, stream_forward, input, weights.data(), input_scales, weight_scales.data(), biases.data(), output.data(), output_scales.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
		}
	}
	void backward(const universal_vector<type_learnable_parameters>& grad_output, const uint32_t* input, const int8_t* input_scales, universal_vector<type_learnable_parameters>& grad_input)
	{
		if (kernel_size > 1)
		{
			dim3 grid_b((in_channels + gpu_block_threads - 1) / gpu_block_threads);
			dim3 grid_w(grid_b.x);
			dim3 grid_i(grid_b.x, dimension_in, dimension_in);
			LAUNCH_KERNEL(convDepthwiseBwdGradBiasKernel<type_learnable_parameters>, grid_b, gpu_block_threads, 0, stream_backward, grad_output.data(), grad_biases.data(), in_channels, dimension_out * dimension_out);
			#define LAUNCH_CONV_BWD_KERNEL(K_SIZE) \
				LAUNCH_KERNEL((convDepthwiseBwdGradWeightKernel<type_learnable_parameters, K_SIZE>), grid_w, gpu_block_threads, 0, stream_backward, input, input_scales, grad_output.data(), grad_weights.data(), in_channels, dimension_in, dimension_out, stride); \
				LAUNCH_KERNEL((convDepthwiseBwdGradInputKernel<type_learnable_parameters, K_SIZE>), grid_i, gpu_block_threads, 0, stream_backward, grad_output.data(), master_weights.data(), grad_input.data(), in_channels, dimension_in, dimension_out, stride)
			switch (kernel_size)
			{
				case 3:  LAUNCH_CONV_BWD_KERNEL(3);  break;
				case 5:  LAUNCH_CONV_BWD_KERNEL(5);  break;
				case 7:  LAUNCH_CONV_BWD_KERNEL(7);  break;
				case 9:  LAUNCH_CONV_BWD_KERNEL(9);  break;
				case 11: LAUNCH_CONV_BWD_KERNEL(11); break;
				case 13: LAUNCH_CONV_BWD_KERNEL(13); break;
				case 15: LAUNCH_CONV_BWD_KERNEL(15); break;
				default: throw std::runtime_error("Unsupported kernel size: " + kernel_size); break;
			}
			#undef LAUNCH_CONV_BWD_KERNEL
		}
		else
		{
			dim3 grid_b((out_channels + gpu_block_threads - 1) / gpu_block_threads);
			if (disable_activation) LAUNCH_KERNEL((convPointwiseBwdGradBiasKernel<type_learnable_parameters, true>), grid_b, gpu_block_threads, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), grad_biases.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			else LAUNCH_KERNEL((convPointwiseBwdGradBiasKernel<type_learnable_parameters, false>), grid_b, gpu_block_threads, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), grad_biases.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			dim3 block_w(dim_block_pw_bwd_gw_size, dim_block_pw_bwd_gw_size);
			dim3 grid_w(out_channels / dim_block_pw_bwd_gw_size, in_channels / dim_block_pw_bwd_gw_size);
			if (disable_activation) LAUNCH_KERNEL((convPointwiseBwdGradWeightKernel<type_learnable_parameters, true>), grid_w, block_w, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), grad_weights.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			else LAUNCH_KERNEL((convPointwiseBwdGradWeightKernel<type_learnable_parameters, false>), grid_w, block_w, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), grad_weights.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			dim3 grid_in(grid_b.x, dimension_out, dimension_out);
			if (disable_activation) LAUNCH_KERNEL((convPointwiseBwdGradInputKernel<type_learnable_parameters, true>), grid_in, gpu_block_threads, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), master_weights.data(), grad_input.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
			else LAUNCH_KERNEL((convPointwiseBwdGradInputKernel<type_learnable_parameters, false>), grid_in, gpu_block_threads, 0, stream_backward, input, input_scales, weights.data(), weight_scales.data(), biases.data(), grad_output.data(), master_weights.data(), grad_input.data(), in_channels, out_channels, dimension_in, dimension_out, stride);
		}
		const float sum_sq_w = grad_weights.transform_reduce<float>(SquareOp<type_learnable_parameters>{}, real_size_weight, stream_backward);
		const float sum_sq_b = grad_biases.transform_reduce<float>(SquareOp<type_learnable_parameters>{}, out_channels, stream_backward);
		auto clip_op = [] (float sum_sq) -> float
		{
			const float norm = sqrtf(sum_sq + epsilon);
			return norm > 1.0f ? 1.0f / norm : 1.0f;
		};
		LAUNCH_KERNEL(updateParamsKernel<>, (real_size_weight + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, master_weights.data(), grad_weights.data(), real_size_weight, clip_op(sum_sq_w), learning_rate, f_penalty);
		LAUNCH_KERNEL(updateParamsKernel<>, (out_channels + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, biases.data(), grad_biases.data(), out_channels, clip_op(sum_sq_b), learning_rate, 0.0f);
		LAUNCH_KERNEL(quantizeKernel<>, (num_groups + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, master_weights.data(), weights.data(), weight_scales.data(), real_size_weight, real_size_weight);
	}
};
