//block.h
#pragma once
#include "cl.h"

class Block
{
private:
	const int input_channels, output_channels, final_dimension, input_dimension;
	int *pipo_db, *pipo_db_num_groups;
	std::vector<std::unique_ptr<ConvLayer>> conv_layers;
	cudaStream_t stream_forward, stream_backward;
	const float survival_prob_inv;
	universal_vector<__half> grad_buffer_a, grad_buffer_b, grad_residual;
	universal_vector<uint32_t> buffer_a, buffer_b;
	universal_vector<int8_t> buffer_a_scales, buffer_b_scales;
public:
	void saveToFile(std::string pathFile) const
	{
		conv_layers[0]->saveToFile((std::filesystem::path(pathFile) / "depthwise_layer.bin").string());
		conv_layers[1]->saveToFile((std::filesystem::path(pathFile) / "conv_layer1.bin").string());
		conv_layers[2]->saveToFile((std::filesystem::path(pathFile) / "conv_layer2.bin").string());
		conv_layers[3]->saveToFile((std::filesystem::path(pathFile) / "residual_layer.bin").string());
	}
	void loadFromFile(std::string pathFile)
	{
		conv_layers[0]->loadFromFile((std::filesystem::path(pathFile) / "depthwise_layer.bin").string());
		conv_layers[1]->loadFromFile((std::filesystem::path(pathFile) / "conv_layer1.bin").string());
		conv_layers[2]->loadFromFile((std::filesystem::path(pathFile) / "conv_layer2.bin").string());
		conv_layers[3]->loadFromFile((std::filesystem::path(pathFile) / "residual_layer.bin").string());
	}
	Block(int in_channels, int out_channels, int stride, int in_dimension, int out_dimension, cudaStream_t stream_fwd,
		  int* total_size_grad_weights, int* total_size_grad_biases, int* db_pipo, int* db_pipo_num_groups,
		  int kernel_size_dw, float survival_probability, cudaStream_t stream_bwd, __half* learning_rate, float penalty,
		  size_t* total_mem_main_device, size_t* total_mem_main_host, size_t* total_learnable_data, size_t* total_learnable_data_count) :
		input_channels(in_channels),
		output_channels(out_channels),
		final_dimension(out_dimension),
		input_dimension(in_dimension),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd),
		pipo_db(db_pipo),
		pipo_db_num_groups(db_pipo_num_groups),
		survival_prob_inv(1.0f / survival_probability)
    {
		const int middle_channels = output_channels << 2,
				  local_pipo_db = middle_channels * final_dimension * final_dimension,
				  local_pipo_db_num_groups = local_pipo_db >> 5;
		*pipo_db = *pipo_db < local_pipo_db ? local_pipo_db : *pipo_db;
		*pipo_db_num_groups = *pipo_db_num_groups < local_pipo_db_num_groups ? local_pipo_db_num_groups : *pipo_db_num_groups;
		conv_layers.push_back(std::make_unique<ConvLayer>(input_channels, input_channels, kernel_size_dw, stride, input_dimension, final_dimension, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count));
		conv_layers.push_back(std::make_unique<ConvLayer>(input_channels, middle_channels, 1, 1, final_dimension, final_dimension, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count));
		conv_layers.push_back(std::make_unique<ConvLayer>(middle_channels, output_channels, 1, 1, final_dimension, final_dimension, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count));
		conv_layers.push_back(std::make_unique<ConvLayer>(input_channels, output_channels, 1, input_dimension == final_dimension ? 1 : stride, input_dimension, final_dimension, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count));
	}
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, bool is_in_inference, __half**& meta_adapted_data, size_t*& meta_adapted_offsets, uint8_t*& dev_side_ptr_grad_buffer_a, uint8_t*& dev_side_ptr_grad_buffer_b, uint8_t*& dev_side_ptr_grad_residual, uint8_t*& dev_main_ptr_buffer_a, uint8_t*& dev_main_ptr_buffer_a_scales, uint8_t*& dev_main_ptr_buffer_b, uint8_t*& dev_main_ptr_buffer_b_scales)
	{
		conv_layers[0]->initData(isLoaded, (std::filesystem::path(dir_path) / "depthwise_layer.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets);
		conv_layers[1]->initData(isLoaded, (std::filesystem::path(dir_path) / "conv_layer1.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets);
		conv_layers[2]->initData(isLoaded, (std::filesystem::path(dir_path) / "conv_layer2.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets);
		conv_layers[3]->initData(isLoaded, (std::filesystem::path(dir_path) / "residual_layer.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets);
		buffer_a.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_a), *pipo_db_num_groups << 2, MemoryType::Device);
		buffer_a_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_a_scales), *pipo_db_num_groups, MemoryType::Device);
		buffer_b.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_b), *pipo_db_num_groups << 2, MemoryType::Device);
		buffer_b_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_b_scales), *pipo_db_num_groups, MemoryType::Device);
		if (!is_in_inference)
		{
			grad_buffer_a.resize(reinterpret_cast<__half*>(dev_side_ptr_grad_buffer_a), *pipo_db, MemoryType::Device);
			grad_buffer_b.resize(reinterpret_cast<__half*>(dev_side_ptr_grad_buffer_b), *pipo_db, MemoryType::Device);
			grad_residual.resize(reinterpret_cast<__half*>(dev_side_ptr_grad_residual), *pipo_db, MemoryType::Device);
		}
	}
    void forward(const uint32_t* input, const int8_t* input_scales, uint32_t* output, int8_t* output_scales, bool train_mode = true, bool is_skipped = false)
    {
		if (!is_skipped || !train_mode)
		{
			conv_layers[0]->forward(input, input_scales, buffer_a, buffer_a_scales);
			const int middle_channels = output_channels << 2;
			dim3 block(gpu_block_threads);
			dim3 grid((middle_channels + block.x - 1) / block.x, final_dimension, final_dimension);
			const size_t shared_mem_bytes = (middle_channels >> 3) * sizeof(uint32_t);
			LAUNCH_KERNEL(convFusedPointwiseFwdKernel<>, grid, block, shared_mem_bytes, stream_forward, buffer_a.data(), buffer_a_scales.data(), conv_layers[1]->get_weights_ptr(), conv_layers[1]->get_weight_scales_ptr(), conv_layers[1]->get_biases_ptr(), conv_layers[2]->get_weights_ptr(), conv_layers[2]->get_weight_scales_ptr(), conv_layers[2]->get_biases_ptr(), buffer_a.data(), buffer_a_scales.data(), input_channels, middle_channels, output_channels, input_dimension, final_dimension, 1);
			//conv_layers[1]->forward(buffer_a.data(), buffer_a_scales.data(), buffer_b, buffer_b_scales);
			//conv_layers[2]->forward(buffer_b.data(), buffer_b_scales.data(), buffer_a, buffer_a_scales);
		}
		else
		{
			buffer_a.fill(0, buffer_a.size(), 0, stream_forward);
			buffer_a_scales.fill(0, buffer_a_scales.size(), 0, stream_forward);
		}
		conv_layers[3]->forward(input, input_scales, buffer_b, buffer_b_scales);
		dim3 blocks_grid(final_dimension, final_dimension);
		LAUNCH_KERNEL(residualAddFwdKernel<>, blocks_grid, gpu_block_threads, output_channels * sizeof(float), stream_forward, buffer_a.data(), buffer_a_scales.data(), buffer_b.data(), buffer_b_scales.data(), output, output_scales, output_channels, final_dimension, train_mode && !is_skipped ? survival_prob_inv : 1.0f);
	}
	void backward(universal_vector<__half>& grad_input_output, const uint32_t* input, const int8_t* input_scales, bool is_skipped = false)
	{
		if (!is_skipped)
		{
			conv_layers[0]->forward(input, input_scales, buffer_a, buffer_a_scales);
			conv_layers[1]->forward(buffer_a.data(), buffer_a_scales.data(), buffer_b, buffer_b_scales);
		}
		dim3 blocks_grid(final_dimension, final_dimension);
		LAUNCH_KERNEL(residualAddBwdKernel<>, blocks_grid, gpu_block_threads, 0, stream_backward, grad_input_output.data(), grad_residual.data(), output_channels, final_dimension, !is_skipped ? survival_prob_inv : 1.0f);
		if (!is_skipped)
		{
			conv_layers[2]->backward(grad_input_output, buffer_b.data(), buffer_b_scales.data(), grad_buffer_b);
			conv_layers[1]->backward(grad_buffer_b, buffer_a.data(), buffer_a_scales.data(), grad_buffer_a);
			conv_layers[0]->backward(grad_buffer_a, input, input_scales, grad_input_output);
		}
		conv_layers[3]->backward(grad_residual, input, input_scales, grad_buffer_b);
		__half* __restrict__ raw_target = grad_input_output.data();
		const __half* __restrict__ raw_a = grad_buffer_b.data();
		grad_input_output.for_each_n((input_channels * input_dimension * input_dimension) >> 3, [raw_target, raw_a] __device__ (size_t i) 
		{
			const size_t offset = i << 3;
			uint4* target_ptr = reinterpret_cast<uint4*>(raw_target + offset);
			uint4 target_data = *target_ptr;
			auto& target_h2 = reinterpret_cast<__half2(&)[4]>(target_data);
			const uint4 a_data = *reinterpret_cast<const uint4*>(raw_a + offset);
			const auto& a_h2 = reinterpret_cast<const __half2(&)[4]>(a_data);
			#pragma unroll
			for(int j = 0; j < 4; j++) target_h2[j] = __hadd2(target_h2[j], a_h2[j]);
			*target_ptr = target_data;
		}, stream_backward);
	}
};
