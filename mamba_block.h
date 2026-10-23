#pragma once
#include "mamba_layer.h"

class MambaBlock
{
private:
	const int input_channels, output_channels;
	std::vector<std::unique_ptr<MambaLayer>> mamba_layers;
	cudaStream_t stream_forward, stream_backward;
	universal_vector<uint32_t> buffer_a, buffer_b;
	universal_vector<int8_t> buffer_a_scales, buffer_b_scales;
	universal_vector<type_learnable_parameters> grad_buffer_a, grad_residual;
	using DataType1 = std::remove_pointer_t<decltype(mamba_layers[0]->get_output().data())>;
	using DataType2 = std::remove_pointer_t<decltype(mamba_layers[1]->get_output().data())>;
public:
	auto& get_output() { return mamba_layers[mamba_layers.size() < 1 ? 0 : mamba_layers.size() - 1]->get_output(); }
	void saveToFile(std::string pathFile) const
	{
		mamba_layers[0]->saveToFile((std::filesystem::path(pathFile) / "mamba_strategist.bin").string());
		mamba_layers[1]->saveToFile((std::filesystem::path(pathFile) / "mamba_tactic.bin").string());
		mamba_layers[2]->saveToFile((std::filesystem::path(pathFile) / "mamba_executor.bin").string());
	}
	void loadFromFile(std::string pathFile)
	{
		mamba_layers[0]->loadFromFile((std::filesystem::path(pathFile) / "mamba_strategist.bin").string());
		mamba_layers[1]->loadFromFile((std::filesystem::path(pathFile) / "mamba_tactic.bin").string());
		mamba_layers[2]->loadFromFile((std::filesystem::path(pathFile) / "mamba_executor.bin").string());
	}
	MambaBlock(int in_channels, int out_channels, cudaStream_t stream_fwd, int* total_size_grad_weights,
			   int* total_size_grad_biases, cudaStream_t stream_bwd, float learning_rate, float penalty,
			   size_t* total_mem_main_device, size_t* total_mem_main_host, size_t* total_learnable_data, size_t* total_learnable_data_count, bool use_silu = false) :
		input_channels(in_channels),
		output_channels(out_channels),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd)
    {
		mamba_layers.reserve(3);
		mamba_layers.push_back(std::make_unique<MambaLayer>(input_channels, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count, true));
		mamba_layers.push_back(std::make_unique<MambaLayer>(input_channels, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count, true));
		mamba_layers.push_back(std::make_unique<MambaLayer>(output_channels, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learning_rate, penalty, total_mem_main_device, total_mem_main_host, total_learnable_data, total_learnable_data_count, use_silu));
	}
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& dev_side_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, bool is_in_inference, void**& meta_adapted_data, size_t*& meta_adapted_offsets, uint8_t*& dev_side_ptr_grad_buffer_a, uint8_t*& dev_side_ptr_grad_residual, uint8_t*& dev_main_ptr_buffer_a, uint8_t*& dev_main_ptr_buffer_a_scales, uint8_t*& dev_main_ptr_buffer_b, uint8_t*& dev_main_ptr_buffer_b_scales, int pipo_db_num_groups, int total_size_grad_weights, int total_size_grad_biases)
	{
		mamba_layers[0]->initData(isLoaded, (std::filesystem::path(dir_path) / "mamba_strategist.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets, total_size_grad_weights, total_size_grad_biases);
		mamba_layers[1]->initData(isLoaded, (std::filesystem::path(dir_path) / "mamba_tactic.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets, total_size_grad_weights, total_size_grad_biases);
		mamba_layers[2]->initData(isLoaded, (std::filesystem::path(dir_path) / "mamba_executor.bin").string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, meta_adapted_data, meta_adapted_offsets, total_size_grad_weights, total_size_grad_biases);
		buffer_a.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_a), pipo_db_num_groups << 2, MemoryType::Device);
		buffer_a_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_a_scales), pipo_db_num_groups, MemoryType::Device);
		buffer_b.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_b), pipo_db_num_groups << 2, MemoryType::Device);
		buffer_b_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_b_scales), pipo_db_num_groups, MemoryType::Device);
		if (!is_in_inference)
		{
			grad_buffer_a.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_grad_buffer_a), input_channels, MemoryType::Device);
			grad_residual.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_grad_residual), input_channels, MemoryType::Device);
		}
	}
    void forward(const uint32_t* input, const int8_t* input_scales, bool on_training = true, int add_seed = 1234)
    {
		mamba_layers[0]->forward(input, input_scales);
		const int total_elements1 = mamba_layers[0]->get_output().size();
		const int grid_dim1 = ((total_elements1 >> 5) + gpu_block_threads - 1) / gpu_block_threads;
		if (on_training) LAUNCH_KERNEL((quantizeKernel<DataType1, false>), grid_dim1, gpu_block_threads, 0, stream_forward, mamba_layers[0]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements1, add_seed);
		else LAUNCH_KERNEL((quantizeKernel<DataType1, true>), grid_dim1, gpu_block_threads, 0, stream_forward, mamba_layers[0]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements1, 0);
		LAUNCH_KERNEL(residualAddFwdKernel<>, 1, gpu_block_threads, input_channels * sizeof(float), stream_forward, buffer_a.data(), buffer_a_scales.data(), input, input_scales, buffer_b.data(), buffer_b_scales.data(), input_channels, 1, 1.0f, false);
		mamba_layers[1]->forward(buffer_b.data(), buffer_b_scales.data());
		const int total_elements2 = mamba_layers[1]->get_output().size();
		const int grid_dim2 = ((total_elements2 >> 5) + gpu_block_threads - 1) / gpu_block_threads;
		if (on_training) LAUNCH_KERNEL((quantizeKernel<DataType2, false>), grid_dim2, gpu_block_threads, 0, stream_forward, mamba_layers[1]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements2, add_seed);
		else LAUNCH_KERNEL((quantizeKernel<DataType2, true>), grid_dim2, gpu_block_threads, 0, stream_forward, mamba_layers[1]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements2, 0);
		mamba_layers[2]->forward(buffer_a.data(), buffer_a_scales.data());
	}
	template <typename T>
	void backward(universal_vector<T>& grad_output, const uint32_t* input, const int8_t* input_scales, universal_vector<type_learnable_parameters>& grad_input, int add_seed = 1234)
	{
		mamba_layers[0]->forward(input, input_scales);
		const int total_elements1 = mamba_layers[0]->get_output().size();
		LAUNCH_KERNEL((quantizeKernel<DataType1, false>), ((total_elements1 >> 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward, mamba_layers[0]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements1, add_seed);
		LAUNCH_KERNEL(residualAddFwdKernel<>, 1, gpu_block_threads, input_channels * sizeof(float), stream_forward, buffer_a.data(), buffer_a_scales.data(), input, input_scales, buffer_b.data(), buffer_b_scales.data(), input_channels, 1, 1.0f, false);
		mamba_layers[1]->forward(buffer_b.data(), buffer_b_scales.data());
		const int total_elements2 = mamba_layers[1]->get_output().size();
		LAUNCH_KERNEL((quantizeKernel<DataType2, false>), ((total_elements2 >> 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward, mamba_layers[1]->get_output().data(), buffer_a.data(), buffer_a_scales.data(), total_elements2, add_seed);
		checkCudaError(cudaStreamSynchronize(stream_forward));
		mamba_layers[2]->backward(grad_output, buffer_a.data(), buffer_a_scales.data(), grad_buffer_a);
		mamba_layers[1]->backward(grad_buffer_a, buffer_b.data(), buffer_b_scales.data(), grad_residual);
		LAUNCH_KERNEL(residualAddBwdKernel<>, 1, gpu_block_threads, 0, stream_backward, grad_residual.data(), grad_input.data(), input_channels, 1, 1.0f);
		mamba_layers[0]->backward(grad_residual, input, input_scales, grad_buffer_a);
		auto* __restrict__ raw_target = grad_input.data();
		const auto* __restrict__ raw_a = grad_buffer_a.data();
		constexpr int elements_per_uint4 = sizeof(uint4) / sizeof(type_learnable_parameters);
		grad_input.for_each_n(input_channels / elements_per_uint4, [raw_target, raw_a] __device__ (size_t i) 
		{
			const size_t offset = i * elements_per_uint4;
			type_learnable_parameters local_target[elements_per_uint4], local_a[elements_per_uint4];
			uint4* target_ptr = reinterpret_cast<uint4*>(raw_target + offset);
			*reinterpret_cast<uint4*>(local_target) = *target_ptr;
			*reinterpret_cast<uint4*>(local_a) = *reinterpret_cast<const uint4*>(raw_a + offset);
			#pragma unroll
			for (int j = 0; j < elements_per_uint4; ++j) local_target[j] += local_a[j];
			*target_ptr = *reinterpret_cast<uint4*>(local_target);
		}, stream_backward);
	}
	void reset(cudaStream_t stream_lnk = 0) { for (int i = 0; i < mamba_layers.size(); i++) mamba_layers[i]->reset(stream_lnk); }
};
