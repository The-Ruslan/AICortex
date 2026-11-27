#pragma once
#include "conv_layer.h"
#include "mamba_layer.h"

class Critic
{
private:
	const int out_channels;
	cudaStream_t stream_forward, stream_backward;
	int *pipo_db, *pipo_db_num_groups;
	std::vector<std::unique_ptr<ConvLayer>> conv_layers;
	std::unique_ptr<MambaLayer> evaluator;
	universal_vector<float> grad_buffer_a, score_output, score_real;
	universal_vector<uint32_t> buffer_a, buffer_b;
	universal_vector<int8_t> buffer_a_scales, buffer_b_scales;
public:
	void saveToFile(std::string pathFile) const
	{
		for (int i = 0; i < conv_layers.size(); i++) conv_layers[i]->saveToFile((std::filesystem::path(pathFile) / "conv_layers" / (std::to_string(i) + ".bin")).string());
		evaluator->saveToFile((std::filesystem::path(pathFile) / "evaluator_layer.bin").string());
	}
	void loadFromFile(std::string pathFile)
	{
		for (int i = 0; i < conv_layers.size(); i++) conv_layers[i]->loadFromFile((std::filesystem::path(pathFile) / "conv_layers" / (std::to_string(i) + ".bin")).string());
		evaluator->loadFromFile((std::filesystem::path(pathFile) / "evaluator_layer.bin").string());
	}
	Critic(int inChannels, int outChannels, int in_dimension, cudaStream_t stream_fwd, int* total_size_grad_weights,
		   int* total_size_grad_biases, int* db_pipo, int* db_pipo_num_groups, cudaStream_t stream_bwd,
		   float learningRate, float penalty, size_t* total_mem_main_device, size_t* total_mem_main_host) :
		out_channels(outChannels), stream_forward(stream_fwd), stream_backward(stream_bwd),
		pipo_db(db_pipo), pipo_db_num_groups(db_pipo_num_groups)
	{
		int in_channels = inChannels, middle_channels = in_channels << 2;
		*total_mem_main_host = *total_mem_main_host + align16(out_channels * sizeof(float)) * 2;
		conv_layers.reserve(4);
		for (int i = in_dimension; i > 1; i >>=4)
		{
			const int max_output_dim = std::max(1, i >> 4);
			const int local_pipo_db = std::max(middle_channels * max_output_dim * max_output_dim, in_channels * i * i);
			const int local_pipo_db_num_groups = local_pipo_db >> 5;
			if (*pipo_db < local_pipo_db) *pipo_db = local_pipo_db;
			if (*pipo_db_num_groups < local_pipo_db_num_groups) *pipo_db_num_groups = local_pipo_db_num_groups;
			conv_layers.push_back(std::make_unique<ConvLayer>(in_channels, middle_channels, i < 16 ? i : 16, i < 16 ? i : 16, i, max_output_dim, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learningRate, penalty, total_mem_main_device, total_mem_main_host, nullptr, nullptr, false));
			in_channels = middle_channels;
			middle_channels = in_channels << 2;
		}
		evaluator = std::make_unique<MambaLayer>(in_channels, out_channels, stream_forward, total_size_grad_weights, total_size_grad_biases, stream_backward, learningRate, penalty, total_mem_main_device, total_mem_main_host, nullptr, nullptr, 0, -8.8f);
    }
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& current_host_main_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, int total_size_grad_weights, int total_size_grad_biases, uint8_t*& dev_main_ptr_buffer_a, uint8_t*& dev_main_ptr_buffer_a_scales, uint8_t*& dev_main_ptr_buffer_b, uint8_t*& dev_main_ptr_buffer_b_scales, uint8_t*& dev_main_ptr_grad_buffer_a)
	{
		score_output.resize(reinterpret_cast<float*>(current_host_main_ptr), out_channels, MemoryType::PinnedHost);
		current_host_main_ptr += align16(out_channels * sizeof(float));
		score_real.resize(reinterpret_cast<float*>(current_host_main_ptr), out_channels, MemoryType::PinnedHost);
		current_host_main_ptr += align16(out_channels * sizeof(float));
		for (int i = 0; i < conv_layers.size(); i++) conv_layers[i]->initData(isLoaded, (std::filesystem::path(dir_path) / "conv_layers" / (std::to_string(i) + ".bin")).string(), current_dev_main_ptr, current_host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, false, nullptr, nullptr, total_size_grad_weights, total_size_grad_biases);
		evaluator->initData(isLoaded, (std::filesystem::path(dir_path) / "evaluator_layer.bin").string(), current_dev_main_ptr, current_host_main_ptr, current_dev_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, false, nullptr, nullptr, total_size_grad_weights, total_size_grad_biases);
		buffer_a.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_a), *pipo_db_num_groups << 2, MemoryType::Device);
		buffer_a_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_a_scales), *pipo_db_num_groups, MemoryType::Device);
		buffer_b.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_b), *pipo_db_num_groups << 2, MemoryType::Device);
		buffer_b_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_b_scales), *pipo_db_num_groups, MemoryType::Device);
		grad_buffer_a.resize(reinterpret_cast<float*>(dev_main_ptr_grad_buffer_a), *pipo_db, MemoryType::Device);
	}
	void forward(const uint32_t* input, const int8_t* input_scales)
	{
		const uint32_t* base_input = input;
		const int8_t* base_input_scales = input_scales;
		int mask = 0;
		for (size_t i = 0; i < conv_layers.size(); i++)
		{
			mask = (mask + 1) & 1;
			uint32_t* next_input = mask ? buffer_a.data() : buffer_b.data();
			int8_t* next_scales = mask ? buffer_a_scales.data() : buffer_b_scales.data();
			conv_layers[i]->forward(base_input, base_input_scales, next_input, next_scales);
			base_input = next_input;
			base_input_scales = next_scales;
		}
		evaluator->forward(base_input, base_input_scales);
		score_output.copy(evaluator->get_output(), 0, out_channels, 0, out_channels, stream_forward);
		checkCudaError(cudaStreamSynchronize(stream_forward));
	}
	void backward(const float* real_score, const uint32_t* input, const int8_t* input_scales, universal_vector<float>& grad_input)
	{
		score_real.copy(real_score, 0, out_channels, 0, out_channels, stream_backward);
		int grad_mask = conv_layers.size() & 1, mask = 0;
		auto* __restrict__ raw_target = grad_mask ? grad_input.data() : grad_buffer_a.data();
		const auto* __restrict__ raw_score = evaluator->get_output().data();
		const float* real_score_ptr = score_real.data();
		grad_input.for_each_n(evaluator->get_output().size(), [raw_target, raw_score, real_score_ptr] __device__ (size_t i)
		{
			const float target = real_score_ptr[i];
			const float pred = raw_score[i];
			const float diff = pred - target;
			const float delta = fmaxf(0.1f, fmaxf(fabsf(target), fabsf(pred)) * 0.25f);
			raw_target[i] = fabsf(diff) <= delta ? diff : copysignf(delta, diff);
		}, stream_backward);
		const uint32_t* eval_input = input;
		const int8_t* eval_input_scales = input_scales;
		for (size_t i = 0; i < conv_layers.size(); i++)
		{
			mask = (mask + 1) & 1;
			uint32_t* next_input = mask ? buffer_a.data() : buffer_b.data();
			int8_t* next_scales = mask ? buffer_a_scales.data() : buffer_b_scales.data();
			conv_layers[i]->forward(eval_input, eval_input_scales, next_input, next_scales);
			eval_input = next_input;
			eval_input_scales = next_scales;
		}
		const float* eval_grad_out = grad_mask ? grad_input.data() : grad_buffer_a.data();
		checkCudaError(cudaStreamSynchronize(stream_forward));
		evaluator->backward(eval_grad_out, eval_input, eval_input_scales, grad_mask ? grad_buffer_a : grad_input);
		for (int i = conv_layers.size() - 1; i >= 0; i--)
		{
			const uint32_t* layer_input = input;
			const int8_t* layer_input_scales = input_scales;
			mask = 0;
			for (int j = 0; j < i; j++)
			{
				mask = (mask + 1) & 1;
				uint32_t* next_input = mask ? buffer_a.data() : buffer_b.data();
				int8_t* next_scales = mask ? buffer_a_scales.data() : buffer_b_scales.data();
				conv_layers[j]->forward(layer_input, layer_input_scales, next_input, next_scales);
				layer_input = next_input;
				layer_input_scales = next_scales;
			}
			checkCudaError(cudaStreamSynchronize(stream_forward));
			const float* base_grad = grad_mask ? grad_buffer_a.data() : grad_input.data();
			grad_mask ^= 1;
			float* target_grad = grad_mask ? grad_buffer_a.data() : grad_input.data();
			conv_layers[i]->backward(base_grad, layer_input, layer_input_scales, target_grad);
			checkCudaError(cudaStreamSynchronize(stream_backward));
		}
	}
	void reset(cudaStream_t stream_lnk = 0) { evaluator->reset(stream_lnk); }
	float get_score_output(size_t idx) const { return score_output[idx]; }
	auto& get_output() { return score_output; }
};
