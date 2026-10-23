#pragma once
#include "kernels.h"

template <typename T>
struct LambdaWeightsOP
{
	T* ptr;
	const int real_size_weight;
	__host__ __device__ void operator()(size_t i) const { ptr[i] = static_cast<T>(2.0f + 0.2f * (i & (real_size_weight - 1)) / (real_size_weight - 1)); }
};

class Critic
{
private:
	const int in_channels, out_channels, dimension_in, real_size_weight;
	universal_vector<__half> hidden_pool;
	universal_vector<float> hidden_state, score_output, trace_w1, trace_lambda;
	universal_vector<type_learnable_parameters> weights1, weights2, weights_lambda, grad_weights, grad_biases, grad_lambda;
	cudaStream_t stream_forward, stream_backward;
	const float learning_rate, f_penalty;
public:
	float get_score_output() const { return score_output[0]; }
	void saveToFile(std::string pathFile) const
	{
		std::filesystem::path file_path(pathFile);
		std::filesystem::path dir = file_path.parent_path();
		if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
		std::ofstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for writing: " + pathFile);
		universal_vector<type_learnable_parameters> weights1_tmp(real_size_weight, MemoryType::PinnedHost), weights2_tmp(out_channels, MemoryType::PinnedHost), weights_lambda_tmp(real_size_weight, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		weights1_tmp.copy(weights1, 0, real_size_weight, 0, real_size_weight, 0);
		weights2_tmp.copy(weights2, 0, out_channels, 0, out_channels, 0);
		weights_lambda_tmp.copy(weights_lambda, 0, real_size_weight, 0, real_size_weight, 0);
		file.write(reinterpret_cast<const char*>(weights1_tmp.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.write(reinterpret_cast<const char*>(weights2_tmp.data()), out_channels * sizeof(type_learnable_parameters));
		file.write(reinterpret_cast<const char*>(weights_lambda_tmp.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.close();
		checkCudaError(cudaStreamSynchronize(0));
	}
	void loadFromFile(std::string pathFile)
	{
		std::ifstream file(pathFile, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for reading: " + pathFile);
		universal_vector<type_learnable_parameters> weights1_tmp(real_size_weight, MemoryType::PinnedHost), weights2_tmp(out_channels, MemoryType::PinnedHost), weights_lambda_tmp(real_size_weight, MemoryType::PinnedHost);
		checkCudaError(cudaStreamSynchronize(stream_backward));
		checkCudaError(cudaStreamSynchronize(stream_forward));
		file.read(reinterpret_cast<char*>(weights1_tmp.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.read(reinterpret_cast<char*>(weights2_tmp.data()), out_channels * sizeof(type_learnable_parameters));
		file.read(reinterpret_cast<char*>(weights_lambda_tmp.data()), real_size_weight * sizeof(type_learnable_parameters));
		file.close();
		weights1.copy(weights1_tmp, 0, real_size_weight, 0, real_size_weight, 0);
		weights2.copy(weights2_tmp, 0, out_channels, 0, out_channels, 0);
		weights_lambda.copy(weights_lambda_tmp, 0, real_size_weight, 0, real_size_weight, 0);
		checkCudaError(cudaStreamSynchronize(0));
	}
	Critic(int inChannels, int outChannels, int in_dimension, cudaStream_t stream_fwd, int* total_size_grad_weights, int* total_size_grad_biases, cudaStream_t stream_bwd, float learningRate, float penalty, size_t* total_mem_main_device) :
		in_channels(inChannels), out_channels(outChannels), dimension_in(in_dimension),
		real_size_weight(in_channels * out_channels),
		stream_forward(stream_fwd),
		stream_backward(stream_bwd),
		learning_rate(learningRate),
		f_penalty(penalty)
	{
		*total_size_grad_weights = *total_size_grad_weights < real_size_weight ? real_size_weight : *total_size_grad_weights;
		*total_size_grad_biases = *total_size_grad_biases < out_channels ? out_channels : *total_size_grad_biases;
		*total_mem_main_device = *total_mem_main_device + align16(real_size_weight * sizeof(type_learnable_parameters)) * 3 + align16(real_size_weight * sizeof(float)) * 2 + align16(out_channels * sizeof(type_learnable_parameters)) + align16(out_channels * sizeof(float)) + align16(in_channels * sizeof(__half));
    }
	void initData(bool isLoaded, std::string dir_path, uint8_t*& current_dev_main_ptr, uint8_t*& dev_side_ptr_weights, uint8_t*& dev_side_ptr_biases, int total_size_grad_weights, int total_size_grad_biases)
	{
		score_output.resize(1, MemoryType::PinnedHost);
		size_t w_bytes = align16(real_size_weight * sizeof(type_learnable_parameters));
		weights1.resize(reinterpret_cast<type_learnable_parameters*>(current_dev_main_ptr), real_size_weight, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(type_learnable_parameters));
		weights_lambda.resize(reinterpret_cast<type_learnable_parameters*>(current_dev_main_ptr), real_size_weight, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(type_learnable_parameters));
		grad_lambda.resize(reinterpret_cast<type_learnable_parameters*>(current_dev_main_ptr), real_size_weight, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(real_size_weight * sizeof(float));
		trace_w1.resize(reinterpret_cast<float*>(current_dev_main_ptr), real_size_weight, 0.0f, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		trace_lambda.resize(reinterpret_cast<float*>(current_dev_main_ptr), real_size_weight, 0.0f, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(out_channels * sizeof(type_learnable_parameters));
		weights2.resize(reinterpret_cast<type_learnable_parameters*>(current_dev_main_ptr), out_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(out_channels * sizeof(float));
		hidden_state.resize(reinterpret_cast<float*>(current_dev_main_ptr), out_channels, 0, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(in_channels * sizeof(__half));
		hidden_pool.resize(reinterpret_cast<__half*>(current_dev_main_ptr), in_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		grad_weights.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_weights), total_size_grad_weights, MemoryType::Device);
		grad_biases.resize(reinterpret_cast<type_learnable_parameters*>(dev_side_ptr_biases), total_size_grad_biases, MemoryType::Device);
		if (isLoaded) loadFromFile(dir_path);
		else
		{
			generateNormalRandoms(weights1, real_size_weight, std::sqrtf(2.0f / in_channels), 0, 1234);
			generateNormalRandoms(weights2, out_channels, std::sqrtf(1.0f / out_channels), 0, 5678);
			weights_lambda.for_each_n(real_size_weight, LambdaWeightsOP<type_learnable_parameters>{weights_lambda.data(), real_size_weight}, 0);
		}
	}
	void forward(const uint32_t* input, const int8_t* input_scales)
	{
		hidden_pool.fill(0, hidden_pool.size(), 0, stream_forward);
		const int spatial_size = dimension_in * dimension_in;
		dim3 grid(((in_channels >> 5) + gpu_block_threads - 1) / gpu_block_threads, (spatial_size + 15) >> 4);
		LAUNCH_KERNEL(gapKernel<>, grid, gpu_block_threads, 0, stream_forward, input, input_scales, hidden_pool.data(), spatial_size, in_channels);
		const int complex_out_channels = out_channels >> 1;
		LAUNCH_KERNEL(criticFwd1Kernel<>, ((complex_out_channels << 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward, hidden_pool.data(), weights1.data(), weights_lambda.data(), hidden_state.data(), trace_w1.data(), trace_lambda.data(), complex_out_channels, in_channels);
		LAUNCH_KERNEL(criticFwd2Kernel<>, 1, complex_out_channels, 0, stream_forward, hidden_state.data(), weights2.data(), score_output.data(), complex_out_channels);
		checkCudaError(cudaStreamSynchronize(stream_forward));
	}
	void backward(float real_score)
	{
		const int complex_out_channels = out_channels >> 1;
		LAUNCH_KERNEL(criticApplyTracesKernel<>, ((real_size_weight >> 1) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, weights2.data(), trace_w1.data(), trace_lambda.data(), grad_weights.data(), grad_lambda.data(), score_output[0], real_score, complex_out_channels, in_channels);
		LAUNCH_KERNEL(criticOutputGradKernel<>, (complex_out_channels + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, hidden_state.data(), grad_biases.data(), score_output[0], real_score, complex_out_channels);
		const float sum_sq_w = grad_weights.transform_reduce<float>(SquareOp<type_learnable_parameters>{}, real_size_weight, stream_backward);
		const float sum_sq_b = grad_biases.transform_reduce<float>(SquareOp<type_learnable_parameters>{}, out_channels, stream_backward);
		const float sum_sq_l = grad_lambda.transform_reduce<float>(SquareOp<type_learnable_parameters>{}, real_size_weight, stream_backward);
		auto clip_op = [] (float sum_sq) -> float
		{
			const float norm = sqrtf(sum_sq + epsilon);
			return norm > 1.0f ? 1.0f / norm : 1.0f;
		};
		LAUNCH_KERNEL(updateParamsKernel<>, (real_size_weight + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, weights1.data(), grad_weights.data(), real_size_weight, clip_op(sum_sq_w), learning_rate, f_penalty);
		LAUNCH_KERNEL(updateParamsKernel<>, (out_channels + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, weights2.data(), grad_biases.data(), out_channels, clip_op(sum_sq_b), learning_rate, f_penalty);
		LAUNCH_KERNEL(updateParamsKernel<>, (real_size_weight + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward, weights_lambda.data(), grad_lambda.data(), real_size_weight, clip_op(sum_sq_l), learning_rate, f_penalty);
	}
	void reset(cudaStream_t stream_lnk = 0) 
    {
        hidden_state.fill(0, hidden_state.size(), 0, stream_lnk);
        trace_w1.fill(0, trace_w1.size(), 0.0f, stream_lnk);
        trace_lambda.fill(0, trace_lambda.size(), 0.0f, stream_lnk);
    }
};
