//cs.cu
#include "kernels.h"
#include <cuda_d3d11_interop.h>

struct ProcessBgraToRgbOP
{
	const uchar4* d_ptr_bgra;
	__half* d_rgb_ptr;
	const int plane_size;
	const uint32_t frame_seed;
	const bool is_train_mode;
	__device__ void operator()(int idx) const 
	{
		uchar4 bgra = d_ptr_bgra[idx];
		auto& bgr = reinterpret_cast<unsigned char (&)[4]>(bgra);
		if (is_train_mode) 
		{
			const uint32_t rand_val = getFastRandomBounded(frame_seed ^ idx, 10000);
			if (rand_val < 50)
			{
				#pragma unroll
				for (int i = 0; i < cs_channels - 1; ++i) bgr[i] = 0;
			}
			else if (rand_val < 100)
			{
				#pragma unroll
				for (int i = 0; i < cs_channels - 1; ++i) bgr[i] = 255;
			}
		}
		#pragma unroll
		for (int i = 0; i < cs_channels; ++i) d_rgb_ptr[idx + plane_size * i] = __float2half(bgr[i] * (1.0f / 255.0f));
	}
};

struct AugmentBrightnessContrastOP
{
	__half2* vec_image;
	const __half2 brightness;
	const __half2 contrast;
	__device__ void operator()(int idx) const 
	{
		constexpr __half2_raw h2_half = __half2_raw{0x3800, 0x3800};
		vec_image[idx] = __hmax2(__hmin2(__hfma2(__hsub2(vec_image[idx], h2_half), contrast, __hadd2(h2_half, brightness)), h2_one()), h2_zero());
	}
};

struct InternalCudaVisualState 
{
	static constexpr int screen_size = img_resolution * img_resolution;
	universal_vector<__half> screen_buffer[2];
	universal_vector<uchar4> d_raw_bgra;
	std::atomic<bool> current_screen_buffer{false};
	unique_stream stream_screen{nullptr};
	cudaGraphicsResource* dynamicRes = nullptr;
	InternalCudaVisualState(size_t* total_device_bytes_main)
	{
		*total_device_bytes_main = *total_device_bytes_main + align16(img_resolution * img_resolution * cs_channels * sizeof(__half)) * 2 + align16(screen_size * sizeof(uchar4));
		cudaStream_t temp_stream;
		checkCudaError(cudaStreamCreate(&temp_stream));
		stream_screen.reset(temp_stream);
	}
	void initData(uint8_t*& current_dev_main_ptr)
	{
		size_t w_bytes = align16(img_resolution * img_resolution * cs_channels * sizeof(__half));
		screen_buffer[0].resize(reinterpret_cast<__half*>(current_dev_main_ptr), img_resolution * img_resolution * cs_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		screen_buffer[1].resize(reinterpret_cast<__half*>(current_dev_main_ptr), img_resolution * img_resolution * cs_channels, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(screen_size * sizeof(uchar4));
		d_raw_bgra.resize(reinterpret_cast<uchar4*>(current_dev_main_ptr), screen_size, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
	}
};

extern "C" {
	void* create_cuda_state_visual(size_t* total_device_bytes_main) { return new InternalCudaVisualState(total_device_bytes_main); }
	void init_cuda_state_visual(void* state_ptr, uint8_t*& current_dev_main_ptr) { reinterpret_cast<InternalCudaVisualState*>(state_ptr)->initData(current_dev_main_ptr); }
	void init_cuda_shared_tex(void* state_ptr, ID3D11Texture2D* sharedTexture)
	{
		auto* state = reinterpret_cast<InternalCudaVisualState*>(state_ptr);
		checkCudaError(cudaGraphicsD3D11RegisterResource(&(state->dynamicRes), sharedTexture, cudaGraphicsRegisterFlagsNone));
	}
	void free_cuda_state_visual(void* state_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaVisualState*>(state_ptr);
		if (state->dynamicRes) checkCudaError(cudaGraphicsUnregisterResource(state->dynamicRes));
		delete state;
	}
	IDXGIAdapter* cudaFindAndSelectDevice_visual(IDXGIFactory1* dxgiFactory)
	{
		UINT adapterIndex = 0;
		IDXGIAdapter* currentAdapter = nullptr;
		while (dxgiFactory->EnumAdapters(adapterIndex, &currentAdapter) != DXGI_ERROR_NOT_FOUND) 
		{
			int cudaDev = -1;
			if (cudaD3D11GetDevice(&cudaDev, currentAdapter) == cudaSuccess) 
			{
				checkCudaError(cudaSetDevice(cudaDev));
				return currentAdapter;
			}
			currentAdapter->Release();
			adapterIndex++;
		}
		return nullptr;
	}
	void run_cuda_processing_visual(void* state_ptr, void* train_mode_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaVisualState*>(state_ptr);
		auto* train_mode = reinterpret_cast<std::atomic<E_Workmode>*>(train_mode_ptr);
		checkCudaError(cudaGraphicsMapResources(1, &(state->dynamicRes), state->stream_screen.get()));
		cudaArray_t cuArray = nullptr;
		checkCudaError(cudaGraphicsSubResourceGetMappedArray(&cuArray, state->dynamicRes, 0, 0));
		checkCudaError(cudaMemcpy2DFromArrayAsync(state->d_raw_bgra.data(), img_resolution * sizeof(uchar4), cuArray, 0, 0, img_resolution * sizeof(uchar4), img_resolution, cudaMemcpyDefault, state->stream_screen.get()));
		checkCudaError(cudaGraphicsUnmapResources(1, &(state->dynamicRes), state->stream_screen.get()));
		const bool is_train_mode = train_mode->load(std::memory_order_acquire) != E_Workmode::INFERENCE && train_mode->load(std::memory_order_acquire) != E_Workmode::INFERENCE_ON_TRAIN;
		const uchar4* d_ptr_bgra = state->d_raw_bgra.data();
		const int next_buf = !state->current_screen_buffer.load(std::memory_order_acquire);
		state->screen_buffer[next_buf].for_each_n(state->screen_size, ProcessBgraToRgbOP{d_ptr_bgra, state->screen_buffer[next_buf].data(), state->screen_size, generateUniqueSeed(), is_train_mode}, state->stream_screen.get());
		if (is_train_mode)
		{
			__half2* vec_image = reinterpret_cast<__half2*>(state->screen_buffer[next_buf].data());
			const __half2 brightness = __float2half2_rn(getFastRandomFloatBoundedNormal(generateUniqueSeed(), 0.15f));
			const __half2 contrast = __float2half2_rn(getFastRandomFloatBoundedNormal(generateUniqueSeed(), 0.2f) + 1.0f);
			state->screen_buffer[next_buf].for_each_n(state->screen_buffer[next_buf].size() >> 1, AugmentBrightnessContrastOP{vec_image, brightness, contrast}, state->stream_screen.get());
		}
		state->current_screen_buffer.store(next_buf, std::memory_order_release);
	}
	void copy_cuda_buffer_to_output_visual(void* state_ptr, void* result_vector_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaVisualState*>(state_ptr);
		auto* result = reinterpret_cast<universal_vector<__half>*>(result_vector_ptr);
		const int buf_idx = state->current_screen_buffer.load(std::memory_order_acquire);
		result->copy(state->screen_buffer[buf_idx], 0, state->screen_buffer[buf_idx].size(), 0, state->screen_buffer[buf_idx].size(), state->stream_screen.get());
		checkCudaError(cudaStreamSynchronize(state->stream_screen.get()));
	}
}
