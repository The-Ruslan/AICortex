//ca.cu
#include "kernels.h"

struct AudioProcessorOP
{
	__half* __restrict__ target_raw;
	const short* __restrict__ rag_raw;
	__device__ void operator()(size_t i) const 
	{
		const size_t offset = i << 3;
		const int4 chunk = *reinterpret_cast<const int4*>(rag_raw + offset);
		const auto& chunk_ptr = reinterpret_cast<const short2(&)[4]>(chunk);
		uint4 res_h;
		auto& res_h_ptr = reinterpret_cast<__half2(&)[4]>(res_h);
		const __half inv_scale = __float2half(1.0f / 32768.0f);
		#pragma unroll
		for(int j = 0; j < 4; j++) res_h_ptr[j] = __halves2half2(__hmul(__short2half_rn(chunk_ptr[j].x), inv_scale), __hmul(__short2half_rn(chunk_ptr[j].y), inv_scale));
		*reinterpret_cast<uint4*>(target_raw + offset) = res_h;
	}
};

struct InternalCudaAudioState
{
	static constexpr int BUFFER_SIZE_SAMPLES = 1024, HOP_SIZE_SAMPLES = BUFFER_SIZE_SAMPLES >> 2, RING_BUFFER_SIZE = BUFFER_SIZE_SAMPLES << 3, RING_BUFFER_MASK = RING_BUFFER_SIZE - 1;
	universal_vector<__half> audio_buffer[2], audio_buffer_fft[2], audio_cache;
	universal_vector<short> raw_audio_gpu;
	universal_vector<__half2> d_interim;
	std::atomic<bool> current_audio_buffer{false};
	unique_stream stream_audio{nullptr};
	std::vector<short> ring_buffer;
	size_t write_index = 0, read_index = 0, available_shorts = 0;
	InternalCudaAudioState(size_t* total_device_bytes_main)
	{
		*total_device_bytes_main = *total_device_bytes_main + align16(BUFFER_SIZE_SAMPLES * 2 * sizeof(short)) +
									align16(BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES * sizeof(__half)) +
									align16(BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES * sizeof(__half2)) +
									align16(BUFFER_SIZE_SAMPLES * sizeof(__half)) * 2 +
									align16(img_resolution * img_resolution * sizeof(__half) * 2) * 2;
		cudaStream_t temp_stream;
		checkCudaError(cudaStreamCreate(&temp_stream));
		stream_audio.reset(temp_stream);
		ring_buffer.resize(RING_BUFFER_SIZE, 0);
	}
	void initData(uint8_t*& current_dev_main_ptr)
	{
		size_t w_bytes = align16(BUFFER_SIZE_SAMPLES * 2 * sizeof(short));
		raw_audio_gpu.resize(reinterpret_cast<short*>(current_dev_main_ptr), BUFFER_SIZE_SAMPLES * 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES * sizeof(__half));
		audio_cache.resize(reinterpret_cast<__half*>(current_dev_main_ptr), BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES * sizeof(__half2));
		d_interim.resize(reinterpret_cast<__half2*>(current_dev_main_ptr), BUFFER_SIZE_SAMPLES * BUFFER_SIZE_SAMPLES, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(BUFFER_SIZE_SAMPLES * sizeof(__half));
		audio_buffer[0].resize(reinterpret_cast<__half*>(current_dev_main_ptr), BUFFER_SIZE_SAMPLES, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		audio_buffer[1].resize(reinterpret_cast<__half*>(current_dev_main_ptr), BUFFER_SIZE_SAMPLES, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		w_bytes = align16(img_resolution * img_resolution * sizeof(__half) * 2);
		audio_buffer_fft[0].resize(reinterpret_cast<__half*>(current_dev_main_ptr), img_resolution * img_resolution * 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
		audio_buffer_fft[1].resize(reinterpret_cast<__half*>(current_dev_main_ptr), img_resolution * img_resolution * 2, MemoryType::Device);
		current_dev_main_ptr += w_bytes;
	}
};

extern "C" {
	void* create_cuda_state_audio(size_t* total_device_bytes_main) { return new InternalCudaAudioState(total_device_bytes_main); }
	void init_cuda_state_audio(void* state_ptr, uint8_t*& current_dev_main_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaAudioState*>(state_ptr);
		state->initData(current_dev_main_ptr);
	}
	void free_cuda_state_audio(void* state_ptr) { delete reinterpret_cast<InternalCudaAudioState*>(state_ptr); }
	void run_cuda_processing_audio(void* state_ptr, BYTE* pData, uint32_t numFramesAvailable, DWORD flags, uint32_t packetLength)
	{
		auto* state = reinterpret_cast<InternalCudaAudioState*>(state_ptr);
		size_t new_shorts_count = numFramesAvailable * 2;
		short* pShortData = reinterpret_cast<short*>(pData);
		size_t space_till_end = InternalCudaAudioState::RING_BUFFER_SIZE - state->write_index;
		if (new_shorts_count <= space_till_end)
		{
			std::copy(pShortData, pShortData + new_shorts_count, state->ring_buffer.begin() + state->write_index);
			state->write_index = (state->write_index + new_shorts_count) & InternalCudaAudioState::RING_BUFFER_MASK;
		}
		else
		{
			std::copy(pShortData, pShortData + space_till_end, state->ring_buffer.begin() + state->write_index);
			std::copy(pShortData + space_till_end, pShortData + new_shorts_count, state->ring_buffer.begin());
			state->write_index = new_shorts_count - space_till_end;
		}
		state->available_shorts += new_shorts_count;
		constexpr size_t REQUIRED_SHORTS = InternalCudaAudioState::BUFFER_SIZE_SAMPLES * 2, HOP_SHORTS = InternalCudaAudioState::HOP_SIZE_SAMPLES * 2;
		if (state->available_shorts < REQUIRED_SHORTS) return;
		const int next_buf = !state->current_audio_buffer.load(std::memory_order_acquire);
		size_t chunk1_size = REQUIRED_SHORTS, chunk2_size = 0;
		if (state->read_index + REQUIRED_SHORTS > InternalCudaAudioState::RING_BUFFER_SIZE)
		{
			chunk1_size = InternalCudaAudioState::RING_BUFFER_SIZE - state->read_index;
			chunk2_size = REQUIRED_SHORTS - chunk1_size;
		}
		checkCudaError(cudaMemcpyAsync(state->raw_audio_gpu.data(), state->ring_buffer.data() + state->read_index, chunk1_size * sizeof(short), cudaMemcpyDefault, state->stream_audio.get()));
		if (chunk2_size > 0) checkCudaError(cudaMemcpyAsync(state->raw_audio_gpu.data() + chunk1_size, state->ring_buffer.data(), chunk2_size * sizeof(short), cudaMemcpyDefault, state->stream_audio.get()));
		const short* __restrict__ rag_raw = state->raw_audio_gpu.data();
		state->audio_buffer[next_buf].for_each_n(REQUIRED_SHORTS >> 3, AudioProcessorOP{state->audio_buffer[next_buf].data(), rag_raw}, state->stream_audio.get());
		dim3 blockSize(16, 16);
		dim3 gridSize(InternalCudaAudioState::BUFFER_SIZE_SAMPLES / (blockSize.x << 1), InternalCudaAudioState::BUFFER_SIZE_SAMPLES / blockSize.y, 1);
		LAUNCH_KERNEL(resizeKernel<>, gridSize, blockSize, 0, state->stream_audio.get(), state->audio_buffer[next_buf].data(), state->audio_cache.data(), state->audio_buffer[next_buf].size(), 1, InternalCudaAudioState::BUFFER_SIZE_SAMPLES, 1);
		LAUNCH_KERNEL(fftRowKernel<InternalCudaAudioState::BUFFER_SIZE_SAMPLES>, InternalCudaAudioState::BUFFER_SIZE_SAMPLES, gpu_block_threads, 0, state->stream_audio.get(), state->audio_cache.data(), state->d_interim.data());
		LAUNCH_KERNEL(fftColumnKernel<InternalCudaAudioState::BUFFER_SIZE_SAMPLES>, InternalCudaAudioState::BUFFER_SIZE_SAMPLES, gpu_block_threads, 0, state->stream_audio.get(), state->d_interim.data(), reinterpret_cast<__half*>(state->d_interim.data()));
		dim3 resizeGridSize(img_resolution / (blockSize.x << 1), img_resolution / blockSize.y, 2);
		LAUNCH_KERNEL(resizeKernel<>, resizeGridSize, blockSize, 0, state->stream_audio.get(), reinterpret_cast<__half*>(state->d_interim.data()), state->audio_buffer_fft[next_buf].data(), InternalCudaAudioState::BUFFER_SIZE_SAMPLES, InternalCudaAudioState::BUFFER_SIZE_SAMPLES, img_resolution, 2);
		state->current_audio_buffer.store(next_buf, std::memory_order_release);
		state->read_index = (state->read_index + HOP_SHORTS) & InternalCudaAudioState::RING_BUFFER_MASK;
		state->available_shorts -= HOP_SHORTS;
	}
	void copy_cuda_buffer_to_output_audio(void* state_ptr, void* result_vector_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaAudioState*>(state_ptr);
		auto* result = reinterpret_cast<universal_vector<__half>*>(result_vector_ptr);
		const int buf_idx = state->current_audio_buffer.load(std::memory_order_acquire);
		result->copy(state->audio_buffer_fft[buf_idx], img_resolution * img_resolution * cs_channels, result->size(), 0, state->audio_buffer_fft[buf_idx].size(), state->stream_audio.get());
		checkCudaError(cudaStreamSynchronize(state->stream_audio.get()));
	}
	void fill_buffer_zero_audio(void* state_ptr)
	{
		auto* state = reinterpret_cast<InternalCudaAudioState*>(state_ptr);
		const int buf_idx = state->current_audio_buffer.load(std::memory_order_acquire);
		state->audio_buffer[buf_idx].fill(0, state->audio_buffer[buf_idx].size(), h_zero(), state->stream_audio.get());
	}
	void cudaSetDevice_audio()
	{
		checkCudaError(cudaSetDevice(0));
	}
}
