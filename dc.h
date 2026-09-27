//dc.h
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <cassert>

#ifdef DEBUG
	#define LAUNCH_KERNEL(kernel, blocks, threads, shared_mem_size, stream, ...) \
		do { \
			kernel<<<blocks, threads, shared_mem_size, stream>>>(__VA_ARGS__); \
			cudaStreamSynchronize(stream); \
			cudaError_t err = cudaGetLastError(); \
			if (err != cudaSuccess) \
			{ \
				fprintf(stderr, "CUDA kernel launch error in %s at %s:%d\n", #kernel, __FILE__, __LINE__); \
				fprintf(stderr, "CUDA kernel error description: %s\n", cudaGetErrorString(err)); \
				throw std::runtime_error("CUDA kernel error code: " + std::to_string(err)); \
			} \
		} while(0)
#else
	#define LAUNCH_KERNEL(kernel, blocks, threads, shared_mem_size, stream, ...) \
		kernel<<<blocks, threads, shared_mem_size, stream>>>(__VA_ARGS__)
#endif

inline void checkCudaError(cudaError_t err)
{
	if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorContextIsDestroyed) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
}

template <int dummy = 0>
struct CudaStreamDeleter
{
	void operator()(cudaStream_t stream) const
	{
		if (stream)
		{
			checkCudaError(cudaStreamSynchronize(stream));
			checkCudaError(cudaStreamDestroy(stream));
		}
	}
};
using unique_stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter<>>;

inline size_t align16(size_t bytes) { return (bytes + 15) & ~15; };

struct SquareOp
{
    __device__ __forceinline__ float operator()(const __half x) const
	{
        const float val = __half2float(x);
        return val * val;
    }
};

inline void printGpuMem(const std::string& step_name)
{
	size_t free_bytes = 0, total_bytes = 0;
	cudaDeviceSynchronize();
	cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (err != cudaSuccess)
	{
		std::cout << "[MEM ERROR] " << step_name << " failed to get info: " << cudaGetErrorString(err) << std::endl;
		return;
	}
	const float total_mb = total_bytes / (1024.0f * 1024.0f);
	std::cout << "[VRAM LOG] " << step_name << " -> Used: " << total_mb - free_bytes / (1024.0f * 1024.0f) << " MB / Total: " << total_mb << " MB" << std::endl;
}
