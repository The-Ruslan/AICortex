#pragma once
#include <algorithm>
#include "dc.h"
#include "bc.h"

template<typename T>
__global__ void fillUniversalVectorKernel(T* d_out, T value, size_t size) { for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) d_out[idx] = value; }
template<typename Tin, typename Tout, typename Op>
__global__ void transformReduceKernel(const Tin* d_in, Tout* d_global_res, Op op, size_t size)
{
	Tout sum = 0;
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) sum += op(d_in[i]);
	for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
	__shared__ Tout s_warp_sums[32]; 
	const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
	if (lane == 0) s_warp_sums[wid] = sum;
	__syncthreads();
	if (wid == 0)
	{
		sum = threadIdx.x < (blockDim.x >> 5) ? s_warp_sums[lane] : 0;
		for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
	}
	if (threadIdx.x == 0) atomicAdd(d_global_res, sum);
}
template<typename Op>
__global__ void forEachNUniversalVectorKernel(Op op, size_t n) { for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) op(i); }

enum class VectorMode { Allocation, View };
enum class MemoryType { Device, PinnedHost, Default };

template<typename T>
class universal_vector
{
private:
	T* m_data = nullptr;
	size_t m_size = 0, m_capacity = 0;
	MemoryType m_mem_type = MemoryType::Device;
	VectorMode m_mode = VectorMode::Allocation;
	template<typename KernelPtr>
    struct SizeCache { static size_t* get() { static size_t instance = 0; return &instance; } };
public:
	using value_type = T;
	template<typename... Args>
	size_t get_optimal_blocks(void (*kernel)(Args...)) const
	{
		auto* cached_blocks = SizeCache<void(*)(Args...)>::get();
		if (*cached_blocks == 0)
		{
			int device_id = -1, num_sm = 0, max_threads_per_sm = 0;
			checkCudaError(cudaGetDevice(&device_id));
			checkCudaError(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_id));
			checkCudaError(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
			size_t blocks_per_sm = max_threads_per_sm / gpu_block_threads;
			if (blocks_per_sm < 1) blocks_per_sm = 1;
			*cached_blocks = num_sm * blocks_per_sm;
		}
		return *cached_blocks;
	}
	MemoryType alloc_type() const { return m_mem_type; }
	VectorMode alloc_mode() const { return m_mode; }
	bool empty() const { return m_size == 0; }
	T* data() { return m_data; }
	const T* data() const { return m_data; }
	size_t size() const { return m_size; }
	size_t capacity() const { return m_capacity; }
	T& operator[](size_t idx) { return m_data[idx]; }
	const T& operator[](size_t idx) const { return m_data[idx]; }
	T* begin() { return m_data; }
	T* end() { return m_data + m_size; }
	template<typename U>
	void fill(size_t start, size_t end, U value, cudaStream_t stream = 0)
	{
		if (m_size == 0 || start >= end) return;
		const size_t final_end = end < m_size ? end : m_size;
		const size_t fill_elements = final_end - start;
		if (fill_elements == 0) return;
		if (m_mem_type == MemoryType::Device)
		{
			const size_t blocks = std::max<size_t>(1, std::min(get_optimal_blocks(fillUniversalVectorKernel<T>), (fill_elements + gpu_block_threads - 1) / gpu_block_threads));
			LAUNCH_KERNEL(fillUniversalVectorKernel<T>, blocks, gpu_block_threads, 0, stream, m_data + start, static_cast<T>(value), fill_elements);
		}
		else
		{
			if (stream != 0) checkCudaError(cudaStreamSynchronize(stream));
			std::fill(m_data + start, m_data + final_end, static_cast<T>(value));
		}
	}
	void copy(const universal_vector<T>& src, size_t dest_begin, size_t dest_end, size_t src_begin, size_t src_end, cudaStream_t stream = 0)
	{
		if (src.empty() || m_size == 0 || dest_begin >= m_size || src_begin >= src.size() || dest_begin >= dest_end || src_begin >= src_end) return;
		const size_t dest_len = (dest_end > m_size ? m_size : dest_end) - dest_begin;
		const size_t src_len = (src_end > src.size() ? src.size() : src_end) - src_begin;
		const size_t copy_elements = dest_len < src_len ? dest_len : src_len;
		if (copy_elements == 0) return; 
		if (m_mem_type == MemoryType::Device || src.alloc_type() == MemoryType::Device) checkCudaError(cudaMemcpyAsync(m_data + dest_begin, src.data() + src_begin, copy_elements * sizeof(T), cudaMemcpyDefault, stream));
		else
		{
			if (stream != 0) checkCudaError(cudaStreamSynchronize(stream));
			std::copy(src.data() + src_begin, src.data() + src_begin + copy_elements, m_data + dest_begin);
		}
	}
	void clear()
	{
		if (m_data && m_mode == VectorMode::Allocation)
		{
			if (m_mem_type == MemoryType::Device) checkCudaError(cudaFree(m_data));
			else checkCudaError(cudaFreeHost(m_data));
		}
		m_data = nullptr;
		m_size = 0;
		m_capacity = 0;
	}
	void reserve(size_t new_capacity, MemoryType mem_type = MemoryType::Default)
	{
		if (new_capacity <= m_capacity || m_mode == VectorMode::View) return;
		const size_t bytes = new_capacity * sizeof(T);
		T* new_data = nullptr;
		MemoryType old_mem_type = m_mem_type; 
		if (mem_type != MemoryType::Default) m_mem_type = mem_type;
		else m_mem_type = MemoryType::Device;
		if (m_mem_type == MemoryType::Device) checkCudaError(cudaMalloc(&new_data, bytes));
		else checkCudaError(cudaHostAlloc(&new_data, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
		if (m_size > 0) checkCudaError(cudaMemcpy(new_data, m_data, m_size * sizeof(T), cudaMemcpyDefault));
		if (m_data)
		{
			if (old_mem_type == MemoryType::Device) checkCudaError(cudaFree(m_data));
			else checkCudaError(cudaFreeHost(m_data));
		}
		m_data = new_data;
		m_capacity = new_capacity;
	}
	void resize(size_t new_size, MemoryType mem_type = MemoryType::Default)
	{
		if (m_size == new_size || m_mode == VectorMode::View) return;
		if (new_size > m_capacity) reserve(new_size, mem_type);
		m_size = new_size;
	}
	template<typename U>
	void resize(size_t new_size, U value, MemoryType mem_type = MemoryType::Default)
	{
		if (m_size == new_size || m_mode == VectorMode::View) return;
		if (new_size > m_capacity) reserve(new_size, mem_type);
		m_size = new_size;
		if (m_size > 0) fill(0, m_size, value);
	}
	void resize(T* external_ptr, size_t new_size, MemoryType mem_type = MemoryType::Default)
	{
		m_data = external_ptr;
		m_size = new_size;
		m_capacity = m_size;
		m_mem_type = mem_type;
		m_mode = VectorMode::View;
	}
	template<typename U>
	void resize(T* external_ptr, size_t new_size, U value, MemoryType mem_type = MemoryType::Default)
	{
		m_data = external_ptr;
		m_size = new_size;
		m_capacity = m_size;
		m_mem_type = mem_type;
		m_mode = VectorMode::View;
		if (m_size > 0) fill(0, m_size, value);
	}
	universal_vector() = default;
	universal_vector(size_t new_size, MemoryType mem_type = MemoryType::Device) { resize(new_size, mem_type); }
	universal_vector(size_t new_size, T value, MemoryType mem_type = MemoryType::Device) { resize(new_size, value, mem_type); }
	universal_vector(T* external_ptr, size_t new_size, MemoryType mem_type) { resize(external_ptr, new_size, mem_type); }
	universal_vector(T* external_ptr, size_t new_size, T value, MemoryType mem_type) { resize(external_ptr, new_size, value, mem_type); }
	~universal_vector() { clear(); }
	universal_vector(const universal_vector& other)
	{
		m_size = other.m_size;
		m_capacity = other.m_capacity;
		m_mem_type = other.m_mem_type;
		m_mode = other.m_mode;
		if (other.m_data != nullptr && m_size > 0 && m_mode == VectorMode::Allocation)
		{
			const size_t bytes = m_capacity * sizeof(T);
			if (m_mem_type == MemoryType::Device) checkCudaError(cudaMalloc(&m_data, bytes));
			else checkCudaError(cudaHostAlloc(&m_data, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
			checkCudaError(cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDefault));
		}
		else m_data = other.m_data;
	}
	universal_vector(universal_vector&& other) noexcept : m_data(other.m_data), m_size(other.m_size), m_capacity(other.m_capacity), m_mem_type(other.m_mem_type), m_mode(other.m_mode)
	{
		other.m_data = nullptr;
		other.m_size = 0;
		other.m_capacity = 0;
	}
	universal_vector& operator=(const universal_vector& other)
	{
		if (this != &other)
		{
			clear();
			m_size = other.m_size;
			m_capacity = other.m_capacity;
			m_mem_type = other.m_mem_type;
			m_mode = other.m_mode;
			if (other.m_data != nullptr && m_size > 0 && m_mode == VectorMode::Allocation)
			{
				const size_t bytes = m_capacity * sizeof(T);
				if (m_mem_type == MemoryType::Device) checkCudaError(cudaMalloc(&m_data, bytes));
				else checkCudaError(cudaHostAlloc(&m_data, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
				checkCudaError(cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDefault));
			}
			else m_data = other.m_data;
		}
		return *this;
	}
	universal_vector& operator=(universal_vector&& other) noexcept
	{
		if (this != &other)
		{
			clear();
			m_data = other.m_data;
			m_size = other.m_size;
			m_capacity = other.m_capacity;
			m_mem_type = other.m_mem_type;
			m_mode = other.m_mode;
			other.m_data = nullptr;
			other.m_size = 0;
			other.m_capacity = 0;
		}
		return *this;
	}
	template<typename Tout, typename Op>
	Tout transform_reduce(Op op, size_t end, cudaStream_t stream = 0) const
	{
		if (m_size == 0 || end > m_size) return 0;
		struct DeviceResultCache
		{
			Tout* d_ptr = nullptr;
			DeviceResultCache() { checkCudaError(cudaMalloc(&d_ptr, sizeof(Tout))); }
			~DeviceResultCache() { if (d_ptr) cudaFree(d_ptr); }
		};
		static DeviceResultCache cache; 
		checkCudaError(cudaMemsetAsync(cache.d_ptr, 0, sizeof(Tout), stream));
		const size_t final_size = end < m_size ? end : m_size;
		const size_t blocks = std::max<size_t>(1, std::min(get_optimal_blocks(transformReduceKernel<T, Tout, Op>), (final_size + gpu_block_threads - 1) / gpu_block_threads));
		LAUNCH_KERNEL((transformReduceKernel<T, Tout, Op>), blocks, gpu_block_threads, 0, stream, m_data, cache.d_ptr, op, final_size);
		checkCudaError(cudaStreamSynchronize(stream));
		Tout host_result;
		checkCudaError(cudaMemcpy(&host_result, cache.d_ptr, sizeof(Tout), cudaMemcpyDefault));
		return host_result;
	}
	template<typename Op>
	void for_each_n(size_t n, Op op, cudaStream_t stream = 0)
	{
		if (n == 0) return;
		if (m_mem_type == MemoryType::Device)
		{
			const size_t blocks = std::max<size_t>(1, std::min(get_optimal_blocks(forEachNUniversalVectorKernel<Op>), (n + gpu_block_threads - 1) / gpu_block_threads));
			LAUNCH_KERNEL(forEachNUniversalVectorKernel<Op>, blocks, gpu_block_threads, 0, stream, op, n);
		}
		else
		{
			if (stream != 0) checkCudaError(cudaStreamSynchronize(stream));
			#pragma omp parallel for schedule(guided, 1024) if(n >= 1024)
			for (size_t i = 0; i < n; ++i) op(i);
		}
	}
};
