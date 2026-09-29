//hos.h
#pragma once
#include "kernels.h"
#include "exp.h"

struct EntropyOP
{
	__device__ __forceinline__ float operator()(const __half p_half) const
	{
		const float p = __half2float(p_half), q = 1.0f - p;
		return -1.442695f * fmaf(p, __logf(fmaxf(p, epsilon)), q * __logf(fmaxf(q, epsilon)));
	}
};

class HallOfShame
{
private:
    const size_t exp_buffer_size_total, exp_main_range;
	std::vector<S_Experience> entries, history_buffer;
	universal_vector<float> loss;
	size_t min_group_start = 0, history_index = 0;
	float avg_ent = 1.0f, avg_huber = 0.01f, avg_shame_rolling = 1.0f;
	static constexpr float momentum = 0.99f;
	bool history_ready = false;
	float get_group_total_shame(size_t start_idx)
	{
		float total = 0.0f;
		for (size_t k = 0; k < meta_seq_len; ++k) total += entries[start_idx + k].shame_score;
		return total;
	}
public:
	float getHuberDelta() const { return sqrtf(avg_huber + epsilon); }
	HallOfShame(size_t exp_mem_size_total, size_t exp_buffer_size_active) :
		exp_buffer_size_total(exp_mem_size_total),
		exp_main_range(exp_buffer_size_total - exp_buffer_size_active),
		entries(exp_buffer_size_total),
		history_buffer(meta_seq_len),
		loss(1, MemoryType::PinnedHost) {}
	void initData(uint8_t*& current_host_main_ptr, int blocks_data_size)
	{
		for (auto& entry : entries)
		{
			size_t w_bytes = align16(blocks_data_size * sizeof(uint32_t) << 2);
			entry.blocks_data.resize(reinterpret_cast<uint32_t*>(current_host_main_ptr), blocks_data_size << 2, MemoryType::PinnedHost);
			current_host_main_ptr += w_bytes;
			w_bytes = align16(blocks_data_size * sizeof(int8_t));
			entry.blocks_data_scales.resize(reinterpret_cast<int8_t*>(current_host_main_ptr), blocks_data_size, MemoryType::PinnedHost);
			current_host_main_ptr += w_bytes;
		}
		for (size_t i = 0; i < history_buffer.size(); i++)
		{
			auto& entry = entries[exp_main_range + i];
			auto& entry_hb = history_buffer[i];
			entry_hb.blocks_data.resize(entry.blocks_data.data(), entry.blocks_data.size(), MemoryType::PinnedHost);
			entry_hb.blocks_data_scales.resize(entry.blocks_data_scales.data(), entry.blocks_data_scales.size(), MemoryType::PinnedHost);
		}
	}
	void addEntry(const S_Experience& exp, universal_vector<__half>& output_fwd, universal_vector<__half>& probs_fwd, universal_vector<__half>& output_predictor, cudaStream_t stream = 0)
	{
		const float entropy_sum = probs_fwd.transform_reduce<float>(EntropyOP{}, probs_fwd.size(), stream) / probs_fwd.size();
		loss[0] = 0.0f;
		LAUNCH_KERNEL(huberKernel<>, (((output_predictor.size() + 7) >> 3) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream, output_predictor.data(), output_fwd.data(), loss.data(), output_predictor.size(), getHuberDelta());
		checkCudaError(cudaStreamSynchronize(stream));
		const float predict_err = *const_cast<volatile float*>(loss.data()) / output_fwd.size();
		avg_ent = fmaf(momentum, avg_ent, (1.0f - momentum) * entropy_sum);
		avg_huber = fmaf(momentum, avg_huber, (1.0f - momentum) * predict_err);
		const float shame_score = entropy_sum / (avg_ent + epsilon) + predict_err / (avg_huber + epsilon);
		history_buffer[history_index](exp, stream);
		history_buffer[history_index].shame_score = shame_score;
	}
	void updateEntries(cudaStream_t stream = 0)
	{
		const float shame_score = history_buffer[history_index].shame_score;
		avg_shame_rolling = fmaf(momentum, avg_shame_rolling, (1.0f - momentum) * shame_score);
		history_buffer[history_index].past_pass_count = 1;
		history_index = (history_index + 1) & (meta_seq_len - 1);
		if (history_index == 0) history_ready = true;
		if (!history_ready) return;
		float current_history_total_shame = 0.0f;
		for (size_t i = 0; i < meta_seq_len; ++i) current_history_total_shame += history_buffer[i].shame_score;
		if (current_history_total_shame <= get_group_total_shame(min_group_start) || shame_score <= avg_shame_rolling * 1.1f) return;
		for (size_t i = 0; i < exp_main_range; i++) entries[i].past_pass_count++;
		size_t read_ptr = history_index;
		for (size_t i = 0; i < meta_seq_len; ++i)
		{
			entries[min_group_start + i](history_buffer[read_ptr], stream);
			read_ptr = (read_ptr + 1) & (meta_seq_len - 1);
		}
		float min_group_shame = get_group_total_shame(0);
		size_t worst_group_start = 0;
		for (size_t i = 0; i < exp_main_range; i += meta_seq_len)
		{
			float current_group_shame = get_group_total_shame(i);
			if (current_group_shame < min_group_shame)
			{
				min_group_shame = current_group_shame;
				worst_group_start = i;
			}
		}
		min_group_start = worst_group_start;
	}
	void getEntries(std::vector<S_Experience*>& result)
	{
		result.reserve(entries.size());
		std::vector<S_Experience*> preresult(meta_seq_len);
		for (size_t i = 0; i < entries.size(); i += meta_seq_len)
		{
			auto& entry = entries[i];
			bool full_entried = true;
			for (size_t j = 0; j < meta_seq_len; j++)
			{
				if (entry.shame_score > fminimum) preresult[j] = &entry;
				else
				{
					full_entried = false;
					break;
				}
			}
			if (full_entried) for (size_t j = 0; j < meta_seq_len; j++) result.push_back(preresult[j]);
		}
	}
};
