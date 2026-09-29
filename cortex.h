//cortex.h
#pragma once
#include "block.h"
#include "mb.h"
#include "cs.h"
#include "ca.h"
#include "hos.h"
#include "critic.h"
#include "cc.h"

inline constexpr float constexpr_pow(float base, int exp)
{
    float result = 1.0f;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

struct BackwardStepOP
{
	__half* output_grad_ptr;
	const __half* exp_output_ptr;
	const action_type* exp_action_ptr;
	const float score_val;
	__device__ void operator()(size_t i) const
	{
		size_t target_idx = 0;
		int shift = 0;
		if (i < total_bits_mouse)
		{
			target_idx = 0;
			shift = i;
		}
		else
		{
			const size_t k = i - total_bits_mouse;
			if (k < total_bits_difference)
			{
				shift = k + total_bits_mouse;
				target_idx = 0;
			}
			else
			{
				const size_t k_adj = k - total_bits_difference;
				target_idx = k_adj / total_bits_per_element + 1;
				shift = k_adj & (total_bits_per_element - 1);
			}
		}
		output_grad_ptr[i] = __float2half(score_val * ((test_bit(exp_action_ptr[target_idx], shift) ? 1.0f : 0.0f) - __half2float(exp_output_ptr[i])));
	}
};
struct BackwardMetaOP
{
	__half* __restrict__ g_sum;
	__half** __restrict__ a_ptrs;
	const __half* __restrict__ b_raw;
	const size_t* __restrict__ topo_offsets;
	const size_t* __restrict__ topo_sizes;
	const __half2 factor2;
	__device__ void operator()(size_t tensor_idx) const
	{
		__half* __restrict__ a_base_ptr = a_ptrs[tensor_idx];
		const size_t global_start_offset = topo_offsets[tensor_idx];
		const size_t num_u4 = topo_sizes[tensor_idx] >> 3;
		#pragma unroll 2
		for (size_t u4_idx = 0; u4_idx < num_u4; ++u4_idx)
		{
			const size_t local_idx = u4_idx << 3;
			const size_t global_idx = global_start_offset + local_idx;
			const uint4 a_val = *reinterpret_cast<const uint4*>(a_base_ptr + local_idx);
			const auto& a_h2 = reinterpret_cast<const __half2(&)[4]>(a_val);
			const uint4 b_val = *reinterpret_cast<const uint4*>(b_raw + global_idx);
			const auto& b_h2 = reinterpret_cast<const __half2(&)[4]>(b_val);
			uint4* __restrict__ mgs_ptr = reinterpret_cast<uint4*>(g_sum + global_idx);
			uint4 raw_sum = *mgs_ptr;
			auto& h2_sum = reinterpret_cast<__half2(&)[4]>(raw_sum);
			#pragma unroll
			for(int j = 0; j < 4; j++) h2_sum[j] = __hfma2(__hfma2(a_h2[j], __half2_raw{0x068E, 0x068E}, __hsub2(a_h2[j], b_h2[j])), factor2, h2_sum[j]);
			*mgs_ptr = raw_sum;
		}
	}
};
struct BackwardScaleOP
{
	__half** __restrict__ target_raw;
	const __half* __restrict__ a_raw;
	const size_t* __restrict__ topo_offsets;
	const size_t* __restrict__ topo_sizes;
	const __half2 meta_scale2;
	const __half2 meta_lr2_inv;
	__device__ void operator()(size_t tensor_idx) const
	{
		__half* __restrict__ layer_target_ptr = target_raw[tensor_idx];
		const size_t global_start_offset = topo_offsets[tensor_idx];
		const size_t num_u4 = topo_sizes[tensor_idx] >> 3;
		#pragma unroll 2
		for (size_t u4_idx = 0; u4_idx < num_u4; ++u4_idx)
		{
			const size_t local_idx = u4_idx << 3;
			uint4* __restrict__ layer_w_v4_ptr = reinterpret_cast<uint4*>(layer_target_ptr + local_idx);
			uint4 raw_w = *layer_w_v4_ptr;
			auto& h2_w = reinterpret_cast<__half2(&)[4]>(raw_w);
			const uint4 raw_sum = *reinterpret_cast<const uint4*>(a_raw + global_start_offset + local_idx);
			const auto& h2_sum = reinterpret_cast<const __half2(&)[4]>(raw_sum);
			#pragma unroll
			for (int j = 0; j < 4; ++j) h2_w[j] = __hfma2(__hmul2(h2_sum[j], meta_scale2), meta_lr2_inv, h2_w[j]);
			*layer_w_v4_ptr = raw_w;
		}
	}
};

class Cortex
{
private:
	inline bool get_gen_value(float survival_prob = 1.0f)
	{
		if (survival_prob >= 1.0f) return true;
		if (survival_prob <= 0.0f) return false;
		return getFastRandomFloatBoundedUniform(generateUniqueSeed(), 1.0f) < survival_prob;
    }
	int fps, pipo_db = 0, pipo_db_num_groups = 0;
	S_Experience exp;
	std::vector<std::unique_ptr<Block>> blocks;
	std::unique_ptr<CaptureScreen> screenCapture;
	std::unique_ptr<CaptureAudio> audioCapture;
	std::unique_ptr<MambaBlock> mamba, predictor;
	std::unique_ptr<HallOfShame> hall_of_shame;
	std::unique_ptr<Critic> critic;
	std::unique_ptr<ConvergenceController> convergence_controller;
	unique_stream stream_forward{nullptr};
	unique_stream stream_backward{nullptr};
	unique_stream stream_exp{nullptr};
	universal_vector<uint8_t> global_device_main_arena, global_device_side_arena, global_host_main_arena, global_host_side_arena;
	universal_vector<__half> raw_visaud_fwd, output_fwd, probs_fwd, output_predictor, probs_predictor, grad_output, grad_input, meta_backup_data, meta_gradient_sum;
	universal_vector<uint32_t> blocks_data;
	universal_vector<int8_t> blocks_data_scales;
	universal_vector<action_type> action_ptr, action_prevs;
	universal_vector<__half*> meta_adapted_data;
	universal_vector<size_t> meta_adapted_offsets, meta_global_offsets;
	const std::string dir_path;
    std::thread addExpThread;
	std::atomic<E_Workmode> &work_mode;
	std::atomic<bool> &isEliminating, &is_inputing;
	std::atomic<float> &isScored;
	std::mutex &mtx_exp, &mtx_running;
	std::condition_variable &cv_exp, &cv_running;
	bool &exp_add, ready_to_receive_exp = false, &is_running, ready_to_update = false;
	std::vector<uint8_t> &keybinds;
	__half learning_rate;
	E_BackwardStrategy next_backward_strategy = E_BackwardStrategy::STANDARD;
	std::atomic<bool> exp_is_dirty{false};
	void executeActions()
	{
		std::vector<INPUT> inputs;
		inputs.reserve(action_ptr.size() * total_bits_per_element - 32);
		const action_type& mouse_action = action_ptr[0];
		const action_type& mouse_prev = action_prevs[0];
		if (test_bit(mouse_action, 0) != test_bit(mouse_prev, 0))
		{
			INPUT inp = {};
			inp.type = INPUT_MOUSE;
			inp.mi.dwFlags = test_bit(mouse_action, 0) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
			inputs.push_back(inp);
		}
		if (test_bit(mouse_action, 1) != test_bit(mouse_prev, 1))
		{
			INPUT inp = {};
			inp.type = INPUT_MOUSE;
			inp.mi.dwFlags = test_bit(mouse_action, 1) ? MOUSEEVENTF_RIGHTDOWN : MOUSEEVENTF_RIGHTUP;
			inputs.push_back(inp);
		}
		const short dx = static_cast<short>((mouse_action >> 2) & 0xFFFF);
		const short dy = static_cast<short>((mouse_action >> 18) & 0xFFFF);
		if (dx != 0 || dy != 0)
		{
			INPUT inp = {};
			inp.type = INPUT_MOUSE;
			inp.mi.dx = dx;
			inp.mi.dy = dy;
			inp.mi.dwFlags = MOUSEEVENTF_MOVE;
			inputs.push_back(inp);
		}
		for (size_t k = 0; k < keybinds.size(); ++k)
		{
			size_t target_idx = 0, shift = 0;
			if (k < total_bits_difference) shift = k + total_bits_mouse;
			else
			{
				const size_t k_adj = k - total_bits_difference;
				target_idx = k_adj / total_bits_per_element + 1;
				shift = k_adj & (total_bits_per_element - 1);
			}
			const bool current = (action_ptr[target_idx] >> shift) & 1;
			if (current != ((action_prevs[target_idx] >> shift) & 1))
			{
				INPUT inp = {};
				inp.type = INPUT_KEYBOARD;
				inp.ki.wVk = keybinds[k];
				inp.ki.dwFlags = current ? 0 : KEYEVENTF_KEYUP;
				inputs.push_back(inp);
			}
		}
		action_prevs.copy(action_ptr, 0, action_prevs.size(), 0, action_ptr.size());
		if (!inputs.empty()) SendInput(static_cast<UINT>(inputs.size()), inputs.data(), sizeof(INPUT));
	}
	void runAddExp()
	{
		addExpThread = std::thread([this]()
		{
			checkCudaError(cudaSetDevice(0));
			bool readyHOS = false, readyPred = false, ema_initialized = false, trigger_backward = false;
			float ema_mean = 0.0f, ema_variance = 1.0f;
			constexpr float EMA_ALPHA = 0.05f, PEAK_Z_THRESHOLD = 1.8f;
			while (true)
			{
				{
					std::unique_lock<std::mutex> lock(mtx_exp);
					ready_to_receive_exp = true;
					cv_exp.wait(lock, [this] { return exp_add || isEliminating.load(std::memory_order_acquire) || work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE; });
					if (isEliminating.load(std::memory_order_acquire) || work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE) break;
				}
				if (exp_is_dirty.exchange(false, std::memory_order_acq_rel))
				{
					if (!readyPred)
					{
						readyPred = true;
						exp_add = false;
						ready_to_receive_exp = false;
						cv_exp.notify_all();
					}
					else
					{
						if (!readyHOS && hall_of_shame) readyHOS = true;
						else if (hall_of_shame) hall_of_shame->addEntry(exp, output_fwd, probs_fwd, output_predictor, stream_exp.get());
						if (predictor) predictor->forward(exp.visaud_fwd, exp.visaud_fwd_scales);
						critic->forward(exp.blocks_data.data(), exp.blocks_data_scales.data());
						exp_add = false;
						ready_to_receive_exp = false;
						cv_exp.notify_all();
						if (hall_of_shame) hall_of_shame->updateEntries(stream_exp.get());
						if (work_mode.load(std::memory_order_acquire) != E_Workmode::TRAIN_BY_USER_CRITIC_ONLY && work_mode.load(std::memory_order_acquire) != E_Workmode::TRAIN_BY_USER_EVALUATE_BY_USER)
						{
							const float critic_score_output = critic->get_score_output();
							trigger_backward = false;
							if (!ema_initialized)
							{
								ema_mean = critic_score_output;
								ema_initialized = true;
							}
							else
							{
								const float dynamic_min_std = std::abs(ema_mean) * EMA_ALPHA + epsilon;
								trigger_backward = std::abs((critic_score_output - ema_mean) / std::sqrtf(fmaxf(ema_variance, dynamic_min_std * dynamic_min_std))) > PEAK_Z_THRESHOLD;
								const float weight = trigger_backward ? EMA_ALPHA * 0.1f : EMA_ALPHA, diff = critic_score_output - ema_mean;
								ema_mean = std::fmaf(weight, diff, ema_mean);
								ema_variance = std::fmaf(weight, std::fmaf(diff, diff, -ema_variance), ema_variance);
								if (trigger_backward) isScored.store(critic_score_output, std::memory_order_release);
							}
						}
					}
				}
				else
				{
					exp_add = false;
					ready_to_receive_exp = false;
					cv_exp.notify_all();
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(fps << 2));
			}
		});
	}
	void collectNetworkData() { for (size_t i = 0; i < meta_adapted_data.size(); ++i) checkCudaError(cudaMemcpyAsync(meta_backup_data.data() + meta_global_offsets[i], meta_adapted_data[i], meta_adapted_offsets[i] * sizeof(__half), cudaMemcpyDefault, stream_backward.get())); }
	void distributeNetworkData() { for (size_t i = 0; i < meta_adapted_data.size(); ++i) checkCudaError(cudaMemcpyAsync(meta_adapted_data[i], meta_backup_data.data() + meta_global_offsets[i], meta_adapted_offsets[i] * sizeof(__half), cudaMemcpyDefault, stream_backward.get())); }
	void performBackwardStep(float score, int epoch)
	{
		const float entropy_sum = probs_fwd.transform_reduce<float>(EntropyOP{}, probs_fwd.size(), stream_backward.get()) / probs_fwd.size();
		score = fmaf(entropy_sum, fmaxf(0.001f, 0.01f * powf(0.999f, static_cast<float>(epoch))), score) * score_gamma;
		grad_output.for_each_n(grad_output.size(), BackwardStepOP{grad_output.data(), probs_fwd.data(), action_ptr.data(), score}, stream_backward.get());
		mamba->backward(grad_output, exp.visaud_fwd, exp.visaud_fwd_scales, grad_input);
		for (int i = blocks.size() - 1; i > -1; i--) blocks[i]->backward(grad_input, blocks_data.data() + i * (blocks_data.size() / blocks.size()), blocks_data_scales.data() + i * (blocks_data_scales.size() / blocks.size()), exp.stochasticDepth[i]);
		grad_output.fill(0, grad_output.size(), h_zero(), stream_backward.get());
		LAUNCH_KERNEL(huberGradOutputKernel<>, (probs_fwd.size() + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_backward.get(), output_predictor.data(), output_fwd.data(), grad_output.data(), output_fwd.size() / probs_fwd.size(), probs_fwd.size(), score, hall_of_shame->getHuberDelta());
		predictor->backward(grad_output, exp.visaud_fwd, exp.visaud_fwd_scales, grad_input);
		checkCudaError(cudaStreamSynchronize(stream_backward.get()));
	}
	void performInference()
	{
		if (work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE_ON_TRAIN && !exp_add && ready_to_receive_exp)
		{
			std::unique_lock<std::mutex> lock(mtx_exp);
			exp_add = true;
			cv_exp.notify_all();
			cv_exp.wait(lock, [this] { return !exp_add || isEliminating.load(std::memory_order_acquire); });
			if (isEliminating.load(std::memory_order_acquire)) return;
		}
		if (audioCapture) audioCapture->getAudioBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
		screenCapture->getScreenBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
		dim3 block(32, 32);
		dim3 grid(img_resolution * img_resolution / block.x, input_channels_first / block.y);
		LAUNCH_KERNEL(NCHWtoNHWCKernel<>, grid, block, 0, stream_forward.get(), raw_visaud_fwd.data(), raw_visaud_fwd.data(), img_resolution, input_channels_first);
		LAUNCH_KERNEL(quantizeKernel<>, ((raw_visaud_fwd.size() >> 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward.get(), raw_visaud_fwd.data(), exp.visaud_fwd.data(), exp.visaud_fwd_scales.data(), raw_visaud_fwd.size());
		for (int i = 0; i < blocks.size(); i++) blocks[i]->forward(exp.visaud_fwd.data(), exp.visaud_fwd_scales.data(), exp.visaud_fwd.data(), exp.visaud_fwd_scales.data(), false, false);
		mamba->forward(exp.visaud_fwd, exp.visaud_fwd_scales);
		action_ptr.fill(0, action_ptr.size(), 0, stream_forward.get());
		LAUNCH_KERNEL(setActionsKernel<>, (probs_fwd.size() + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward.get(), probs_fwd.data(), action_ptr.data(), probs_fwd.size());
		checkCudaError(cudaStreamSynchronize(stream_forward.get()));
		executeActions();
	}
	void performInferenceCritic()
	{
		if (!exp_add && ready_to_receive_exp)
		{
			std::unique_lock<std::mutex> lock(mtx_exp);
			exp_add = true;
			cv_exp.notify_all();
			cv_exp.wait(lock, [this] { return !exp_add || isEliminating.load(std::memory_order_acquire); });
			if (isEliminating.load(std::memory_order_acquire)) return;
		}
		if (audioCapture) audioCapture->getAudioBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
		screenCapture->getScreenBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
		dim3 block(32, 32);
		dim3 grid(img_resolution * img_resolution / block.x, input_channels_first / block.y);
		LAUNCH_KERNEL(NCHWtoNHWCKernel<>, grid, block, 0, stream_forward.get(), raw_visaud_fwd.data(), raw_visaud_fwd.data(), img_resolution, input_channels_first);
		LAUNCH_KERNEL(quantizeKernel<>, ((raw_visaud_fwd.size() >> 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward.get(), raw_visaud_fwd.data(), exp.blocks_data.data(), exp.blocks_data_scales.data(), raw_visaud_fwd.size());
		checkCudaError(cudaStreamSynchronize(stream_forward.get()));
	}
	void performInferenceTrain(bool meta_forward = false)
	{
		if (!meta_forward)
		{
			if (!exp_add && ready_to_receive_exp)
			{
				std::unique_lock<std::mutex> lock(mtx_exp);
				exp_add = true;
				cv_exp.notify_all();
				cv_exp.wait(lock, [this] { return !exp_add || isEliminating.load(std::memory_order_acquire); });
				if (isEliminating.load(std::memory_order_acquire)) return;
			}
			if (audioCapture) audioCapture->getAudioBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
			screenCapture->getScreenBufferCurrent(static_cast<void*>(&raw_visaud_fwd));
			dim3 block(32, 32);
			dim3 grid(img_resolution * img_resolution / block.x, input_channels_first / block.y);
			LAUNCH_KERNEL(NCHWtoNHWCKernel<>, grid, block, 0, stream_forward.get(), raw_visaud_fwd.data(), raw_visaud_fwd.data(), img_resolution, input_channels_first);
			LAUNCH_KERNEL(quantizeKernel<>, ((raw_visaud_fwd.size() >> 5) + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward.get(), raw_visaud_fwd.data(), exp.visaud_fwd.data(), exp.visaud_fwd_scales.data(), raw_visaud_fwd.size());
			exp.blocks_data.copy(exp.visaud_fwd, 0, exp.blocks_data.size(), 0, exp.visaud_fwd.size(), stream_forward.get());
			exp.blocks_data_scales.copy(exp.visaud_fwd_scales, 0, exp.blocks_data_scales.size(), 0, exp.visaud_fwd_scales.size(), stream_forward.get());
		}
		blocks_data.copy(exp.blocks_data, 0, exp.blocks_data.size(), 0, exp.blocks_data.size(), stream_forward.get());
		blocks_data_scales.copy(exp.blocks_data_scales, 0, exp.blocks_data_scales.size(), 0, exp.blocks_data_scales.size(), stream_forward.get());
		for (int i = 0; i < blocks.size(); i++)
		{
			if (!meta_forward) exp.stochasticDepth[i] = get_gen_value(1.0f - i * 0.5f / blocks.size());
			const bool final_block = i == blocks.size() - 1;
			blocks[i]->forward(blocks_data.data() + i * (blocks_data.size() / blocks.size()), blocks_data_scales.data() + i * (blocks_data_scales.size() / blocks.size()), final_block ? exp.visaud_fwd.data() : blocks_data.data() + (i + 1) * (blocks_data.size() / blocks.size()), final_block ? exp.visaud_fwd_scales.data() : blocks_data_scales.data() + (i + 1) * (blocks_data_scales.size() / blocks.size()), true, exp.stochasticDepth[i]);
		}
		mamba->forward(exp.visaud_fwd, exp.visaud_fwd_scales);
		if (work_mode.load(std::memory_order_acquire) != E_Workmode::TRAIN_BY_SELF && !meta_forward)
		{
			checkCudaError(cudaStreamSynchronize(stream_forward.get()));
			return;
		}
		action_ptr.fill(0, action_ptr.size(), 0, stream_forward.get());
		LAUNCH_KERNEL(setActionsKernel<>, (probs_fwd.size() + gpu_block_threads - 1) / gpu_block_threads, gpu_block_threads, 0, stream_forward.get(), probs_fwd.data(), action_ptr.data(), probs_fwd.size());
		checkCudaError(cudaStreamSynchronize(stream_forward.get()));
		if (meta_forward) return;
		executeActions();
	}
	void performForward(bool meta_forward = false)
	{
		if (work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE || work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE_ON_TRAIN) performInference();
		else if (work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_CRITIC_ONLY) performInferenceCritic();
		else performInferenceTrain(meta_forward);
	}
	void onTrainFinish(bool really_fully_finish = false)
	{
		if (really_fully_finish) work_mode.store(E_Workmode::INFERENCE, std::memory_order_release);
		const bool is_in_inference = work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE;
		const bool is_in_inference_on_train = work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE_ON_TRAIN;
		const bool is_in_inference_on_critic_only = work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_CRITIC_ONLY;
		if (!is_in_inference && !is_in_inference_on_train && !is_in_inference_on_critic_only) return;
		if (is_in_inference || is_in_inference_on_train) stream_backward.reset();
		if (is_in_inference)
		{
			critic.reset();
			stream_exp.reset();
			global_device_side_arena.clear();
			global_host_side_arena.clear();
		}
		predictor.reset();
		hall_of_shame.reset();
		convergence_controller.reset();
	}
	void performBackwardMeta(float score, int epoch)
	{
		is_inputing.store(false, std::memory_order_release);
		{
			std::unique_lock<std::mutex> lock(mtx_running);
			is_running = false;
			cv_running.notify_all();
		}
		std::vector<S_Experience*> shame_experiences;
		hall_of_shame->getEntries(shame_experiences);
		const size_t num_sequences = shame_experiences.size() / meta_seq_len;
		if (num_sequences < 1) return;
		collectNetworkData();
		meta_gradient_sum.fill(0, meta_gradient_sum.size(), h_zero(), stream_backward.get());
		for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
		{
			distributeNetworkData(); 
			const size_t start_idx = seq_idx * meta_seq_len;
			float final_shame_score = 0.0f;
			for (int inner_step = 0; inner_step < 5; ++inner_step)
			{
				mamba->reset(stream_backward.get());
				predictor->reset(stream_backward.get());
				for (size_t idx = 0; idx < meta_seq_len; ++idx)
				{
					auto* shame_exp = shame_experiences[start_idx + idx];
					exp(*shame_exp, stream_backward.get());
					performForward(true);
					if (idx == 0) predictor->forward(exp.visaud_fwd, exp.visaud_fwd_scales);
					if (inner_step == 0) final_shame_score += shame_exp->shame_score;
					if (inner_step == 4) shame_exp->shame_score = fminimum;
				}
				performBackwardStep(final_shame_score / meta_seq_len * score * (1.0f / shame_experiences[start_idx + meta_seq_len - 1]->past_pass_count), epoch);
			}
			meta_adapted_data.for_each_n(meta_adapted_data.size(), BackwardMetaOP{meta_gradient_sum.data(), meta_adapted_data.data(), meta_backup_data.data(), meta_global_offsets.data(), meta_adapted_offsets.data(), __float2half2_rn(final_shame_score)}, stream_backward.get());
			if ((seq_idx & 3) == 3)
			{
				checkCudaError(cudaStreamSynchronize(stream_backward.get()));
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
		}
		const __half2 meta_lr2_inv = __float2half2_rn(-1.0f * (0.01f - 1.0f / (1.0f + std::expf(-0.001f * (epoch - 2000))) * (0.01f - 0.003f)));
		meta_adapted_data.for_each_n(meta_adapted_data.size(), BackwardScaleOP{meta_adapted_data.data(), meta_gradient_sum.data(), meta_global_offsets.data(), meta_adapted_offsets.data(), __float2half2_rn(1.0f / num_sequences), meta_lr2_inv}, stream_backward.get());
	}
	void performBackwardStandard(float score, int epoch)
	{
		std::vector<S_Experience*> shame_experiences;
		hall_of_shame->getEntries(shame_experiences);
		const size_t num_sequences = shame_experiences.size() / meta_seq_len;
		if (num_sequences < 1) return;
		for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
		{
			mamba->reset(stream_backward.get());
			predictor->reset(stream_backward.get());
			const size_t start_idx = seq_idx * meta_seq_len;
			float final_shame_score = 0.0f;
			for (size_t idx = 0; idx < meta_seq_len; ++idx)
			{
				auto* shame_exp = shame_experiences[start_idx + idx];
				exp(*shame_exp, stream_backward.get());
				performForward(true);
				if (idx == 0) predictor->forward(exp.visaud_fwd, exp.visaud_fwd_scales);
				final_shame_score += shame_exp->shame_score;
				shame_exp->shame_score = fminimum;
			}
			performBackwardStep(final_shame_score / meta_seq_len * score * (1.0f / shame_experiences[start_idx + meta_seq_len - 1]->past_pass_count), epoch);
			if ((seq_idx & 3) == 3)
			{
				checkCudaError(cudaStreamSynchronize(stream_backward.get()));
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
		}
	}
	void performBackward(float score, int epoch)
	{
		is_inputing.store(false, std::memory_order_release);
		{
			std::unique_lock<std::mutex> lock(mtx_running);
			is_running = false;
			cv_running.notify_all();
		}
		std::cout << "[SYSTEM LOG] Starting new backward epoch..." << std::endl;
		const auto frame_start = std::chrono::high_resolution_clock::now();
		if (work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_CRITIC_ONLY) critic->backward(score);
		else
		{
			if (work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_EVALUATE_BY_USER) critic->backward(score);
			if (next_backward_strategy == E_BackwardStrategy::META)
			{
				std::cout << "[SYSTEM LOG] Executing meta-learning algorythm for adaptation..." << std::endl;
				performBackwardMeta(score, epoch);
				if (convergence_controller) next_backward_strategy = convergence_controller->updateAndGetStrategy(meta_gradient_sum, stream_backward.get(), true);
			}
			else
			{
				performBackwardStandard(score, epoch);
				if (convergence_controller) next_backward_strategy = convergence_controller->updateAndGetStrategy(grad_input, stream_backward.get(), false);
			}
			if (next_backward_strategy == E_BackwardStrategy::CONVERGED)
			{
				std::cout << "[SYSTEM LOG] Global convergence reached. Training finished." << std::endl;
				onTrainFinish(true);
			}
			else
			{
				mamba->reset(stream_backward.get());
				predictor->reset(stream_backward.get());
				critic->reset(stream_backward.get());
			}
		}
		const auto frame_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start);
		std::cout << "[SYSTEM LOG] Backward epoch has finished for " << frame_end.count() << " ms." << std::endl;
		isScored.store(fminimum, std::memory_order_relaxed);
		is_inputing.store(true, std::memory_order_release);
		std::unique_lock<std::mutex> lock(mtx_running);
		is_running = true;
		cv_running.notify_all();
	}
public:
	void saveToFile()
	{
		{
			std::unique_lock<std::mutex> lock(mtx_running);
			ready_to_update = true;
			cv_running.wait(lock, [this] { return !is_running || isEliminating.load(std::memory_order_acquire); });
			ready_to_update = false;
		}
		constexpr int max_retries = 5;
		int attempt = 0;
		bool success = false;
		while (!success && attempt < max_retries)
		{
			try 
			{
				for (int i = 0; i < blocks.size(); i++) blocks[i]->saveToFile((std::filesystem::path(dir_path) / std::to_string(i)).string());
				mamba->saveToFile((std::filesystem::path(dir_path) / "mamba.bin").string());
				if (predictor) predictor->saveToFile((std::filesystem::path(dir_path) / "predictor.bin").string());
				if (critic) critic->saveToFile((std::filesystem::path(dir_path) / "critic.bin").string());
				std::cout << "[SYSTEM LOG] Data saved successfully to network folder." << std::endl;
				success = true;
			}
			catch (const std::exception& e) 
			{
				attempt++;
				std::cerr << "[WARNING] Network file is locked by another PC. Retrying save (" << attempt << "/" << max_retries << ")..." << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
		}
		std::unique_lock<std::mutex> lock(mtx_running);
		is_running = true;
		cv_running.notify_all();
	}
	void loadFromFile()
	{
		{
			std::unique_lock<std::mutex> lock(mtx_running);
			ready_to_update = true;
			cv_running.wait(lock, [this] { return !is_running || isEliminating.load(std::memory_order_acquire); });
			ready_to_update = false;
		}
		constexpr int max_retries = 10;
		int attempt = 0;
		bool success = false;
		while (!success && attempt < max_retries)
		{
			try 
			{
				for (int i = 0; i < blocks.size(); i++) blocks[i]->loadFromFile((std::filesystem::path(dir_path) / std::to_string(i)).string());
				mamba->loadFromFile((std::filesystem::path(dir_path) / "mamba.bin").string());
				if (predictor) predictor->loadFromFile((std::filesystem::path(dir_path) / "predictor.bin").string());
				if (critic) critic->loadFromFile((std::filesystem::path(dir_path) / "critic.bin").string());
				std::cout << "[SYSTEM LOG] Data loaded successfully from network folder." << std::endl;
				success = true;
			}
			catch (const std::exception& e) 
			{
				attempt++;
				std::cerr << "[WARNING] Network file is busy (writing in progress). Retrying load (" << attempt << "/" << max_retries << ")..." << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(300));
			}
		}
		std::unique_lock<std::mutex> lock(mtx_running);
		is_running = true;
		cv_running.notify_all();
	}
	Cortex(bool isLoaded, std::string process_conf_path, std::atomic<E_Workmode>& working_mode, std::atomic<bool>& is_eliminating, std::atomic<float>& is_scored,
		   std::mutex& mtxExp, std::mutex& mtxRunning,
		   std::condition_variable& cvExp, std::condition_variable& cvRunning,
		   bool& expAdd, bool& isRunning, std::atomic<bool>& isInputing, std::vector<uint8_t>& key_binds,
		   size_t computationSpeed = 30,
		   float learningRate = 0.001f,
		   size_t exp_buffer_size_total = meta_seq_len * meta_seq_len,
		   size_t exp_buffer_size_active = meta_seq_len * 2) :
		work_mode(working_mode), isEliminating(is_eliminating), isScored(is_scored),
		mtx_exp(mtxExp), mtx_running(mtxRunning),
		cv_exp(cvExp), cv_running(cvRunning),
		exp_add(expAdd), is_running(isRunning), is_inputing(isInputing), keybinds(key_binds),
		fps(computationSpeed),
		dir_path(process_conf_path),
		learning_rate(__float2half(learningRate))
	{
		cudaStream_t temp_stream_fwd, temp_stream_bwd, temp_stream_exp;
        checkCudaError(cudaStreamCreate(&temp_stream_fwd));
		checkCudaError(cudaStreamCreate(&temp_stream_bwd));
		checkCudaError(cudaStreamCreate(&temp_stream_exp));
        stream_forward.reset(temp_stream_fwd);
		stream_backward.reset(temp_stream_bwd);
		stream_exp.reset(temp_stream_exp);
		int dimension = img_resolution, dimension_old = dimension, total_size_grad_weights = 0, total_size_grad_biases = 0;
		constexpr float inv_depth = 1.0f / depth, step_val = (penalty_first - penalty_last) / (depth - 1), penalty_val = penalty_first - (depth - 1) * step_val, delta_channels = (final_channels - base_channels) * inv_depth;
		blocks.reserve(depth);
		size_t total_device_bytes_main = 0, total_host_bytes_main = 0, total_device_bytes_side = 0, total_host_bytes_side = 0, w_bytes = 0, total_learnable_data = 0, total_learnable_data_count = 0;
		for (int i = 0; i < depth; ++i)
		{
			const int stride = (i & 1) == 0 && i > 0 ? 2 : 1;
			dimension = stride == 2 ? dimension >> 1 : dimension;
			const int input_channels = i == 0 ? input_channels_first : (std::lround(base_channels + i * delta_channels) + 31) & ~31;
			const int output_channels = (std::lround(base_channels + (i + 1) * delta_channels) + 31) & ~31;
			blocks.push_back(std::make_unique<Block>(input_channels, output_channels, stride, dimension_old, dimension,
													 stream_forward.get(), &total_size_grad_weights, &total_size_grad_biases,
													 &pipo_db, &pipo_db_num_groups, std::min(3 + (i & ~1), 11),
													 get_gen_value(1.0f - i * 0.5f * inv_depth), stream_backward.get(),
													 &learning_rate, penalty_first - i * step_val, &total_device_bytes_main,
													 &total_host_bytes_main, &total_learnable_data, &total_learnable_data_count));
			dimension_old = dimension;
		}
		const int output_channels_mamba = keybinds.size() + total_bits_mouse, mamba_output_size = dimension * dimension * output_channels_mamba;
		const bool is_in_inference = work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE || work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE_ON_TRAIN;
		const bool is_in_inference_on_inference = work_mode.load(std::memory_order_acquire) == E_Workmode::INFERENCE;
		const bool is_in_inference_on_critic_only = work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_CRITIC_ONLY;
		mamba = std::make_unique<MambaBlock>(output_channels_mamba, dimension, stream_forward.get(), &total_size_grad_weights, &total_size_grad_biases, stream_backward.get(), &fps, &learning_rate, penalty_val, &total_device_bytes_main, &total_host_bytes_main, &total_learnable_data, &total_learnable_data_count);
		if (!is_in_inference && !is_in_inference_on_critic_only)
		{
			predictor = std::make_unique<MambaBlock>(output_channels_mamba, dimension, stream_exp.get(), &total_size_grad_weights, &total_size_grad_biases, stream_backward.get(), nullptr, &learning_rate, penalty_val, &total_device_bytes_side, &total_host_bytes_side, &total_learnable_data, &total_learnable_data_count);
			hall_of_shame = std::make_unique<HallOfShame>(exp_buffer_size_total, exp_buffer_size_active);
			convergence_controller = std::make_unique<ConvergenceController>();
		}
		if (!is_in_inference_on_inference) critic = std::make_unique<Critic>(input_channels_first, ((input_channels_first << 2) + 31) & ~31, img_resolution, stream_exp.get(), &total_size_grad_weights, &total_size_grad_biases, stream_backward.get(), &learning_rate, penalty_val, &total_device_bytes_side);
		screenCapture = std::make_unique<CaptureScreen>(work_mode, isEliminating, is_running, mtx_running, cv_running, &total_device_bytes_main);
		audioCapture = std::make_unique<CaptureAudio>(isEliminating, is_running, mtx_running, cv_running, &total_device_bytes_main);
		constexpr size_t blocks_data_size = (img_resolution * img_resolution * input_channels_first) >> 5;
		total_device_bytes_main += align16(pipo_db_num_groups * sizeof(uint32_t) << 2) * 2 + align16(pipo_db_num_groups * sizeof(int8_t)) * 2;
		total_device_bytes_side += align16(total_size_grad_weights * sizeof(__half)) +
								   align16(total_size_grad_biases * sizeof(__half)) +
								   align16(blocks_data_size * sizeof(uint32_t) << 2) +
								   align16(blocks_data_size * sizeof(int8_t)) +
								   align16(blocks.size() * pipo_db_num_groups * sizeof(uint32_t) << 2) +
								   align16(blocks.size() * pipo_db_num_groups * sizeof(int8_t)) +
								   align16(mamba_h_states_size * sizeof(float)) * 2 +
								   align16(pipo_db * sizeof(__half)) * 4 +
								   align16(output_channels_mamba * sizeof(__half)) +
								   align16(total_learnable_data * sizeof(__half)) * 2 +
								   align16(mamba_output_size * sizeof(__half));
		total_host_bytes_main += align16((output_channels_mamba + total_bits_per_element - 1) / total_bits_per_element * sizeof(action_type)) * 2;
		total_host_bytes_side += (align16(blocks_data_size * sizeof(uint32_t) << 2) + align16(blocks_data_size * sizeof(int8_t))) * exp_buffer_size_total + align16(total_learnable_data_count * sizeof(size_t)) * 2 + align16(total_learnable_data_count * sizeof(__half*));
		global_device_main_arena.resize(total_device_bytes_main, 0, MemoryType::Device);
		global_device_side_arena.resize(total_device_bytes_side, 0, MemoryType::Device);
		global_host_main_arena.resize(total_host_bytes_main, 0, MemoryType::PinnedHost);
		global_host_side_arena.resize(total_host_bytes_side, 0, MemoryType::PinnedHost);
		uint8_t* dev_main_ptr_buffer_a = global_device_main_arena.data();
		raw_visaud_fwd.resize(reinterpret_cast<__half*>(dev_main_ptr_buffer_a), blocks_data_size << 5, MemoryType::Device);
		uint8_t* dev_main_ptr_buffer_a_scales = dev_main_ptr_buffer_a + align16(pipo_db_num_groups * sizeof(uint32_t) << 2);
		uint8_t* dev_main_ptr_buffer_b = dev_main_ptr_buffer_a_scales + align16(pipo_db_num_groups * sizeof(int8_t));
		exp.visaud_fwd.resize(reinterpret_cast<uint32_t*>(dev_main_ptr_buffer_b), pipo_db_num_groups << 2, MemoryType::Device);
		uint8_t* dev_main_ptr_buffer_b_scales = dev_main_ptr_buffer_b + align16(pipo_db_num_groups * sizeof(uint32_t) << 2);
		exp.visaud_fwd_scales.resize(reinterpret_cast<int8_t*>(dev_main_ptr_buffer_b_scales), pipo_db_num_groups, MemoryType::Device);
		uint8_t* dev_main_ptr = dev_main_ptr_buffer_b_scales + align16(pipo_db_num_groups * sizeof(int8_t));
		uint8_t* dev_side_ptr_weights = global_device_side_arena.data();
		uint8_t* dev_side_ptr_biases = dev_side_ptr_weights + align16(total_size_grad_weights * sizeof(__half));
		uint8_t* dev_side_ptr_grad_buffer_a = dev_side_ptr_biases + align16(total_size_grad_biases * sizeof(__half));
		uint8_t* dev_side_ptr_grad_buffer_b = dev_side_ptr_grad_buffer_a + align16(pipo_db * sizeof(__half));
		uint8_t* dev_side_ptr_grad_residual = dev_side_ptr_grad_buffer_b + align16(pipo_db * sizeof(__half));
		uint8_t* dev_side_ptr = dev_side_ptr_grad_residual + align16(pipo_db * sizeof(__half));
		uint8_t* host_main_ptr = global_host_main_arena.data();
		uint8_t* host_side_ptr = global_host_side_arena.data();
		w_bytes = align16(total_learnable_data_count * sizeof(__half*));
		meta_adapted_data.resize(reinterpret_cast<__half**>(host_side_ptr), total_learnable_data_count, MemoryType::PinnedHost);
		host_side_ptr += w_bytes;
		w_bytes = align16(total_learnable_data_count * sizeof(size_t));
		meta_adapted_offsets.resize(reinterpret_cast<size_t*>(host_side_ptr), total_learnable_data_count, MemoryType::PinnedHost);
		host_side_ptr += w_bytes;
		meta_global_offsets.resize(reinterpret_cast<size_t*>(host_side_ptr), total_learnable_data_count, MemoryType::PinnedHost);
		host_side_ptr += w_bytes;
		__half** host_mad_ptr = meta_adapted_data.data();
		size_t* host_mao_ptr = meta_adapted_offsets.data();
		screenCapture->initData(dev_main_ptr);
		audioCapture->initData(dev_main_ptr);
		for(int i = 0; i < depth; ++i) blocks[i]->initData(isLoaded, (std::filesystem::path(dir_path) / std::to_string(i)).string(), dev_main_ptr, host_main_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, host_mad_ptr, host_mao_ptr, dev_side_ptr_grad_buffer_a, dev_side_ptr_grad_buffer_b, dev_side_ptr_grad_residual, dev_main_ptr_buffer_a, dev_main_ptr_buffer_a_scales, dev_main_ptr_buffer_b, dev_main_ptr_buffer_b_scales);
		output_fwd.resize(reinterpret_cast<__half*>(dev_main_ptr), mamba_output_size, MemoryType::Device);
		probs_fwd.resize(reinterpret_cast<__half*>(dev_main_ptr) + mamba_output_size - output_channels_mamba, output_channels_mamba, MemoryType::Device);
		mamba->initData(isLoaded, (std::filesystem::path(dir_path) / "mamba.bin").string(), dev_main_ptr, host_main_ptr, dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, host_mad_ptr, host_mao_ptr);
		w_bytes = align16((output_channels_mamba + total_bits_per_element - 1) / total_bits_per_element * sizeof(action_type));
		action_ptr.resize(reinterpret_cast<action_type*>(host_main_ptr), (output_channels_mamba + total_bits_per_element - 1) / total_bits_per_element, 0, MemoryType::PinnedHost);
		host_main_ptr += w_bytes;
		action_prevs.resize(reinterpret_cast<action_type*>(host_main_ptr), action_ptr.size(), 0, MemoryType::PinnedHost);
		host_main_ptr += w_bytes;
		if (critic) critic->initData(isLoaded, (std::filesystem::path(dir_path) / "critic.bin").string(), dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases);
		w_bytes = align16(blocks_data_size * sizeof(uint32_t) << 2);
		exp.blocks_data.resize(reinterpret_cast<uint32_t*>(dev_side_ptr), blocks_data_size << 2, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(blocks_data_size * sizeof(int8_t));
		exp.blocks_data_scales.resize(reinterpret_cast<int8_t*>(dev_side_ptr), blocks_data_size, MemoryType::Device);
		dev_side_ptr += w_bytes;
		if (predictor)
		{
			output_predictor.resize(reinterpret_cast<__half*>(dev_side_ptr), mamba_output_size, MemoryType::Device);
			probs_predictor.resize(reinterpret_cast<__half*>(dev_side_ptr) + mamba_output_size - output_channels_mamba, output_channels_mamba, MemoryType::Device);
			predictor->initData(isLoaded, (std::filesystem::path(dir_path) / "predictor.bin").string(), dev_side_ptr, host_side_ptr, dev_side_ptr, dev_side_ptr_weights, dev_side_ptr_biases, is_in_inference, host_mad_ptr, host_mao_ptr);
		}
		if (hall_of_shame) hall_of_shame->initData(host_side_ptr, blocks_data_size);
		w_bytes = align16(blocks.size() * pipo_db_num_groups * sizeof(uint32_t) << 2);
		blocks_data.resize(reinterpret_cast<uint32_t*>(dev_side_ptr), blocks.size() * pipo_db_num_groups << 2, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(blocks.size() * pipo_db_num_groups * sizeof(int8_t));
		blocks_data_scales.resize(reinterpret_cast<int8_t*>(dev_side_ptr), blocks.size() * pipo_db_num_groups, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(output_channels_mamba * sizeof(__half));
		grad_output.resize(reinterpret_cast<__half*>(dev_side_ptr), output_channels_mamba, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(pipo_db * sizeof(__half));
		grad_input.resize(reinterpret_cast<__half*>(dev_side_ptr), pipo_db, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(total_learnable_data * sizeof(__half));
		meta_backup_data.resize(reinterpret_cast<__half*>(dev_side_ptr), total_learnable_data, MemoryType::Device);
		dev_side_ptr += w_bytes;
		w_bytes = align16(total_learnable_data * sizeof(__half));
		meta_gradient_sum.resize(reinterpret_cast<__half*>(dev_side_ptr), total_learnable_data, MemoryType::Device);
		dev_side_ptr += w_bytes;
		size_t current_global_offset = 0;
		for (size_t i = 0; i < meta_adapted_offsets.size(); ++i)
		{
			meta_global_offsets[i] = current_global_offset;
			current_global_offset += meta_adapted_offsets[i];
		}
		onTrainFinish();
		printGpuMem("MAIN_DEV: " + std::to_string(total_device_bytes_main / 1024.0f / 1024.0f) + " MB, SIDE_DEV: " + std::to_string(total_device_bytes_side / 1024.0f / 1024.0f) + " MB, MAIN_HOST: " + std::to_string(total_host_bytes_main / 1024.0f / 1024.0f) + " MB, SIDE_HOST: " + std::to_string(total_host_bytes_side / 1024.0f / 1024.0f) + " MB.");
		printGpuMem("Tensors adjusted. Ready to work.");
	}
	void run()
	{
		const std::chrono::milliseconds frame_rate(1000 / fps);
		int epoch = 0, winCount = 0, looseCount = 0;
		constexpr int maxCount = 5;
		if (!audioCapture->run(fps))
		{
			std::cerr << "[WARNING] Failed to start system audio capture." << std::endl;
			audioCapture.reset();
		}
		else std::cout << "[SYSTEM LOG] System audio capture started successfully." << std::endl;
		screenCapture->run(fps);
		runAddExp();
		while (true)
		{
			{
				std::unique_lock<std::mutex> lock(mtx_running);
				if (ready_to_update)
				{
					is_running = false;
					cv_running.notify_all();
				}
				cv_running.wait(lock, [this] { return is_running || isEliminating.load(std::memory_order_acquire); });
				if (isEliminating.load(std::memory_order_acquire)) break;
			}
			if (!screenCapture->hasNewFrame() && audioCapture && !audioCapture->hasNewAudio())
			{
				std::this_thread::sleep_for(frame_rate);
				continue;
			}
			const auto frame_start = std::chrono::high_resolution_clock::now();
			performForward();
			exp_is_dirty.store(true, std::memory_order_release);
			const float score = isScored.load(std::memory_order_acquire);
			if (score > fminimum)
			{
				if (work_mode.load(std::memory_order_acquire) != E_Workmode::INFERENCE_ON_TRAIN)
				{
					performBackward(score, epoch);
					epoch++;
				}
				else isScored.store(fminimum, std::memory_order_relaxed);
				if (score > 0.0f)
				{
					winCount++;
					looseCount = std::max(0, looseCount - 1);
					if (winCount >= maxCount)
					{
						saveToFile();
						winCount = 0;
						looseCount = 0;
					}
				}
				else
				{
					looseCount++;
					winCount = std::max(0, winCount - 1);
					if (looseCount >= maxCount)
					{
						loadFromFile();
						winCount = 0;
						looseCount = 0;
					}
				}
			}
			const auto frame_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start);
			if (frame_end < frame_rate) std::this_thread::sleep_for(frame_rate - frame_end);
			else std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		isEliminating.store(true, std::memory_order_release);
	}
	~Cortex()
	{
		isEliminating.store(true, std::memory_order_release);
		cv_running.notify_all();
		cv_exp.notify_all();
		if (addExpThread.joinable()) addExpThread.join();
		stream_forward.reset(nullptr);
		stream_backward.reset(nullptr);
		stream_exp.reset(nullptr);
	}
	universal_vector<action_type>& getActionsPtr() { return action_ptr; }
};
