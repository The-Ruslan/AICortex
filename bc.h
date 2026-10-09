#pragma once
#include <cstdint>
#include <utility>

enum class E_Workmode : uint8_t
{
	INFERENCE,
	INFERENCE_ON_TRAIN,
	TRAIN_BY_USER_EVALUATE_BY_USER,
	TRAIN_BY_USER_EVALUATE_BY_CRITIC,
	TRAIN_BY_USER_CRITIC_ONLY,
	TRAIN_BY_SELF,
};
using action_type = uint64_t;
inline constexpr int gpu_block_threads = 256,
					 img_resolution = 256,
					 input_channels_first = 32,
					 base_channels = 64,
					 final_channels = 2048,
					 mamba_layers_amount = 16,
					 mamba_d_state = 128,
					 mamba_d_inner = final_channels << 1,
					 mamba_h_states_size = mamba_d_inner * (mamba_d_state << 1),
					 mamba_d_in_sta = mamba_d_inner * mamba_d_state,
					 mamba_chunk_size = 8,
					 mamba_mimo_group_size = 32,
					 mamba_mimo_matrix_elements = mamba_mimo_group_size * mamba_mimo_group_size,
					 meta_seq_len = 32,
					 cs_channels = 4,
					 total_bits_per_element = sizeof(action_type) << 3,
					 total_bits_mouse = 34,
					 total_bits_difference = total_bits_per_element - total_bits_mouse;
inline constexpr float epsilon = 1.0f / 8192.0f,
					   fmaximum = 65504.0f,
					   fminimum = -fmaximum,
					   score_gamma = 0.999f,
					   quant_scale_min = -120.0f,
					   quant_scale_max = 120.0f,
					   quant_int4_max = 7.0f,
					   f_residual = 1.0f,
					   penalty_first = 0.00025f,
					   penalty_last = 0.000025f;
