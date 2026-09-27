#pragma once
#include "kernels.h"

enum class E_BackwardStrategy
{
	STANDARD,
	META,
	CONVERGED
};

class ConvergenceController
{
private:
	float ema_norm = -1.0f, ema_delta = 0.0f;
	int counter = 0, meta_strike_counter = 0;
public:
	ConvergenceController() {}
	E_BackwardStrategy updateAndGetStrategy(const universal_vector<__half>& input, cudaStream_t stream, bool is_meta_step)
	{
		const float current_norm = sqrtf(input.transform_reduce<float>(SquareOp{}, input.size(), stream) / input.size() + epsilon);
		if (ema_norm < 0)
		{
			ema_norm = current_norm;
			return E_BackwardStrategy::STANDARD;
		}
		const float diff = fabsf(current_norm - ema_norm);
		constexpr float alpha = 0.01f, relative_tolerance = 0.01f;
		ema_norm = fmaf(alpha, current_norm - ema_norm, ema_norm);
		ema_delta = fmaf(alpha, diff - ema_delta, ema_delta);
		if (!is_meta_step) counter++;
		const bool is_plateau = ema_delta < relative_tolerance * ema_norm;
		if ((!is_meta_step && counter > 500 && is_plateau) || (is_meta_step && is_plateau))
		{
			if (meta_strike_counter >= 3) return E_BackwardStrategy::CONVERGED;
			counter = 0;
			meta_strike_counter++;
			return E_BackwardStrategy::META;
		}
		if (ema_delta > relative_tolerance * ema_norm * 1.5f)
		{
			meta_strike_counter = 0;
			if (is_meta_step) counter = 0;
		}
		return E_BackwardStrategy::STANDARD;
	}
};
