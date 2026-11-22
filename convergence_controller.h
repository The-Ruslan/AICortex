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
	struct Tracker
	{
		float ema_norm = -1.0f, ema_delta = 0.0f;
		int steps = 0;
		bool update(float current_norm, float alpha, float relative_tolerance, int min_steps)
		{
			if (ema_norm < 0.0f)
			{
				ema_norm = current_norm;
				ema_delta = current_norm;
				return false;
			}
			const float diff = fabsf(current_norm - ema_norm);
			ema_norm = fmaf(alpha, current_norm - ema_norm, ema_norm);
			ema_delta = fmaf(alpha, diff - ema_delta, ema_delta);
			steps++;
			return steps >= min_steps && ema_delta < relative_tolerance * ema_norm;
		}
		bool isVolatile(float relative_tolerance) const { return ema_norm > 0.0f && ema_delta > relative_tolerance * ema_norm * 1.5f; }
	};
	Tracker std_tracker, meta_tracker;
	int meta_strike_counter = 0;
public:
	ConvergenceController() {}
	template <typename T>
	E_BackwardStrategy updateAndGetStrategy(const universal_vector<T>& input, cudaStream_t stream, bool is_meta_step)
	{
		constexpr float alpha = 0.05f, relative_tolerance = 0.01f;
		const float current_norm = sqrtf(input.template transform_reduce<float>(SquareOp<T>{}, input.size(), stream) / input.size() + epsilon);
		if (!is_meta_step)
		{
			const bool is_plateau = std_tracker.update(current_norm, alpha, relative_tolerance, 50);
			if (std_tracker.isVolatile(relative_tolerance)) meta_strike_counter = 0;
			if (is_plateau)
			{
				std_tracker.steps = 0;
				return E_BackwardStrategy::META;
			}
			return E_BackwardStrategy::STANDARD;
		}
		else
		{
			const bool is_plateau = meta_tracker.update(current_norm, alpha, relative_tolerance, 5);
			if (meta_tracker.isVolatile(relative_tolerance))
			{
				meta_strike_counter = 0;
				meta_tracker.steps = 0;
				return E_BackwardStrategy::STANDARD;
			}
			if (is_plateau)
			{
				meta_strike_counter++;
				meta_tracker.steps = 0;
				if (meta_strike_counter >= 3) return E_BackwardStrategy::CONVERGED;
				return E_BackwardStrategy::META;
			}
			return E_BackwardStrategy::META;
		}
	}
};
