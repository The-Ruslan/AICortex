#pragma once
#include "uv.h"

struct S_Experience
{
public:
	universal_vector<uint32_t> visaud_fwd, blocks_data;
	universal_vector<int8_t> visaud_fwd_scales, blocks_data_scales;
	float shame_score = fminimum;
	std::vector<uint8_t> stochasticDepth;
	int past_pass_count = 0;
	void copy_from(const S_Experience& other, cudaStream_t stream = 0)
	{
		blocks_data.copy(other.blocks_data, 0, blocks_data.size(), 0, other.blocks_data.size(), stream);
		blocks_data_scales.copy(other.blocks_data_scales, 0, blocks_data_scales.size(), 0, other.blocks_data_scales.size(), stream);
		shame_score = other.shame_score;
		stochasticDepth = other.stochasticDepth;
		past_pass_count = other.past_pass_count;
	}
	S_Experience() { stochasticDepth.resize(depth, 0); }
    ~S_Experience() {}
    S_Experience(const S_Experience& other) { this->copy_from(other); }
	S_Experience& operator=(const S_Experience& other)
    {
		if (this != &other) this->copy_from(other);
		return *this;
    }
	S_Experience& operator()(const S_Experience& other, cudaStream_t stream = 0)
	{
		if (static_cast<const void*>(this) != static_cast<const void*>(&other)) this->copy_from(other, stream);
		return *this;
	}
};
