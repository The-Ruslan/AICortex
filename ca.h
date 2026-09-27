//ca.h
#pragma once
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

struct __half;

class CaptureAudio
{
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
public:
    void initData(uint8_t*& current_dev_main_ptr);
	CaptureAudio(std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex &running_mtx, std::condition_variable &running_cv, size_t* total_device_bytes_main);
    ~CaptureAudio();
    bool run(int fps);
    void getAudioBufferCurrent(void* pDeviceVectorResult);
	bool hasNewAudio();
};
