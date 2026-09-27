//cs.h
#pragma once
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdint>

enum class E_Workmode : uint8_t;
struct __half;

class CaptureScreen
{
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
public:
    void initData(uint8_t*& current_dev_main_ptr);
	CaptureScreen(std::atomic<E_Workmode>& trainMode, std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex& running_mtx, std::condition_variable& running_cv, size_t* total_device_bytes_main);
    ~CaptureScreen();
    void run(int fps);
    void getScreenBufferCurrent(void* pDeviceVectorResult);
	bool hasNewFrame();
};
