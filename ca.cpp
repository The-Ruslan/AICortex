//ca.cpp
#include "ca.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <comdef.h>

#pragma comment(lib, "Ole32.lib")

extern "C" {
	void* create_cuda_state_audio(size_t* total_device_bytes_main);
	void init_cuda_state_audio(void* state_ptr, uint8_t*& current_dev_main_ptr);
    void free_cuda_state_audio(void* state_ptr);
    void run_cuda_processing_audio(void* state_ptr, BYTE* pData, uint32_t numFramesAvailable, DWORD flags, uint32_t packetLength);
    void copy_cuda_buffer_to_output_audio(void* state_ptr, void* result_vector_ptr);
    void fill_buffer_zero_audio(void* state_ptr);
	void cudaSetDevice_audio();
}

struct CaptureAudio::Impl
{
    std::atomic<bool>& isEliminating;
    std::mutex &mtx_running;
    std::condition_variable &running_cv;
    bool &is_running;
    IAudioClient* pAudioClient = nullptr;
    IAudioCaptureClient* pCaptureClient = nullptr;
    WAVEFORMATEX* pwfx = nullptr;
    std::thread captureThread;
    uint32_t numFramesAvailable = 0, packetLength = 0;
    BYTE* pData = nullptr;
    DWORD flags = 0;
    void* cudaAudioState = nullptr;
	std::atomic<bool> is_dirty{false};
    void initData(uint8_t*& current_dev_main_ptr) { init_cuda_state_audio(cudaAudioState, current_dev_main_ptr); }
	Impl(std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex &running_mtx, std::condition_variable &cv_running, size_t* total_device_bytes_main) : isEliminating(is_eliminating), is_running(isRunning), mtx_running(running_mtx), running_cv(cv_running) { cudaAudioState = create_cuda_state_audio(total_device_bytes_main); }
    ~Impl()
    {
        if (captureThread.joinable()) captureThread.join();
        if (pAudioClient) { pAudioClient->Stop(); pAudioClient->Release(); }
        if (pCaptureClient) pCaptureClient->Release();
        if (pwfx) CoTaskMemFree(pwfx);
        free_cuda_state_audio(cudaAudioState);
        CoUninitialize();
    }
    void captureLoop(int fps)
    {
        if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)))
        {
            fill_buffer_zero_audio(cudaAudioState);
            return;
        }
		cudaSetDevice_audio();
		const std::chrono::milliseconds frame_rate(1000 / std::max(1, fps));
        while (!isEliminating.load(std::memory_order_acquire))
        {
			{
				std::unique_lock<std::mutex> lock(mtx_running);
				running_cv.wait(lock, [this] { return is_running || isEliminating.load(std::memory_order_acquire); });
				if (isEliminating.load(std::memory_order_acquire)) break;
			}
			const auto frame_start = std::chrono::high_resolution_clock::now();
			while (SUCCEEDED(pCaptureClient->GetNextPacketSize(&packetLength)) && packetLength > 0) 
			{
				if (SUCCEEDED(pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &flags, nullptr, nullptr)))
				{
					if (flags & AUDCLNT_BUFFERFLAGS_SILENT) fill_buffer_zero_audio(cudaAudioState);
					else
					{
						run_cuda_processing_audio(cudaAudioState, pData, numFramesAvailable, flags, packetLength);
						is_dirty.store(true, std::memory_order_release);
					}
					pCaptureClient->ReleaseBuffer(numFramesAvailable);
				}
			}
			const auto frame_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start);
			if (frame_end < frame_rate) std::this_thread::sleep_for(frame_rate - frame_end);
			else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CoUninitialize();
    }
};
void CaptureAudio::initData(uint8_t*& current_dev_main_ptr) { pImpl->initData(current_dev_main_ptr); }
CaptureAudio::CaptureAudio(std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex &running_mtx, std::condition_variable &running_cv, size_t* total_device_bytes_main) : pImpl(std::make_unique<Impl>(is_eliminating, isRunning, running_mtx, running_cv, total_device_bytes_main)) {}
CaptureAudio::~CaptureAudio() = default;
bool CaptureAudio::run(int fps)
{
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return false;
    IMMDeviceEnumerator* pEnumerator = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&pEnumerator))) return false;
    IMMDevice* pDevice = nullptr;
    if (FAILED(pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice))) { pEnumerator->Release(); return false; }
    pEnumerator->Release();
    if (FAILED(pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&(pImpl->pAudioClient)))) { pDevice->Release(); return false; }
    pDevice->Release();
    if (FAILED(pImpl->pAudioClient->GetMixFormat(&(pImpl->pwfx)))) return false;
    WAVEFORMATEX desiredFormat = {};
    desiredFormat.wFormatTag = WAVE_FORMAT_PCM;
    desiredFormat.nChannels = 2;
    desiredFormat.nSamplesPerSec = pImpl->pwfx->nSamplesPerSec; 
    desiredFormat.wBitsPerSample = 16; 
    desiredFormat.nBlockAlign = desiredFormat.nChannels * (desiredFormat.wBitsPerSample >> 3);
    desiredFormat.nAvgBytesPerSec = desiredFormat.nSamplesPerSec * desiredFormat.nBlockAlign;
    desiredFormat.cbSize = 0;
    WAVEFORMATEX* pClosestMatch = nullptr;
    WAVEFORMATEX* pFinalFormat = &desiredFormat;
    if (pImpl->pAudioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, &desiredFormat, &pClosestMatch) == S_FALSE && pClosestMatch != nullptr) pFinalFormat = pClosestMatch; 
    if (FAILED(pImpl->pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 0, 0, pFinalFormat, nullptr))) { if (pClosestMatch) CoTaskMemFree(pClosestMatch); return false; }
    if (pClosestMatch) CoTaskMemFree(pClosestMatch);
    if (pImpl->pwfx) { CoTaskMemFree(pImpl->pwfx); pImpl->pwfx = nullptr; }
    if (FAILED(pImpl->pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&(pImpl->pCaptureClient)))) return false;
    if (FAILED(pImpl->pAudioClient->Start())) return false;
    pImpl->captureThread = std::thread(&CaptureAudio::Impl::captureLoop, pImpl.get(), fps);
    return true;
}
void CaptureAudio::getAudioBufferCurrent(void* pDeviceVectorResult) { copy_cuda_buffer_to_output_audio(pImpl->cudaAudioState, pDeviceVectorResult); }
bool CaptureAudio::hasNewAudio() { return pImpl->is_dirty.exchange(false, std::memory_order_acq_rel); }
