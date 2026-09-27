//cs.cpp
#include "cs.h"
#include "bc.h"
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.DirectX.Direct3d11.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <windows.graphics.capture.interop.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <stdexcept>

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "d3d11.lib")

extern "C" {
	void* create_cuda_state_visual(size_t* total_device_bytes_main);
	void init_cuda_state_visual(void* state_ptr, uint8_t*& current_dev_main_ptr);
	void init_cuda_shared_tex(void* state_ptr, ID3D11Texture2D* sharedTexture);
	void free_cuda_state_visual(void* state_ptr);
	void run_cuda_processing_visual(void* state_ptr, void* train_mode_ptr);
	void copy_cuda_buffer_to_output_visual(void* state_ptr, void* result_vector_ptr);
	IDXGIAdapter* cudaFindAndSelectDevice_visual(IDXGIFactory1* dxgiFactory);
}

struct CaptureScreen::Impl 
{
	winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool{ nullptr };
	winrt::Windows::Graphics::Capture::GraphicsCaptureSession captureSession{ nullptr };
	ID3D11Device* d3dDevice = nullptr;
	ID3D11DeviceContext* d3dContext = nullptr;
	ID3D11Texture2D* sharedTexture = nullptr;
	std::thread captureThread;
	std::atomic<bool>& isEliminating;
	std::mutex& mtx_running;
	std::condition_variable& cv_running;
	bool& is_running;
	std::atomic<E_Workmode>& train_mode;
	void* cudaVisualState = nullptr;
	std::atomic<bool> is_dirty{false};
	void initData(uint8_t*& current_dev_main_ptr) { init_cuda_state_visual(cudaVisualState, current_dev_main_ptr); }
	Impl(std::atomic<E_Workmode>& trainMode, std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex& running_mtx, std::condition_variable& running_cv, size_t* total_device_bytes_main) : train_mode(trainMode), isEliminating(is_eliminating), is_running(isRunning), mtx_running(running_mtx), cv_running(running_cv) { cudaVisualState = create_cuda_state_visual(total_device_bytes_main); }
	~Impl() 
	{
		if (captureSession) captureSession.Close();
		if (framePool) framePool.Close();
		if (captureThread.joinable()) captureThread.join();
		free_cuda_state_visual(cudaVisualState);
		if (sharedTexture) sharedTexture->Release();
		if (d3dDevice) d3dDevice->Release();
		if (d3dContext) d3dContext->Release();
	}
	void initWGC()
	{
		winrt::init_apartment(winrt::apartment_type::multi_threaded);
		IDXGIFactory1* dxgiFactory = nullptr;
		if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&dxgiFactory))) throw std::runtime_error("Failed to create DXGIFactory1");
		IDXGIAdapter* targetAdapter = cudaFindAndSelectDevice_visual(dxgiFactory);
		dxgiFactory->Release();
		if (!targetAdapter) throw std::runtime_error("No CUDA-compatible GPU found!");
		D3D11CreateDevice(targetAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dContext);
		targetAdapter->Release();
		ID3D11Multithread* pMultithread = nullptr;
		if (SUCCEEDED(d3dContext->QueryInterface(__uuidof(ID3D11Multithread), (void**)&pMultithread))) 
		{
			pMultithread->SetMultithreadProtected(TRUE);
			pMultithread->Release();
		}
		auto interop_factory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
		winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
		interop_factory->CreateForMonitor(MonitorFromWindow(GetDesktopWindow(), MONITOR_DEFAULTTOPRIMARY), winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item));
		D3D11_TEXTURE2D_DESC desc = {};
		desc.Width = img_resolution;
		desc.Height = img_resolution;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
		d3dDevice->CreateTexture2D(&desc, nullptr, &sharedTexture);
		init_cuda_shared_tex(cudaVisualState, sharedTexture);
		auto dxgiDevice = winrt::com_ptr<IDXGIDevice>();
		d3dDevice->QueryInterface(winrt::guid_of<IDXGIDevice>(), dxgiDevice.put_void());
		winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice device_interop;
		CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(), reinterpret_cast<::IInspectable**>(winrt::put_abi(device_interop)));
		winrt::Windows::Graphics::SizeInt32 customSize{ img_resolution, img_resolution };
		framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(device_interop, winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized, 1, customSize);
		captureSession = framePool.CreateCaptureSession(item);
		captureSession.StartCapture();
	}
	void captureLoop(int fps)
	{
		initWGC();
		const std::chrono::milliseconds frame_rate(1000 / std::max(1, fps));
		while (!isEliminating.load(std::memory_order_acquire))
		{
			{
				std::unique_lock<std::mutex> lock(mtx_running);
				cv_running.wait(lock, [this] { return is_running || isEliminating.load(std::memory_order_acquire); });
				if (isEliminating.load(std::memory_order_acquire)) break;
			}
			const auto frame_start = std::chrono::high_resolution_clock::now();
			auto frame = framePool.TryGetNextFrame();
			if (frame)
			{
				auto access = frame.Surface().as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
				winrt::com_ptr<ID3D11Texture2D> frameTexture;
				access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), frameTexture.put_void());
				d3dContext->CopyResource(sharedTexture, frameTexture.get());
				d3dContext->Flush();
				run_cuda_processing_visual(cudaVisualState, &train_mode);
				frame.Close();
				is_dirty.store(true, std::memory_order_release);
			}
			const auto frame_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start);
			if (frame_end < frame_rate) std::this_thread::sleep_for(frame_rate - frame_end);
			else std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
};
void CaptureScreen::initData(uint8_t*& current_dev_main_ptr) { pImpl->initData(current_dev_main_ptr); }
CaptureScreen::CaptureScreen(std::atomic<E_Workmode>& trainMode, std::atomic<bool>& is_eliminating, bool& isRunning, std::mutex& running_mtx, std::condition_variable& running_cv, size_t* total_device_bytes_main) : pImpl(std::make_unique<Impl>(trainMode, is_eliminating, isRunning, running_mtx, running_cv, total_device_bytes_main)) {}
CaptureScreen::~CaptureScreen() = default;
void CaptureScreen::run(int fps) { pImpl->captureThread = std::thread(&CaptureScreen::Impl::captureLoop, pImpl.get(), fps); }
void CaptureScreen::getScreenBufferCurrent(void* pDeviceVectorResult) { copy_cuda_buffer_to_output_visual(pImpl->cudaVisualState, pDeviceVectorResult); }
bool CaptureScreen::hasNewFrame() { return pImpl->is_dirty.exchange(false, std::memory_order_acq_rel); }
