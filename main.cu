//main.cu
#define NOMINMAX
#include "cortex.h"

bool msgOnPause(Cortex* cortex, HWND& hwnd, std::atomic<float>& isScored, std::mutex& mtx_running, bool& is_running, std::condition_variable& cv_running)
{
	std::cout << "\n================================================================\n";
	std::cout << "                     AI HAS BEEN SUSPENDED!";
	std::cout << "\n================================================================\n";
	std::cout << " Press [0] to pause switching.\n";
	std::cout << " Press [1] to exit with no data.\n";
	std::cout << " Press [2] to exit with data.\n";
	std::cout << " Press [3] to save data.\n";
	std::cout << " Press [4] to load data.\n";
	std::cout << " Press [5] to set a score if mode is train.\n";
	std::cout << "================================================================\n\n";
	std::string option = "", score = "";
	while (true)
	{
		std::cout << "[MENU] Enter option number >> ";
		std::getline(std::cin, option);
		if (option.empty() || option.length() > 1) 
        {
            std::cout << "[SYSTEM LOG] Invalid input. Try again." << std::endl;
            continue;
        }
        if (option[0] == '0') break; 
		switch (option[0])
		{
			case '1':
				std::cout << "[SYSTEM LOG] Data was not save.\n";
				std::cout << "[SYSTEM LOG] Work completion.\n";
				return true;
			case '2':
				cortex->saveToFile();
				std::cout << "[SYSTEM LOG] Data saved.\n";
				std::cout << "[SYSTEM LOG] Work completion.\n";
				return true;
			case '3':
				cortex->saveToFile();
				std::cout << "[SYSTEM LOG] Data saved.\n";
				break;
			case '4':
				cortex->loadFromFile();
				std::cout << "[SYSTEM LOG] Data loaded.\n";
				break;
			case '5':
				std::cout << "[MENU] Enter score >> ";
				std::getline(std::cin, score);
				try { isScored.store(std::stof(score), std::memory_order_relaxed); }
				catch (const std::exception& e) { std::cout << "[SYSTEM LOG] Invalid input. Try again." << std::endl; }
				break;
			default:
				std::cout << "[SYSTEM LOG] Invalid input. Try again." << std::endl;
				break;
		}
	}
	{
		std::unique_lock<std::mutex> lock(mtx_running);
		is_running = true;
		cv_running.notify_all();
	}
	if (hwnd != nullptr) ShowWindow(hwnd, SW_MINIMIZE);
	return false;
}

void inputUser(Cortex* cortex, HWND& hwnd,
			   std::atomic<E_Workmode>& work_mode, std::atomic<bool>& isEliminating, std::atomic<float>& isScored,
			   std::mutex& mtx_exp, std::mutex& mtx_running,
			   std::condition_variable& cv_exp, std::condition_variable& cv_running,
			   bool& exp_add, bool& is_running, std::atomic<bool>& is_inputing, std::vector<uint8_t>& keybinds)
{
	checkCudaError(cudaSetDevice(0));
	static HWND hwnd_add = nullptr;
	if (!hwnd_add)
	{
		WNDCLASSEX wc = { sizeof(WNDCLASSEX), 0, DefWindowProc, 0, 0, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, "RawInputClass", nullptr };
		RegisterClassEx(&wc);
		hwnd_add = CreateWindowEx(0, wc.lpszClassName, nullptr, 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
		RAWINPUTDEVICE rid_mouse;
		rid_mouse.usUsagePage = 0x01;
		rid_mouse.usUsage = 0x02;
		rid_mouse.dwFlags = RIDEV_INPUTSINK;
		rid_mouse.hwndTarget = hwnd_add;
		RegisterRawInputDevices(&rid_mouse, 1, sizeof(RAWINPUTDEVICE));
	}
	int accumulated_dx = 0, accumulated_dy = 0;
	while (!isEliminating.load(std::memory_order_acquire))
	{
		if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
		{
			bool local_is_running = false;
			{
				std::unique_lock<std::mutex> lock(mtx_running);
				is_running = !is_running;
				local_is_running = is_running;
				cv_running.notify_all();
			}
			if (!local_is_running)
			{
				if (hwnd != nullptr) ShowWindow(hwnd, SW_RESTORE);
				if (msgOnPause(cortex, hwnd, isScored, mtx_running, is_running, cv_running))
				{
					isEliminating.store(true, std::memory_order_release);
					{
						std::lock_guard<std::mutex> lock(mtx_running);
						cv_running.notify_all();
					}
					break;
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		else if (GetAsyncKeyState(VK_LEFT) & 0x8000)
		{
			isScored.store(-0.5f, std::memory_order_relaxed);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		else if (GetAsyncKeyState(VK_UP) & 0x8000)
		{
			isScored.store(1.0f, std::memory_order_relaxed);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		else if (GetAsyncKeyState(VK_RIGHT) & 0x8000)
		{
			isScored.store(0.5f, std::memory_order_relaxed);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		else if (GetAsyncKeyState(VK_DOWN) & 0x8000)
		{
			isScored.store(-1.0f, std::memory_order_relaxed);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		else if ((work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_EVALUATE_BY_CRITIC || work_mode.load(std::memory_order_acquire) == E_Workmode::TRAIN_BY_USER_EVALUATE_BY_USER) && is_inputing.load(std::memory_order_acquire))
		{
			{
				std::unique_lock<std::mutex> lock(mtx_exp);
				if (exp_add || isEliminating.load(std::memory_order_acquire))
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					continue;
				}
			}
			auto& action_ptr = cortex->getActionsPtr();
			action_ptr.fill(0, action_ptr.size(), 0);
			if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) action_ptr[0] |= (1ULL << 0);
			if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) action_ptr[0] |= (1ULL << 1);
			MSG msg;
			while (PeekMessage(&msg, hwnd_add, 0, 0, PM_REMOVE))
			{
				if (msg.message == WM_INPUT)
				{
					UINT dwSize = sizeof(RAWINPUT);
					static RAWINPUT raw;
					GetRawInputData((HRAWINPUT)msg.lParam, RID_INPUT, &raw, &dwSize, sizeof(RAWINPUTHEADER));
					if (raw.header.dwType == RIM_TYPEMOUSE)
					{
						accumulated_dx += raw.data.mouse.lLastX;
						accumulated_dy += raw.data.mouse.lLastY;
					}
				}
				DispatchMessage(&msg);
			}
			action_ptr[0] |= (static_cast<action_type>(accumulated_dx) & 0xFFFF) << 2;
			action_ptr[0] |= (static_cast<action_type>(accumulated_dy) & 0xFFFF) << 18;
			accumulated_dx = 0, accumulated_dy = 0;
			for (size_t k = 0; k < keybinds.size(); ++k)
			{
				if (!(GetAsyncKeyState(keybinds[k]) & 0x8000)) continue;
				size_t target_idx = 0, shift = 0;
				if (k < total_bits_difference) shift = k + total_bits_mouse;
				else
				{
					const size_t k_adj = k - total_bits_difference;
					target_idx = k_adj / total_bits_per_element + 1;
					shift = k_adj & (total_bits_per_element - 1);
				}
				action_ptr[target_idx] |= (1ULL << shift);
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

inline std::string getConfigPath()
{
	static const std::string path = []()
	{
		std::vector<wchar_t> buffer(MAX_PATH); 
		while (true)
		{
			const DWORD length = GetModuleFileNameW(NULL, buffer.data(), static_cast<DWORD>(buffer.size()));
			if (length == 0) throw std::runtime_error("[SYSTEM ERROR] Couldn't get path to executable file.");
			if (length < buffer.size())
			{
				buffer.resize(length);
				break;
			}
			buffer.resize(buffer.size() << 1);
		}
		std::filesystem::path exePath(buffer.begin(), buffer.end());
		return (exePath.parent_path() / "config").string();
	}();
	return path;
}

struct KeyMapping
{
	const char* name;
	unsigned short code;
};
inline constexpr KeyMapping VK_MAP[] = {
	{"-", 0xBD}, {".", 0x6E}, {"/", 0x6F}, {"=", 0xBB}, {",", 0xBC}, {"`", 0xC0},
	{"0", 0x30}, {"1", 0x31}, {"2", 0x32}, {"3", 0x33}, {"4", 0x34}, {"5", 0x35}, {"6", 0x36}, {"7", 0x37}, {"8", 0x38}, {"9", 0x39},
	{"a", 0x41}, {"alt", 0x12},
	{"b", 0x42}, {"backspace", 0x08},
	{"c", 0x43}, {"caps", 0x14}, {"ctrl", 0x11},
	{"d", 0x44}, {"delete", 0x2E},
	{"e", 0x45}, {"end", 0x23}, {"enter", 0x0D},
	{"f", 0x46}, {"f1", 0x70}, {"f10", 0x79}, {"f11", 0x7A}, {"f12", 0x7B}, {"f2", 0x71}, {"f3", 0x72}, {"f4", 0x73}, {"f5", 0x74}, {"f6", 0x75}, {"f7", 0x76}, {"f8", 0x77}, {"f9", 0x78},
	{"g", 0x47},
	{"h", 0x48}, {"home", 0x24},
	{"i", 0x49}, {"insert", 0x2D},
	{"j", 0x4A},
	{"k", 0x4B},
	{"l", 0x4C},
	{"m", 0x4D},
	{"n", 0x4E},
	{"o", 0x4F},
	{"p", 0x50}, {"page_down", 0x22}, {"page_up", 0x21},
	{"q", 0x51},
	{"r", 0x52},
	{"s", 0x53}, {"shift", 0x10}, {"space", 0x20},
	{"t", 0x54}, {"tab", 0x09},
	{"u", 0x55},
	{"v", 0x56},
	{"w", 0x57},
	{"x", 0x58},
	{"y", 0x59},
	{"z", 0x5A}
};
inline unsigned short find_by_key(const std::string& str, const std::string& finalPath)
{
	auto it = std::find_if(std::begin(VK_MAP), std::end(VK_MAP), [&str] (const auto& item) { return item.name == str; });
	if (it == std::end(VK_MAP)) throw std::runtime_error("[SYSTEM ERROR] Unknown key: " + str + "; " + finalPath);
	return it->code;
}

int main()
{
	checkCudaError(cudaSetDevice(0));
	checkCudaError(cudaSetDeviceFlags(cudaDeviceScheduleYield));
	checkCudaError(cudaFree(0));
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    std::cout << "\n================================================================\n";
    std::cout << "  LOCAL AI CONTROLLER\n";
    std::cout << "================================================================\n\n";
    std::cout << "[MENU] Enter process name >> ";
	std::string process_name;
    std::getline(std::cin, process_name);
	std::string process_conf_path = (std::filesystem::path(getConfigPath()) / process_name).string();
	std::cout << "[MENU] Options:\n";
    std::cout << "   [1] inference, frozen weights\n";
	std::cout << "   [2] inference, non-frozen weights\n";
	std::cout << "   [3] RL by user actions and user evaluations (for critic evaluation & main AI)\n";
    std::cout << "   [4] RL by user actions and user evaluations (for critic evaluation)\n";
	std::cout << "   [5] RL by user actions and critic evaluations (for main AI)\n";
    std::cout << "   [6] RL by AI actions and critic evaluations (for main AI)\n";
	std::cout << "   [7] credits\n";
    std::cout << "   [8] exit the program\n";
    std::string option = "";
	std::atomic<E_Workmode> work_mode{E_Workmode::TRAIN_BY_USER_EVALUATE_BY_USER};
	while (option.empty() || option[0] < '1' || option[0] > '6')
	{
		std::cout << "[MENU] Enter option number >> ";
		std::getline(std::cin, option);
		switch (option[0])
		{
			case '1':
				work_mode.store(E_Workmode::INFERENCE, std::memory_order_release);
				break;
			case '2':
				work_mode.store(E_Workmode::INFERENCE_ON_TRAIN, std::memory_order_release);
				break;
			case '3':
				work_mode.store(E_Workmode::TRAIN_BY_USER_EVALUATE_BY_USER, std::memory_order_release);
				break;
			case '4':
				work_mode.store(E_Workmode::TRAIN_BY_USER_CRITIC_ONLY, std::memory_order_release);
				break;
			case '5':
				work_mode.store(E_Workmode::TRAIN_BY_USER_EVALUATE_BY_CRITIC, std::memory_order_release);
				break;
			case '6':
				work_mode.store(E_Workmode::TRAIN_BY_SELF, std::memory_order_release);
				break;
			case '7':
				std::cout << "================================================================\n";
				std::cout << " Author: _mrExecutor_\n";
				std::cout << " Models used:\n";
				std::cout << "  - Google Gemini 3 (fast version)\n";
				std::cout << "  - Local model\n";
				std::cout << "----------------------------------------------------------------\n";
				std::cout << " [NOTE]\n";
				std::cout << "  * AI-powered development.\n";
				std::cout << "  * Written by AI, saved from a disaster by a human enthusiast.\n";
				std::cout << "  * Core rewritten globally about 4 times. LoL.\n";
				std::cout << "================================================================\n";
				option = "";
				break;
			case '8':
				return 0;
			default:
				std::cout << "[SYSTEM LOG] Invalid input. Try again." << std::endl;
				option = "";
				break;
		}
	}
	option = "";
	std::cout << "[MENU] Load saved data (y/n) >> ";
	std::getline(std::cin, option);
	HWND hwnd = GetConsoleWindow();
	std::atomic<bool> isEliminating{false}, is_inputing{true};
	std::atomic<float> isScored{fminimum};
	std::mutex mtx_exp, mtx_running;
	std::condition_variable cv_exp, cv_running;
	bool exp_add = false, is_running = true;
	std::vector<uint8_t> keybinds;
	{
		const std::string finalPath = (std::filesystem::path(process_conf_path) / "keybinds.txt").string();
		std::ifstream file(finalPath);
		if (!file.is_open()) throw std::runtime_error("[SYSTEM ERROR] Failed to open file for reading: " + finalPath);
		std::string line;
		while (std::getline(file, line))
		{
			if (line.empty() || line[0] == '#') continue;
			std::istringstream iss(line);
			std::string key_str;
			if (!(iss >> key_str)) throw std::runtime_error("[SYSTEM ERROR] Expected format 'key' in string: " + line + "; " + finalPath);
			keybinds.push_back(find_by_key(key_str, finalPath));
		}
		file.close();
		keybinds.shrink_to_fit();
	}
	Cortex cortex(option == "y", process_conf_path, work_mode, isEliminating, isScored, mtx_exp, mtx_running, cv_exp, cv_running, exp_add, is_running, is_inputing, keybinds);
	if (msgOnPause(&cortex, hwnd, isScored, mtx_running, is_running, cv_running)) return 0;
	std::thread inputThread = std::thread(inputUser, &cortex, std::ref(hwnd),
										  std::ref(work_mode), std::ref(isEliminating), std::ref(isScored),
										  std::ref(mtx_exp), std::ref(mtx_running),
										  std::ref(cv_exp), std::ref(cv_running),
										  std::ref(exp_add), std::ref(is_running), std::ref(is_inputing), std::ref(keybinds));
	cortex.run();
	if (inputThread.joinable()) inputThread.join();
	cudaDeviceReset();
	return 0;
};
