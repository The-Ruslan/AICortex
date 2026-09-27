# 🛠 Contributing Guide

Hi! Thank you for checking out the **AICortex** repository. This project is distributed under the **GNU GPLv3** license and is primarily developed by the author independently, with the support of a small group of volunteer testers.

If you would like to help the project evolve, your contributions are always welcome! We just ask you to follow a few simple rules of courtesy and technical hygiene.

## 🐛 Found a Bug? Let Us Know!
If the program behaves incorrectly, crashes, or throws compilation errors, please report it. You are absolutely not required to provide a ready-made fix.
* Open an **Issue** with a detailed description of the problem.
* Specify your **Windows version** and **GPU model** (remember that CUDA 13.0 physically supports only Turing architecture and newer).
* Describe the steps to reproduce the error and attach the console log if needed.

## ⚡️ Optimization Ideas (Ready Code Only)
If you have a great idea on how to speed up the kernels, optimize computations for Tensor Cores (e.g., for Blackwell/Ada Lovelace), or improve WinAPI/WinRT integration, we are all for it! However, the rule here is simple: **submit only ready-made, fully tested code**.

* Please do not open empty topics like *"What if we rewrite this part to make it faster?"* if you don't have a concrete implementation.
* Create a **Fork**, write the working code, test it locally, and submit a **Pull Request**. If the optimization actually works, it will be gladly merged into the project.

## ⚠️ Technical Requirements for Pull Requests
1. **Build Check:** Before submitting, make sure to run `_build_universal.bat`. The code must compile without warnings in both **Release** and **Debug** modes.
2. **Clean Stack:** We strictly use a codebase built on **C++17, CUDA 13.0, WinAPI, and WinRT**. No external heavy frameworks or third-party build systems like CMake are allowed.
3. **Architecture Compatibility:** Your code must successfully compile for all NVIDIA architectures specified in the script (from `sm_75` to `sm_100`).
