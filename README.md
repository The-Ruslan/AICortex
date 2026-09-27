## ALPHA VERSION! NEEDS TO TEST

A complex local AI system based on the Mamba SSM architecture and convolutional networks. The project is built on the C++ / CUDA / OpenMP stack and is designed for autonomous process management using real-time screen capture and system audio streams.

## ⚠️ DISCLAIMER

**PLEASE READ THIS BEFORE USING THE SOFTWARE:**

This project is created solely for informational, research, and experimental purposes. The author **assumes no liability or responsibility** for how you or any third parties use this software product. 

* **In-Game Use:** The author does not support, endorse, or encourage the creation of cheats, clickers, or automated systems for online games. Any use of this AI to gain an unfair advantage violates the End User License Agreements (EULA) of most developers and may lead to an account ban. You proceed entirely at your own risk.
* **Security & Privacy:** The program captures your screen and audio stream. By using it, you take full responsibility for the privacy and confidentiality of your data.
* **No Warranties:** This software is provided "as is", without warranty of any kind, express or implied, regarding its functionality or performance.

## 🛠 Tech Stack

* **Core:** C++ (MSVC, x64), CUDA 13.0
* **Data Capture:** WinRT Graphics Capture (Direct3D11) for video, Windows Audio Session API (WASAPI) for audio.

## 🧠 Architecture

The AI utilizes a hybrid approach for analysis and decision-making:
- Convolutional and residual layers for initial feature processing.
- Efficient sequence and temporal context processing via State Space Model (SSM).
- Actor-Critic and Mamba-predictor for implementing Reinforcement Learning and World Model algorithms.

## 🚀 Operating Modes

1. **Inference, frozen weights** — Running on pre-trained, frozen weights.
2. **Inference, non-frozen weights** — Running with the capability for dynamic fine-tuning through AI self-play.
3. **RL by user actions and user evaluations** — Reinforcement learning based on user actions and manual user feedback.
4. **RL by user actions and critic evaluations** — Training based on human actions, but evaluated by an internal critic model.
5. **RL by AI actions and critic evaluations** — Fully autonomous mode (the AI acts independently, and the critic evaluates the actions).

Manual user feedback for AI: arrow UP is +1 score, DOWN is -1, LEFT is -0.5 and RIGHT is +0.5 score.
To pause AI work press ESC.

## 📋 System Requirements

* **OS:** Windows 10 / 11 (64-bit)
* **GPU:** NVIDIA Graphics Card with sm75+ architecture (Compute Capability 7.5+).
