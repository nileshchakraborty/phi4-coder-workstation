# ⚡ Phi4-Coder-Workstation
### Microsoft Phi-4 (14.7B) Algorithmic Reasoning & High-Precision Coding Engine for Apple Silicon Metal & Workstations

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Base Model: Microsoft Phi-4](https://img.shields.io/badge/Base_Model-Microsoft%20Phi--4%20(14.7B)-green.svg)](https://huggingface.co/microsoft/phi-4)
[![Hardware: Apple Metal](https://img.shields.io/badge/Hardware-Apple%20Silicon%20Metal-orange.svg)](#-hardware-performance--specifications)

**Phi4-Coder-Workstation** is a turnkey, production-grade coding and algorithmic reasoning architecture built on **Microsoft Phi-4 (14.7B)**. Engineered specifically for high-memory bandwidth developer workstations (such as Apple Silicon M4 Max and M2 Max), it delivers frontier mathematical proofs, complex state machine synthesis, and zero-hallucination code generation.

---

## 🧠 Why Microsoft Phi-4 as a Coding Backbone?

1. **Unmatched Mathematical & Algorithmic Density**:
   - **MATH-500**: **84.8%** (outperforms many 70B models).
   - **GPQA Diamond**: **56.1%** graduate-level scientific and algorithmic reasoning.
   - Trained on high-density synthetic textbook reasoning data with multi-turn verification.
2. **Zero Non-Western Georestrictions**:
   - Developed in the United States by Microsoft Research under the permissive **MIT License**.
   - Fully open weights for commercial and enterprise deployment.
3. **14.7B Optimal Workstation Footprint**:
   - Fits comfortably in **~10.4 GB RAM** (`Q5_K_M`), leaving over **23 GB of free RAM** on a 36 GB workstation for IDEs, Docker, and build tools.

---

## ⚡ 1-Click Launch

### Option A: Ollama
```bash
# Create and launch the custom tuned Phi4-Coder model
ollama create phi4-coder -f Modelfile
ollama run phi4-coder
```

### Option B: llama-server with Apple Metal Acceleration
```bash
llama-server \
  -m models/phi-4-q5_k_m.gguf \
  -c 16384 \
  -ngl 99 \
  --flash-attn \
  --port 8082
```

---

## 🛠️ CLI Quick Test
```bash
python3 scripts/run_phi4.py --prompt "Implement a lock-free Single-Producer Single-Consumer (SPSC) ring buffer in Python."
```

---

## 📜 License
MIT License. Created by [Nilesh Chakraborty](https://github.com/nileshchakraborty).
