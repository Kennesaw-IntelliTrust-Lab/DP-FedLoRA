
# 🧠 DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models

**DP-FedLoRA** extends [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) by integrating [(ε, δ)-Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy) into federated instruction-tuning of [Large Language Models (LLMs)](https://huggingface.co/models).  
It combines [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) with formal DP guarantees to enable **privacy-preserving** and **communication-efficient** fine-tuning across distributed edge devices.

📄 **Paper**: [_DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device LLMs_ (arXiv:2509.09097)](https://arxiv.org/abs/2509.09097)

---

## 🌟 Highlights

- 🔐 **Differential Privacy:** Gaussian noise + clipping on LoRA matrices with formal (ε, δ)-DP guarantees.  
- ⚙️ **LoRA-Based Efficient Tuning:** Supports quantized [LLaMA-2 (7B / 13B)](https://ai.meta.com/llama/) via [PEFT](https://github.com/huggingface/peft).  
- 🔄 **Extends OpenFedLLM:** Compatible with [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), [SCAFFOLD](https://arxiv.org/abs/1910.06378), and other FL algorithms.  
- 📊 **DP Analytics:** Tracks noisy updates, total variance, and gradient norms each round.  
- 🧪 **Comprehensive Evaluation:** Benchmarked on 5 reasoning and language tasks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## ⚙️ Installation

```bash
git clone --recursive https://github.com/user-name/DP-FedLoRA.git
cd DP-FedLoRA

conda create -n dp-fedlora python=3.10
conda activate dp-fedlora

pip install -r requirements.txt
source setup.sh
```

---

## 🚀 Training

DP-FedLoRA augments OpenFedLLM’s instruction-tuning pipeline with privacy-preserving LoRA fine-tuning.

### Quick Start

```bash
bash training_scripts/DP-FedLora.sh
```

### Example CLI Run

```bash
CUDA_VISIBLE_DEVICES=0 python main_DPFedLoRA.py --model_name_or_path "meta-llama/Llama-2-7b-hf" --dataset_name "vicgalle/alpaca-gpt4" --dataset_sample 20000 --fed_alg "fedavg" --num_clients 20 --sample_clients 2 --num_rounds 200 --max_steps 10 --batch_size 16 --gradient_accumulation_steps 1 --seq_length 512 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_8bit --enable_dp --dp_epsilon 25.0 --dp_delta 1e-5 --dp_clip_norm 0.1 --output_dir "./output" --template "alpaca"
```

---

## 🔐 Differential Privacy Pipeline

Implemented in [`utils/dp_utils_stats.py`](./utils/dp_utils_stats.py).

- Clipping LoRA matrices by Frobenius norm  
- Injecting Gaussian noise into `lora_A` and `lora_B` parameters  
- Tracking global ε through [PrivacyTracker](https://github.com/facebookresearch/opacus)

---

## 📊 DP Statistics & Visualization

DP-FedLoRA records per-round statistics such as Frobenius norms, analytical variance, and DP perturbation magnitudes.  
Scripts in [`scripts/`](./scripts) visualize layer-wise variance and convergence trends.

---

## 🧪 Benchmarks & Evaluation

Evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with tasks: [MMLU](https://huggingface.co/datasets/cais/mmlu), [BBH](https://github.com/google/BIG-bench), [CRASS](https://huggingface.co/datasets), [DROP](https://rajpurkar.github.io/SQuAD-explorer/), and [HumanEval](https://github.com/openai/human-eval).

---

## 🧩 Built With

- [OpenFedLLM (KDD 2024)](https://github.com/rui-ye/OpenFedLLM)  
- [Transformers](https://github.com/huggingface/transformers)  
- [PEFT](https://github.com/huggingface/peft)  
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [pandas](https://pandas.pydata.org/)  

---

## 🧾 Citation

```bibtex
@article{shrestha2025dpfedlora,
  title   = {DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models},
  author  = {Shiva Shrestha and Honghui Xu and Lei Li and Abhisekh Parakh},
  year    = {2025},
  journal = {arXiv preprint arXiv:2509.09097},
  url     = {https://arxiv.org/abs/2509.09097}
}
```

---

## 🧩 Note to Reviewers and Users

This repository is a **research artifact accompanying the paper**  
📄 [_DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models_ (arXiv:2509.09097)](https://arxiv.org/abs/2509.09097)

Because experiments were conducted on multiple setups (e.g., Jetson, RTX GPU cluster, local workstation), some paths and parameters are system-specific.  
Please **edit the following script sections** before reproducing results or running analysis.

---

### 🔧 Where to Edit

| Folder | Script | What to Edit | Purpose |
|:--|:--|:--|:--|
| `scripts/` | `dp_convergence_summary.py`, `dp_layerwise_variance_plot.py`, `analyze_dp_pramas.py` | Update `folder_path` and `file_pattern` variables | Points to your local `DP_LORA_STATS/` experiment folder containing `dp_stats_round_*.csv` |
| `scripts/` | `analyze_training_loss.py` | Update path in `np.load("../OpenFedLLM/server_trained_files/...")` | Path to your training loss `.npy` file |
| `training_scripts/` | `DP-FedLora.sh` | Update dataset path, output directory, and model checkpoint | Runs your fine-tuning configuration |
| `utils/config.py` | Edit `output_dir`, `local_data_dir`, and related configs | Sets experiment naming, logging, and dataset storage |
| `main_DPFedLoRA.py` | Optionally adjust CLI argument defaults | Simplifies running experiments without long CLI commands |

---

### 🧩 Typical Edits You’ll Need

```python
# Example: in dp_layerwise_variance_plot.py
folder_path = "./DP_LORA_STATS/fedavg_Rank32/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a128_20250602153836/lora_stats/"
```

Change it to your actual local path:

```python
folder_path = "../OpenFedLLM/server_trained_files/DP_LORA_STATS/meta-llama/Llama-2-13b-hf_fedavg_Rank32/..."
```

or in CLI:

```bash
--output_dir "./outputs/fedprox_dp25_r32"
```

---

### ⚙️ Why It’s Designed This Way

All experiment and analysis paths are kept **explicit** rather than hidden in configs.  
This ensures full transparency for reviewers — each figure, table, or result can be traced to its exact training run.  
For reproducibility, please verify dataset paths and output directories before execution.

---

## 🙏 Acknowledgements

We thank the maintainers of [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM), [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their foundational contributions.

