# MM-BRIGHT: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval

<p align="center">
    <a href="https://github.com/mm-bright/MM-BRIGHT" target="_blank">
        <img src="https://img.shields.io/badge/🌐_Website-MM--BRIGHT-blue?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
    </a>
    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
        <img src="https://img.shields.io/badge/📄_Paper-ArXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="ArXiv">
    </a>
    <a href="https://huggingface.co/datasets/mm-bright/MM-BRIGHT" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Dataset-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Datasets">
    </a>
    <a href="https://github.com/mm-bright/MM-BRIGHT/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/⚖️_License-CC--BY--4.0-green?style=for-the-badge" alt="License">
    </a>
</p>

<p align="center">
    <img src="figures/intro_fig_2.png" width="80%" alt="Overview of MM-BRIGHT Tasks" style="border-radius: 10px;">
</p>

### 🚨 News
- **[2025-01]** 🚀 **MM-BRIGHT Launch**: We release the MM-BRIGHT benchmark, dataset, and evaluation code!
- **[2025-01]** 📄 **Paper**: Our paper describing the construction and analysis of MM-BRIGHT is available on ArXiv.
- **[2025-01]** 🛠️ **Code**: Full evaluation code for all 4 tasks is released.

---

## 📖 Overview

Existing retrieval benchmarks primarily consist of text-based queries where keyword or semantic matching is usually sufficient. Many real-world queries contain **multimodal elements**—particularly images such as diagrams, charts, and screenshots—that require **intensive reasoning** to identify relevant documents.

**MM-BRIGHT** bridges this gap as the **first multimodal benchmark for reasoning-intensive retrieval**.

### Key Features

| Feature | MM-BRIGHT |
|---------|-----------|
| **Total Queries** | 2,803 |
| **Domains** | 29 diverse technical domains |
| **Total Documents** | 2.5M+ |
| **Retrieval Tasks** | 4 (increasing multimodal complexity) |
| **Image Types** | Photos, Diagrams, Charts, Screenshots, Scientific Figures |
| **Source** | Real-world Stack Exchange Q&A |

### Four Retrieval Tasks

MM-BRIGHT evaluates retrieval across four tasks of increasing multimodal complexity:

| Task | Query | Target | Description |
|------|-------|--------|-------------|
| **Task 1** | Text | Text | Text-to-text retrieval (baseline) |
| **Task 2** | Text + Image | Text | Multimodal query → text documents |
| **Task 3** | Text + Image | Image | Multimodal query → relevant images |
| **Task 4** | Text + Image | Text + Image | Multimodal query → multimodal documents |

---

## 🏆 Leaderboard

### Task 1: Text-to-Text Retrieval (nDCG@10)

| Model | BM25 | Contriever | DiVeR | E5 | GritLM | OpenAI | Qwen2 | Rader | ReasonIR | SFR |
|-------|:----:|:----------:|:-----:|:--:|:------:|:------:|:-----:|:-----:|:--------:|:---:|
| **Avg.** | 8.5 | 20.1 | **32.2** | 25.3 | 25.3 | 28.8 | 28.1 | 24.9 | 28.6 | 26.9 |

### Task 2: Multimodal-to-Text Retrieval (nDCG@10)

| Model | BGE-VL | CLIP | GME-2B | GME-7B | Jina-CLIP | Nomic | SigLIP |
|-------|:------:|:----:|:------:|:------:|:---------:|:-----:|:------:|
| **Avg.** | 10.0 | 10.4 | 19.5 | 22.0 | 23.0 | **27.6** | 10.8 |

> **Finding**: Even state-of-the-art models struggle on MM-BRIGHT. BM25 achieves only 8.5 nDCG@10, while the best multimodal model (Nomic-Vision: 27.6) actually **underperforms** the best text-only model (DiVeR: 32.2).

---

## 📊 Dataset Statistics

### Domains by Category

<details>
<summary><b>STEM & Life Sciences (9 domains)</b></summary>

| Domain | Queries | Documents | Avg. Images/Query |
|--------|--------:|----------:|------------------:|
| Academia | 26 | 60,050 | 1.77 |
| Bioacoustics | 41 | 29,812 | 2.17 |
| Bioinformatics | 90 | 45,545 | 1.62 |
| Biology | 99 | 89,435 | 2.96 |
| Chemistry | 65 | 36,043 | 2.54 |
| Earth Science | 85 | 73,451 | 2.15 |
| Math | 45 | 151,867 | 2.64 |
| Medical Sciences | 55 | 240,844 | 1.85 |
| Physics | 100 | 338,291 | 2.45 |

</details>

<details>
<summary><b>Software & Technical Systems (8 domains)</b></summary>

| Domain | Queries | Documents | Avg. Images/Query |
|--------|--------:|----------:|------------------:|
| Apple | 14 | 29,285 | 2.14 |
| Ask Ubuntu | 35 | 90,198 | 2.09 |
| Bitcoin | 64 | 29,595 | 1.48 |
| Crypto | 74 | 24,054 | 1.50 |
| GIS | 44 | 20,705 | 2.98 |
| Quantum Computing | 88 | 127,009 | 1.84 |
| Robotics | 30 | 11,185 | 2.33 |
| Salesforce | 10 | 8,890 | 2.50 |

</details>

<details>
<summary><b>Social Sciences & Humanities (6 domains)</b></summary>

| Domain | Queries | Documents | Avg. Images/Query |
|--------|--------:|----------:|------------------:|
| Christianity | 30 | 37,875 | 1.47 |
| Economics | 31 | 18,431 | 1.84 |
| Islam | 27 | 14,079 | 1.33 |
| Law | 30 | 26,142 | 1.23 |
| Philosophy | 50 | 137,860 | 1.58 |
| Psychology | 87 | 328,520 | 1.67 |

</details>

<details>
<summary><b>Applied Domains (6 domains)</b></summary>

| Domain | Queries | Documents | Avg. Images/Query |
|--------|--------:|----------:|------------------:|
| Aviation | 125 | 203,938 | 2.41 |
| Gaming | 26 | 68,321 | 1.85 |
| PM | 50 | 93,376 | 1.56 |
| Quant | 34 | 64,044 | 1.38 |
| Sustainability | 62 | 32,365 | 1.61 |
| Travel | 68 | 68,063 | 1.84 |

</details>

---

## ⚙️ Setup & Installation

### 1. Clone and Install

```bash
git clone https://github.com/mm-bright/MM-BRIGHT.git
cd MM-BRIGHT
pip install -r requirements.txt
```

### 2. Dataset Access

The dataset is automatically loaded from Hugging Face:

```python
from datasets import load_dataset

# Load documents
docs = load_dataset("mm-bright/MM-BRIGHT", "documents", split="academia")

# Load queries (Task 1/2)
queries = load_dataset("mm-bright/MM-BRIGHT", "examples", split="academia")

# Load multimodal queries (Task 3/4)
mm_queries = load_dataset("mm-bright/MM-BRIGHT", "examples_multimodal", split="academia")
```

---

## 🚀 Running Evaluations

### Task 1: Text-to-Text Retrieval

```bash
python run_task1.py --dataset_dir . --model bm25 --domains academia biology chemistry
```

### Task 2: Multimodal Query → Text Documents

```bash
python run_task2.py --dataset_dir . --model nomic-vision --domains academia biology
```

### Task 3: Multimodal Query → Images

```bash
python run_task3.py --dataset_dir . --model clip --domains academia biology
```

### Task 4: Multimodal Query → Multimodal Documents

```bash
python run_task4.py --dataset_dir . --model clip --domains academia biology
```

### Run All Experiments

Use the experiment runner to evaluate all models across all domains:

```bash
# Dry run - see all commands
python run_experiments.py --dry_run

# Execute all experiments
python run_experiments.py --dataset_dir .

# Run specific tasks only
python run_experiments.py --dataset_dir . --tasks 1 2
```

---

## 📁 Project Structure

```
MM-BRIGHT/
├── run_task1.py          # Task 1: Text → Text
├── run_task2.py          # Task 2: Text+Image → Text
├── run_task3.py          # Task 3: Text+Image → Image
├── run_task4.py          # Task 4: Text+Image → Text+Image
├── run_experiments.py    # Batch experiment runner
├── src/
│   ├── data.py           # HuggingFace data loading
│   ├── caching.py        # Embedding cache management
│   ├── eval_runner.py    # Unified evaluation framework
│   ├── utils.py          # Shared utilities
│   ├── models/           # Custom model definitions
│   │   ├── gritlm7b.py
│   │   └── nvmmembed.py
│   └── retrievers/       # Task-specific retrievers
│       ├── task1_text.py
│       ├── task2_multimodal.py
│       ├── task3_image.py
│       └── task4_pair.py
└── outputs/              # Evaluation results
```

---

## 📊 Benchmark Comparison

| Benchmark | #Queries | #Domains | Modality | Reasoning | Multi-Task |
|-----------|:--------:|:--------:|:--------:|:---------:|:----------:|
| BRIGHT | 1,384 | 12 | Text | ✅ | ✅ |
| RAR-b | 45,745 | 17 | Text | ✅ | ❌ |
| WebQA | 7,540 | Open | IT → IT | ❌ | ❌ |
| UNIIR | 190K | 10 | Mixed | ❌ | ✅ |
| ViDoRe | 3,810 | 10 | T → IT | ❌ | ❌ |
| MMEB | 36K | 36 | Mixed | ❌ | ✅ |
| **MM-BRIGHT (Ours)** | **2,803** | **29** | **Mixed** | **✅** | **✅** |

---

## 📝 Citation

If you use MM-BRIGHT in your work, please cite our paper:

```bibtex
@article{mm-bright2025,
  title={MM-BRIGHT: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval},
  author={...},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📄 License

This project is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Acknowledgments

MM-BRIGHT is built on top of the excellent [BRIGHT](https://github.com/xlang-ai/BRIGHT) benchmark and extends it to the multimodal domain. We thank the Stack Exchange community for providing the raw data that makes this benchmark possible.
