# MM-BRIGHT: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval

<p align="center">
    <!-- <a href="https://github.com/mm-bright/MM-BRIGHT" target="_blank">
        <img src="https://img.shields.io/badge/🌐_Website-MM--BRIGHT-blue?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
    </a>
    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
        <img src="https://img.shields.io/badge/📄_Paper-ArXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="ArXiv">
    </a> -->
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
- **[2026-01]** 🚀 **MM-BRIGHT Launch**: We release the MM-BRIGHT benchmark, dataset, and evaluation code!
- **[2026-01]** 🛠️ **Code**: Full evaluation code for all 4 tasks is released.

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

---

## 📐 Evaluation protocol

### Task 4 relevance is graded

Each Task 4 candidate is a `(passage, image)` pair, written `passage_id|||image_path`.
Every passage also yields a text-only pair, `passage_id|||__NO_IMAGE__`.

| Relevance | Candidate |
|---|---|
| `rel=2` | gold passage paired with an image annotated **positive** for that query |
| `rel=1` | gold passage paired with `__NO_IMAGE__` (its text-only variant) |
| `rel=0` | everything else, including a gold passage paired with an image that exists but is not annotated positive |

Pairs built from an image annotated **negative** are *excluded from the ranking*
rather than scored `0`. If an image is annotated both positive and negative for
the same query, the positive wins.

Qrels are built **before** retrieval, and any positive pair missing from the
generated pair corpus is inserted into the candidate pool, so it can actually be
scored.

### Passage IDs and the `--protocol` flag

Passage IDs take two shapes:

```
academia/a7beca61_6123.txt      # one passage per source document
biology/fec40b40_1635_4.txt     # source document split into chunks
```

The leading hash is the join key linking a passage to its images. **Biology is the
only domain using the chunked form.**

```bash
python run_task4.py --dataset_dir . --model clip --protocol paper    # default
python run_task4.py --dataset_dir . --model clip --protocol legacy   # audit only
```

- `--protocol paper` (default) implements the protocol above.
- `--protocol legacy` reproduces the scripts that generated Tables 5–6 of the
  paper, **including a passage-key bug** that made the parser unable to read
  chunked IDs. Use it only to audit published numbers, never to evaluate a
  new method.

### Reproducing the published tables

Tables 5 and 6 were produced in December 2025 by standalone scripts, not by this
repository, and against a local copy of the corpus that is byte-identical to the
`documents` config published here. Under `--protocol legacy` this repository
reproduces those numbers exactly; `validate_reproduction.py` checks this by
replaying the stored per-domain scores through freshly built qrels.

**Table 6's Biology column is affected by the legacy parser bug.** Because every
Biology gold ID is chunked, no positive image was ever matched to a gold passage,
`rel=2` was empty, and Task 4 in that domain reduced to text-only retrieval. Under
`--protocol paper` the Biology candidate pool grows from 50 image pairs to 27,998
and the numbers change. The other 28 domains are unaffected: none of their chunked
passages come from a source that has images, so both protocols yield an identical
candidate pool.

### Known data issues

- **199 orphaned positive images (28 Biology queries).** Some positive-image
  annotations belong to a source document that contributes no gold passage to
  that query; a few have all their source's chunks listed in `negative_ids`.
  These arose because Biology gold passages were re-chunked *after* image
  annotation, without re-anchoring the images. The evaluator reports them rather
  than dropping them silently, but they are still not scored.
- **Query images.** All shipped retrievers encode only the *first* query image.
  Multi-image queries are therefore evaluated on their first image alone.
- **Text-only candidates.** `__NO_IMAGE__` candidates are encoded by averaging the
  text embedding with the embedding of a blank white image, not as a true
  text-only embedding. This is a property of the baseline implementations, not of
  the benchmark.

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
@inproceedings{abdallah2026mm,
  title={Mm-bright: A multi-task multimodal benchmark for reasoning-intensive retrieval},
  author={Abdallah, Abdelrahman and Mounis, Mohamed Darwish and Abdalla, Mahmoud and Kasem, Mahmoud SalahEldin and Senussi, Mostafa Farouk and Mahmoud, Mohamed and Ali, Mohammed and Jatowt, Adam and Kang, Hyun Soo},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={8604--8612},
  year={2026}
}
```

---

## 📄 License

This project is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Acknowledgments

MM-BRIGHT is built on top of the excellent [BRIGHT](https://github.com/xlang-ai/BRIGHT) benchmark and extends it to the multimodal domain. We thank the Stack Exchange community for providing the raw data that makes this benchmark possible.


