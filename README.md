# AdaptVis_modified — Spatial Reasoning Focus-Area Experiments

Code and datasets for **Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas**  
- Paper: https://arxiv.org/pdf/2503.01773

This repository is a **modified version** of the original AdaptVis codebase, which itself is based on the “What’s up with VLMs?” codebase:

- **Original AdaptVis repo**: https://github.com/shiqichen17/AdaptVis  
- **What’s "up" with vision-language models? (base repo)**: https://github.com/amitakamath/whatsup_vlms  
  - Paper: https://arxiv.org/pdf/2310.19785

---

## 1) Environment Setup

```bash
git clone https://github.com/haewon22/AdaptVis_modified.git
cd AdaptVis_modified

mkdir -p data output outputs

pip install requirements.txt
```

---

## 2) Datasets

- Dataset loading & evaluation code: `dataset_zoo/aro_datasets.py`
- QA prompt files: `prompts/`

The datasets are compatible with the structure used in the original codebase (and the upstream `whatsup_vlms/data` convention).

### Option A — Auto-download (recommended)
When running evaluation, set `--download=True` (e.g., when running `python main_aro.py` or instantiating datasets in code).  
If dataset files are missing, the script downloads JSONs and images automatically.

### Option B — Manual download
You can also download datasets directly and place them under the `data/` directory:

- Google Drive: https://drive.google.com/drive/u/3/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ  
- HuggingFace Datasets: https://huggingface.co/datasets/AdaptVis/all_datasets

> After download, ensure the folder/file names match what the dataset loader expects.

---

## 3) Running Experiments

### Quick run (recommended)
This repo is set up so you can run a complete example quickly:

```bash
bash run.sh
```

All parameter choices are indicated in `run.sh`.

### Common arguments (as used in `run.sh`)
| Argument | Example | Description |
|---|---:|---|
| `dataset` | `Controlled_Images_A` | Dataset to evaluate. Examples: `Controlled_Images_A`, `Controlled_Images_B`, `VG_QA_one_obj`, `VG_QA_two_obj`, `VSR`, etc. |
| `model` | `llava1.5` | Model name (e.g., LLaVA family). |
| `method` | `scaling_vis` | Evaluation method (see below). |
| `weight` | `1.2` | Coefficient for `scaling_vis` / weighted decoding. Often chosen from `[0, 0.5, 0.8, 1.2, 1.5, 2.0]`. |
| `weight1` | `0.5` | Lower weight for adaptive methods. Often chosen from `[0.5, 0.8]`. |
| `weight2` | `1.2` | Higher weight for adaptive methods. Often chosen from `[1.2, 1.5, 2.0]`. |
| `threshold` | `0.3` | Threshold for adaptive switching. |

---

## 4) Experiments Summary (Methods in this modified repo)

This repo evaluates **spatial relations** (e.g., *left/right/on/under* or *yes/no* for VSR) using multiple strategies:

### A) Weight / Focus scaling
- **`scaling_vis`**  
  Applies a coefficient (`weight`) during greedy decoding to bias attention/focus behavior.

### B) Adaptive weight selection (uncertainty → choose `weight1` vs `weight2`)
- **`adapt_vis`**  
  1) Generate once (base weight), compute confidence from the first-step token distribution  
  2) If confidence < threshold → regenerate with `weight1`, else with `weight2`

- **`adapt_vis_jsd`**  
  Uses **Jensen–Shannon divergence (JSD)** between the relation-option distribution and uniform as uncertainty, then switches weights.

- **`adapt_vis_obj`**  
  If uncertain, **adds an object-focused instruction** (e.g., “look carefully at the subject/object”), then regenerates with a weight.

- **`adapt_vis_entropy`**  
  Uses **normalized entropy** over relation options as uncertainty, then switches weights.

- **`adapt_vis_for_oracle_research`**  
  Research/logging mode: evaluates multiple weights and logs distributions to analyze how weight affects predictions.

### C) Reasoning-based decomposition (multi-step prompting)
- **`reasoning_relative_relationship`**  
  Step1: decide whether it’s left–right vs on–under relationship  
  Step2: answer within the reduced candidate set

- **`reasoning_relative_location`**  
  Ask A relative to B and B relative to A; check if answers are opposites for consistency (final uses the first answer).

- **`reasoning_absolute_4directions`**  
  Ask each object’s approximate 9-grid location (top-left/center/...) → infer relation → ask final question with the inferred hint.

- **`chain_of_thought`**  
  Few-shot CoT prompting; produces step-by-step reasoning then final relation answer.

---

## 5) Outputs

- Results and logs are saved under `./output/`
- Typical JSON outputs include per-sample:
  - prompt / generation / golden answer
  - uncertainty metrics (method-dependent)
  - intermediate reasoning traces (for reasoning methods)
- Attention maps may be saved per sample when enabled via `SAVE_ATTN_PATH`.

---

## 6) Citation

If you use this code or data, please consider citing:

```bibtex
@misc{chen2025spatialreasoninghardvlms,
      title={Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas},
      author={Shiqi Chen and Tongyao Zhu and Ruochen Zhou and Jinghan Zhang and Siyang Gao and Juan Carlos Niebles and Mor Geva and Junxian He and Jiajun Wu and Manling Li},
      year={2025},
      eprint={2503.01773},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01773},
}
```
