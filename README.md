# Parspec SMLE-1 Assignment — Table Detection in PDF Documents

Detect and return bounding boxes of tables in PDF pages using a fine-tuned Table Transformer model. Input is a PDF URL; output is page-wise bounding boxes.

---

## What This Does

Given any PDF URL, the pipeline:
1. Downloads the PDF and renders each page as an image
2. Runs a fine-tuned table detection model on each page
3. Returns bounding boxes of all detected tables, page by page

```python
result = extract_table_bboxes_from_pdf("https://arxiv.org/pdf/1706.03762")

# Output:
# {
#   "page_1": [],
#   "page_6": [[83, 227, 528, 513]],
#   "page_8": [[83, 105, 528, 340], [83, 412, 528, 621]],
#   ...
#   "metadata": {
#     "total_pages": 15,
#     "pages_with_tables": 4,
#     "total_tables": 7,
#     "latency_seconds": 8.3
#   }
# }
```

---

## Approach

Three models were evaluated before picking one:

| Model | mAP@50 | Avg Latency | Notes |
|---|---|---|---|
| TATR zero-shot | Strong baseline | ~0.3s/page | Already pretrained on PubTables-1M |
| **TATR fine-tuned** | **Best accuracy** | **~0.3s/page** | **← Used** |
| YOLOv8n (10 epochs) | Lower | ~0.07s/page | Faster but less accurate |

**Model chosen:** `microsoft/table-transformer-detection`, fine-tuned on 2,500 samples from PubTables-1M for 3 epochs. It's purpose-built for document table detection and was originally trained on the same dataset, making it the strongest starting point.

**Metric chosen:** mAP@50 as primary — it correctly penalises both missed tables and spurious extra boxes, and it's the PASCAL VOC standard. Average IoU and latency are reported alongside it.

---

## Repository Structure

```
parspec_table_detection.ipynb   — Main Kaggle notebook (all code + Q&A)
README.md                       — This file
```

---

## How to Run on Kaggle

> The notebook is designed to run top-to-bottom on Kaggle with no code changes.

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Click the upload icon and select `parspec_table_detection.ipynb`
3. Under **Settings → Accelerator**, select **GPU T4 x2**
4. Click **Run All**
5. Do **not** clear outputs before sharing — the assignment requires run outputs to be visible

Estimated runtime: ~45–60 minutes (includes dataset download, fine-tuning 3 epochs, YOLOv8 training, and full evaluation).

---

## Notebook Sections

| Section | What it covers |
|---|---|
| 1 | Environment setup — installs all dependencies |
| 2 | Download test data from Google Drive, explore annotations |
| 3 | Metric selection and evaluation utilities |
| 4 | Zero-shot TATR baseline — no fine-tuning |
| 5 | Fine-tune TATR on PubTables-1M subset |
| 6 | YOLOv8n training — latency benchmark |
| 7 | Side-by-side model comparison (accuracy + latency) |
| 8 | **Inference pipeline** — `extract_table_bboxes_from_pdf(pdf_url)` |
| 9 | Final evaluation on Parspec test set |
| 10 | Q&A — solution explanation, model choice, shortcomings, metric justification |

---

## Dependencies

All installed automatically in the notebook's first cell:

```
transformers==4.40.0
timm
datasets
gdown
PyMuPDF
ultralytics
torchvision
pycocotools
requests
tqdm
pillow
```

Python 3.10+, PyTorch 2.x, CUDA GPU recommended.

---

## Data

| Source | Description |
|---|---|
| [bsmock/pubtables-1m](https://huggingface.co/datasets/bsmock/pubtables-1m) | Training data — 460k document pages with table annotations |
| [Parspec Drive folder](https://drive.google.com/drive/folders/12oMQpjCAMtFbwZvVeFsJrIVzOZpzsoK6) | Test data — `Orig_Image/` folder + `test_annotated_data.csv` |

The Drive folder is downloaded automatically inside the notebook using `gdown`.

---

## Key Results

- **mAP@50:** reported in Section 9 after full test set evaluation
- **Avg IoU:** reported alongside mAP
- **Latency:** mean and p95 per page, measured on Kaggle T4 GPU
- Full per-image breakdown saved to `/kaggle/working/per_image_results.csv`
- Model comparison chart saved to `/kaggle/working/model_comparison.png`

---

*Submitted for Parspec AI Team — SMLE-1 Assignment*
