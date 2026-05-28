# Mobile Actions Agent

A LoRA fine-tuned variant of FunctionGemma 270M that translates natural language commands into structured OS-level function calls. Runs entirely on-device — no internet required at inference time.

---

## Features

- **Natural language → function call** — "Turn on the flashlight" → `turn_on_flashlight{}`
- **7 mobile OS functions** — flashlight, WiFi settings, calendar, maps, email, contacts
- **Multi-function commands** — "Turn on flashlight and open wifi settings" → two parallel calls
- **Argument extraction** — resolves relative times ("tomorrow at 3pm") to ISO datetime
- **3 inference paths** — Float32 LoRA, GGUF Q8_0 (llama.cpp), LiteRT Q8 (Google LiteRT-LM)
- **On-device CPU inference** — ~550 MB RAM, no GPU needed

---

## Project Structure

```text
mobile_actions/
├── inference.py       # Interactive CLI: loads LoRA model, runs inference loop
├── quick_compare.py   # Side-by-side benchmark: Float32 LoRA vs GGUF Q8_0 vs LiteRT Q8
├── requirements.txt   # Pinned pip dependencies
├── .env.example       # Credential template — copy to .env and fill in values
├── .gitignore
└── README.md
```

> `model/`, `models/`, and `document.pdf` are excluded from the repository via `.gitignore`.

---

## Prerequisites

- Python 3.10+
- HuggingFace account with access to [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) (license acceptance required)
- LoRA adapter files in `model/` (download from Google Drive after training)
- GGUF / LiteRT model files in `models/` (only needed for `quick_compare.py`)

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv myvenv
source myvenv/bin/activate        # Windows: myvenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN to your HuggingFace token
```

> **Never commit `.env`** — it contains your HuggingFace token and is listed in `.gitignore`.

### 3. Add LoRA adapter

Place the fine-tuned LoRA adapter files in `model/`:

```text
model/
├── adapter_model.safetensors
├── adapter_config.json
├── tokenizer.json
├── tokenizer_config.json
├── tokenizer.model
├── chat_template.jinja
├── special_tokens_map.json
├── added_tokens.json
└── training_args.bin
```

---

## Running

### Inference (interactive CLI)

```bash
python inference.py
```

The first run downloads the base model (~500 MB) from HuggingFace and caches it. Subsequent runs load from cache instantly.

### Benchmark (3-path comparison)

```bash
python quick_compare.py
```

Runs each prompt through all three loaded paths side-by-side with latency:

| Path         | Runtime          | Format                         |
|--------------|------------------|--------------------------------|
| Float32 LoRA | PyTorch + PEFT   | function name + extracted args |
| GGUF Q8_0    | llama.cpp        | function name + extracted args |
| LiteRT Q8    | Google LiteRT-LM | natural-language confirmation  |

Paths that fail to load (missing file or package) are skipped gracefully.

---

## Environment Variables

| Variable   | Required | Description                                  |
|------------|----------|----------------------------------------------|
| `HF_TOKEN` | Yes      | HuggingFace token to download the base model |

---

## Supported Commands

| Function               | Example trigger                                   |
|------------------------|---------------------------------------------------|
| `turn_on_flashlight`   | "Turn on the flashlight"                          |
| `turn_off_flashlight`  | "Turn off the flashlight"                         |
| `open_wifi_settings`   | "Open WiFi settings"                              |
| `create_calendar_event`| "Schedule lunch with Sarah tomorrow at noon"      |
| `show_map`             | "Navigate to the nearest hospital"                |
| `send_email`           | "Email alice at example.com about the meeting"    |
| `create_contact`       | "Add John Smith, phone 9876543210 to contacts"    |

---

## Results

| Metric                              | Score   |
|-------------------------------------|---------|
| Function name accuracy (fine-tuned) | 100.0%  |
| Function name accuracy (baseline)   | 94.0%   |
| Improvement                         | +6.0 pts|
| Eval examples                       | 200     |

---

## Tech Stack

| Component   | Details                                                    |
|-------------|------------------------------------------------------------|
| Base model  | google/functiongemma-270m-it (270M parameters)             |
| Fine-tuning | LoRA (r=32, alpha=64) via PEFT + SFTTrainer                |
| Training    | Google Colab T4 GPU — ~2 hours, 3 epochs                   |
| Dataset     | google/mobile-actions — 8693 train + ~1000 eval examples   |
| Inference   | CPU-only, PyTorch float32                                  |
| Adapter     | ~65 MB (unmerged LoRA)                                     |
