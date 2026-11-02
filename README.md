# Strata (Flint 40B Edition)

Fork of [Strata](https://github.com/strata-ai/strata) with native support for **Flint 40B MoE** (39.8B total / ~2.5B active, 256 routed + 2 shared experts) and **Tiered KV Context Streaming** up to 262,144 tokens (256K).

![Architecture](https://img.shields.io/badge/Architecture-Recursive_MoE_256E-blue)
![GPU](https://img.shields.io/badge/VRAM-16GB_Compatible-success)
![Speed](https://img.shields.io/badge/Decode-80--88_t%2Fs-brightgreen)
![Context](https://img.shields.io/badge/Context-256K_Tiered-orange)

---

## 🌟 Key Features of this Fork

1. **Native Flint 40B Support**:
   - Pack builder (`tools/flint_pack.py`) to extract embeddings, projection weights, tokenizer and MoE index from `flint-UD-Q3_K_XL.gguf`.
   - Fingerprint registration in `data/gguf_fingerprints.json` and model definitions in `strata_control/models.py`.
2. **Strata Modular Engine (`tools/strata_flint_engine.py`)**:
   - Implements the Strata Engine Line Protocol (`READY`, `GEN`, `T`, `PP`, `DONE`, `STOP`).
   - Seamless integration with Strata's OpenAI API server (`serve/server.py`) and Web Chat UI.
3. **Tiered KV Context Streaming (up to 256K)**:
   - Eliminates CPU layer eviction: preserves all 36 MoE layers on GPU (`ncmoe = 11`) regardless of context size.
   - 32K resident KV window in VRAM (~1.0–1.3 GB) combined with Host RAM checkpointing (`--ctx-checkpoints 16 --cache-ram 8192`).
   - Sustained **~80–88 t/s** decode speed across 32K, 200K, and 256K context lengths on consumer 16GB cards (RTX 5060 Ti).

---

## 🚀 Benchmarks (RTX 5060 Ti 16GB + Ryzen 5 9600X)

| Mode | Context Limit | KV Quant | Layers in VRAM | Layers in RAM (`ncmoe`) | Tiered Memory Layout | Decode Speed | Prefill Speed | VRAM Footprint |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **32K Turbo** | 32,768 | `q8_0` | **36** | 11 | Full VRAM | **~88.9 t/s** | ~1,188 t/s | ~14.8 GB |
| **200K Tiered** | 204,800 | `kvarn3/2` | **36** | 11 | VRAM (1.0 GB) + RAM (8 GB) | **~85–88 t/s** | ~900+ t/s | ~14.4 GB |
| **256K Tiered** | 262,144 | `kvarn3/2` | **36** | 11 | VRAM (1.3 GB) + RAM (8 GB) | **~80–85 t/s** | ~850+ t/s | ~14.7 GB |

---

## 📦 Quick Start

### 1. Build the Flint Pack
```powershell
python tools/flint_pack.py --model E:\AI-models\models\flint\flint-UD-Q3_K_XL.gguf --output E:\AI-models\packs\flint-40b
```

### 2. Launch Flint via Strata Launcher
Run the interactive launcher:
```cmd
.\start-flint.bat
```
Select mode:
* `[1]` **256K Context** (Tiered VRAM/RAM, ~80–85 t/s)
* `[2]` **200K Context** (Tiered VRAM/RAM, ~85–88 t/s)
* `[3]` **32K Context** (Turbo VRAM, ~88 t/s)

OpenAI API will be available at `http://127.0.0.1:8081/v1` and Web UI at `http://127.0.0.1:8081/`.

---

## 📄 License
This fork is distributed under the Apache License 2.0, matching the upstream repository.
