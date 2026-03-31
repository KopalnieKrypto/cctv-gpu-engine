# RTX 5070 vs RTX 4090 — Ograniczenia i Konsekwencje

> Zebrane z testow na serwerze testowym (16-17.03.2026), design docs ml-compute-engine, benchmarkow autoresearch, konfiguracji autoquant.

---

## Specyfikacja porownawcza

| Parametr | RTX 5070 | RTX 4090 |
|----------|----------|----------|
| Architektura | Blackwell (sm_120) | Ada Lovelace (sm_89) |
| VRAM | **12 GB** GDDR7 | **24 GB** GDDR6X |
| Compute capability | 12.0 | 8.9 |
| Cena rynkowa (PLN, marzec 2026) | ~3500-4000 | ~9000-11000 |
| Ilosc na serwerach | 6 per serwer, 60 total | brak (planowane do autoresearch) |

## 1. VRAM — najwazniejsze ograniczenie

**RTX 5070: 12 GB vs RTX 4090: 24 GB — polowa pamieci.**

| Scenariusz | RTX 5070 (12GB) | RTX 4090 (24GB) |
|------------|-----------------|-----------------|
| YOLO v11 (detekcja defektow, CV) | 2-4 GB — OK | 2-4 GB — OK |
| YOLO v11-pose (analiza wideo) | ~2-4 GB — OK | ~2-4 GB — OK |
| XGBoost surrogate (CPU-only) | 0 GB — OK | 0 GB — OK |
| vLLM Llama 8B AWQ INT4 | 5-6 GB — **ciasno** | 5-6 GB — komfortowo |
| vLLM Llama 13B AWQ INT4 | ~8-9 GB — **na granicy** | ~8-9 GB — OK |
| vLLM Llama 70B (bez TP) | **NIE ZMIESCI SIE** | **NIE ZMIESCI SIE** |
| Training 50M params (autoresearch) | 6.2 GB — OK ale na limicie | 6.2 GB — komfortowo |
| Training wiekszych modeli | **ograniczone** | do ~200M params |

**Zrodlo:** `docs/design/04-model-serving.md:328-339`, `autoresearch/docs/17-03-2026-benchmarks.md:5-10`

### VRAM budget na serwerze (MVP)

```
RTX 5070 = 12GB VRAM per GPU
GPU 0: wolne (post-MVP: vLLM Llama 8B AWQ ≈ 5-6GB)
GPU 1: YOLO ≈ 2-4GB (defect detection lub video analysis)
GPU 2-5: idle (mining lub przyszle modele)
Overhead OS: ~0.5GB per GPU
```

**Zrodlo:** `docs/design/04-model-serving.md:328-339`

### Konsekwencje dla projektu

- Max model per GPU: **13B quantized 4-bit** (AWQ/GPTQ). Rekomendacja: uzyj **8B modeli**
- Tensor parallelism w obrebie 1 serwera (6 GPU = 72GB) mozliwy post-MVP via vLLM — pozwoli na wieksza modele
- Surveillance video analysis (YOLO-pose): **bez problemu** — 2-4 GB VRAM, miesci sie na jednym GPU

**Zrodlo:** `CLAUDE.md:119` — "12GB VRAM → max 13B model per GPU (quantized). Use 8B models"

---

## 2. Flash Attention 3 — NIE DZIALA na Blackwell

**To jest krytyczne odkrycie z testow 17.03.2026.**

| | RTX 5070 (Blackwell sm_120) | RTX 4090 (Ada sm_89) |
|---|---|---|
| Flash Attention 2 | nie testowane | dziala |
| Flash Attention 3 | **CRASH** — `no kernel image is available for execution on the device` | nie wspierane (FA3 = Hopper only) |
| PyTorch SDPA | **dziala (fallback)** | dziala |

### Szczegoly

- FA3 Python module (`kernels-community/flash-attn3`) laduje sie poprawnie
- CUDA kernel crashuje w runtime na sm_120
- **Import success ≠ runtime success** — trzeba sprawdzac compute capability PRZED uzyciem
- Fix: `if cap == (9, 0)` — FA3 tylko na Hopper (sm_90), reszta → PyTorch SDPA

**Zrodlo:** `autoresearch/docs/17-03-2026-benchmarks.md:23-27`, `docs/16-03-2026.md:138`

### Konsekwencje dla projektu

- Kazdy serwis AI na RTX 5070 **musi uzywac SDPA**, nie FA3
- gpu-agent / yolo-serve / surveillance-serve: ustawic SDPA jako domyslny attention backend
- vLLM (post-MVP): zweryfikowac czy automatycznie fallbackuje na SDPA na Blackwell
- **Wydajnosc:** SDPA jest wolniejsze niz FA3 (~10-30% w zaleznosci od modelu i sequence length), ale dziala poprawnie

---

## 3. SDPA + GQA (Grouped Query Attention) — bug

Dodatkowy problem z PyTorch SDPA na customowych modelach:

- `F.scaled_dot_product_attention` **nie obsluguje roznych n_head vs n_kv_head**
- Q: (B, 8, T, D), K/V: (B, 4, T, D) → broadcast error
- Fix: `kt = kt.repeat_interleave(n_rep, dim=1)` przed SDPA

**Dotyczy:** customowych modeli z GQA (Llama-style). Nie dotyczy YOLO/ONNX Runtime.

**Zrodlo:** `autoresearch/docs/17-03-2026-benchmarks.md:29-32`

---

## 4. Docker + NVIDIA Container Toolkit — problemy z kompatybilnoscia

| Problem | Szczegoly | Rozwiazanie |
|---------|-----------|-------------|
| Docker 29 + containerd v2 | nvidia-container-runtime nie wspolpracuje z containerd v2. CDI tez nie dziala | Downgrade do Docker 27.5.1 + containerd 1.7.29 |
| Rootless Docker | `docker` CLI laczy sie z rootless zamiast systemowego, ignoruje `/etc/docker/daemon.json` | Wylacz rootless, ustaw `DOCKER_HOST=unix:///var/run/docker.sock` |

**Dotyczy:** Wszystkich serwerow GPU. Provisioning musi ustawic poprawna wersje Docker.

**Zrodlo:** `docs/16-03-2026.md:17-42`, `autoresearch/docs/17-03-2026-benchmarks.md:40-43`

### Wymagana konfiguracja

```bash
# Docker 27 (NIE 29+)
sudo apt install -y --allow-downgrades \
  docker-ce=5:27.5.1-1~ubuntu.24.04~noble \
  docker-ce-cli=5:27.5.1-1~ubuntu.24.04~noble \
  containerd.io=1.7.29-1~ubuntu.24.04~noble

# NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Weryfikacja
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

---

## 5. CUDA / Driver — wersje

| Komponent | Wersja na serwerze testowym | Wymagania |
|-----------|---------------------------|-----------|
| NVIDIA Driver | 595.45.04 | 550+ (design docs) |
| CUDA (driver) | 13.2 | |
| CUDA (PyTorch) | 12.8 (cu128) | PyTorch 2.9.1 |
| CUDA (Docker base image) | 12.6.0 | nvidia/cuda:12.6.0-base-ubuntu24.04 |

**Uwaga:** Driver CUDA 13.2 jest backward compatible z CUDA 12.x workloads.

**Zrodlo:** `docs/16-03-2026.md:10`, `autoresearch/experiments/gpu-ta/pyproject.toml:17-23`

---

## 6. RAM serwera — ograniczenie

Serwer testowy: **8 GB RAM** (systemowy, nie VRAM).

| Workload | Zuzycie RAM | Status |
|----------|-------------|--------|
| Docker daemon + 1 kontener | ~1-2 GB | OK |
| YOLO serve (Node.js) | ~500MB-1GB | OK |
| Surrogate serve (Python) | ~200-500MB | OK |
| vLLM + Qdrant (post-MVP) | **3-5 GB** | **ciasno** |
| Wiele par w pandas (autoquant) | **problematyczne** | 8GB to malo |

**Zrodlo:** `docs/16-03-2026.md:151` — "8 GB RAM na serwerze — mało, mogą być problemy z większymi AI containerami (vLLM + Qdrant)"
**Zrodlo:** `autoquant/docs/plan-autoquant.md:98` — "8GB RAM serwera — ciasno z wieloma parami w pandas"

### Konsekwencje

- MVP (YOLO + surrogate): OK z 8GB RAM
- Post-MVP (vLLM + Qdrant): wymaga upgrade RAM do 16-32 GB
- Surveillance video analysis: ffmpeg + YOLO-pose — powinno zmiescic sie w 8GB (przetwarzanie frame-by-frame, nie wczytywanie calego wideo do RAM)

---

## 7. Training na RTX 5070 — benchmarki

Z testow autoresearch (17.03.2026):

| DEPTH | Params | Time | Steps | VRAM | val_bpb | Wnioski |
|-------|--------|------|-------|------|---------|---------|
| 6 | 26M | 5 min | 987 | 3.7GB | 1.146 | maly model, duzo stepow, sredni wynik |
| 7 | 39M | 5 min | 350 | 4.9GB | 1.170 | kompromis, gorszy od DEPTH=8 |
| 8 | 50M | 5 min | 275 | 6.2GB | 1.184 | undertrained (za malo stepow) |
| **8** | **50M** | **10 min** | **539** | **6.2GB** | **1.104** | **optymalny** |

**Wniosek:** Na RTX 5070 throughput > model size. Lepiej trenowac mniejszy model dluzej niz wiekszy model krotko.

**RTX 4090 (24GB) pozwolilby na:** DEPTH=12-16 (~100-200M params), ~12-15GB VRAM, potencjalnie lepsze val_bpb.

**Zrodlo:** `autoresearch/docs/17-03-2026-benchmarks.md:5-12`

---

## 8. Wplyw na surveillance video analysis (ten prototyp)

| Aspekt | RTX 5070 | Czy problem? |
|--------|----------|-------------|
| YOLO-pose VRAM | ~2-4 GB / 12 GB | **NIE** — duzo zapasu |
| ONNX Runtime CUDA EP | Wymaga weryfikacji na sm_120 | **MOZE BYC** — fallback CPU EP |
| FA3 | Nie dotyczy (YOLO uzywa ONNX, nie PyTorch attention) | **NIE** |
| Inference speed 1080p @ 1fps | ~1 frame/s szacunkowo | **OK** — 1:1 ratio akceptowalny |
| RAM (8GB) | ffmpeg + YOLO frame-by-frame | **OK** — nie ladujemy calego wideo |
| Docker compatibility | Docker 27.5.1 wymagany | **ZNANE** — provisioning skrypt to ogarnia |

### Jedyne ryzyko: ONNX Runtime na Blackwell

ONNX Runtime z CUDA Execution Provider moze miec problemy na sm_120 (Blackwell), analogicznie do FA3. Trzeba zweryfikowac na serwerze testowym jako pierwszy krok. Fallback: CPU Execution Provider (~10x wolniejszy).

---

## Podsumowanie ograniczen RTX 5070 vs RTX 4090

| Ograniczenie | Wplyw | Severity | Mitygacja |
|-------------|-------|----------|-----------|
| 12GB vs 24GB VRAM | Max 13B model (quantized), 8B rekomendowane | **Wysoki** (dla LLM) / **Niski** (dla CV) | Quantization AWQ/GPTQ, tensor parallelism 6 GPU |
| FA3 crash na Blackwell | SDPA fallback (~10-30% wolniejszy attention) | **Sredni** | SDPA dziala poprawnie, koszt wydajnosci akceptowalny |
| SDPA + GQA bug | Wymaga manual repeat_interleave | **Niski** | Jednorazowy fix w kodzie |
| Docker 29 inkompatybilny | Wymaga downgrade do Docker 27 | **Niski** | Skrypt provisioningu, jednorazowy |
| 8GB RAM serwera | Ciasno z wieloma kontenerami AI | **Sredni** | Upgrade RAM post-MVP, frame-by-frame processing |
| ONNX Runtime na sm_120 | Niewiadomo, wymaga testu | **Nieznany** | CPU EP fallback, albo TensorRT |
| Throughput training | Mniejsze modele niz RTX 4090 | **Sredni** | Dluzsza nauka, mniejsze modele, tensor parallelism |
