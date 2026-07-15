# Kumaşgöz

Tekstil üretim hattında kumaş anomali tespiti. **Phase 0 Adım 1:** donanımsız PatchCore + MVTec `carpet` (dokuma proxy).

## Gereksinimler

- NVIDIA GPU (test edilen: RTX 4070 Laptop) + CUDA sürücü
- Python 3.10–3.12 (Conda ile kurulur)
- Yerel MVTec arşivi: `~/Downloads/mvtec_anomaly_detection.tar.xz`

## Kurulum

```bash
# Conda ortamı (Python 3.12)
conda create -y -n kumasgoz python=3.12
conda activate kumasgoz

# PyTorch CUDA (ör. cu124) — disk kotası için --no-cache-dir önerilir
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Proje
pip install --no-cache-dir -e .
```

GPU smoke check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Çalıştırma

```bash
# 1) MVTec carpet extract
bash scripts/00_extract_mvtec.sh

# 2) PatchCore train + test
python scripts/01_train_patchcore.py --config configs/patchcore_carpet.yaml

# 3) AUROC / F1 / FP raporu (varsayılan eşik)
python scripts/02_eval_report.py --config configs/patchcore_carpet.yaml

# 4) Adaptive threshold (hedef FP ≤ %5)
python scripts/03_adaptive_threshold.py --config configs/patchcore_carpet.yaml

# 5) PaDiM (ensemble partner) + adaptive threshold
python scripts/04_train_model.py --config configs/padim_carpet.yaml
python scripts/03_adaptive_threshold.py --config configs/padim_carpet.yaml

# 6) Ensemble vote (PatchCore + PaDiM)
python scripts/05_ensemble_vote.py --config configs/ensemble_carpet.yaml

# 7) Fake GPIO actuation simulation
python scripts/06_simulate_actuation.py --config configs/actuation.yaml --demo-watchdog

# --- Phase 1 MVP ---
# 8) Calibration wizard (blur gate + profile register)
python scripts/07_calibration_wizard.py \
  --good-dir data/mvtec/carpet/train/good \
  --register-run results/carpet_patchcore/latest \
  --profile-id carpet_patchcore_v1

# 9) Ingest actuation decisions into SQLite offline buffer
python scripts/08_ingest_sqlite.py

# 10) FastAPI dashboard (http://127.0.0.1:8080)
python scripts/09_run_dashboard.py --port 8080

# 11) Live loop E2E (blur → ensemble → actuation → SQLite)
python scripts/10_live_loop.py --config configs/live_loop.yaml

# Optional: supervisord (watchdog + dashboard)
# supervisord -c deploy/supervisord.conf
```

## Sonuçlar

### Adım 1 — varsayılan Anomalib eşiği

RTX 4070 Laptop, MVTec `carpet`, PatchCore (Wide-ResNet50):

| Metrik | Değer |
|--------|-------|
| Image AUROC | **0.986** |
| Pixel AUROC | **0.991** |
| Image F1 | **0.972** |
| FP oranı (`test/good`) | **~7.1%** |

### Adım 2 — adaptive threshold (`max_f1_under_fpr`, max FPR 5%)

| Metrik | Değer |
|--------|-------|
| Threshold | **0.521** |
| FPR | **3.57%** (1/28) — hedef ≤%5 **karşılandı** |
| F1 | **0.977** |

### Adım 3 — ensemble (PatchCore + PaDiM)

| Mod | FPR | F1 |
|-----|-----|----|
| Solo PatchCore @0.5 | 7.1% | 0.978 |
| **Vote AND @0.5** | **3.57%** | **0.977** |
| Mean + adaptive | **3.57%** | **0.983** |

### Adım 4 — sahte GPIO actuation

| Kontrol | Sonuç |
|---------|--------|
| anomaly_confirmed (GT anomaly) | çalıştı |
| false STOP (GT good) | **0** |
| Watchdog timeout → STOP | çalıştı |

### Phase 1 — MVP yazılım (PC)

| Bileşen | Durum |
|---------|--------|
| SQLite offline buffer | `data/kumasgoz.db` |
| Blur detection (Laplacian) | `src/kumasgoz/blur.py` |
| Memory-bank profil | `carpet_patchcore_v1` @ 0.521 |
| Calibration wizard | `scripts/07_calibration_wizard.py` |
| FastAPI dashboard | `:8080` |
| Watchdog + supervisord | `deploy/supervisord.conf` |
| **Live loop E2E** | blur → AND vote → debounce STOP → SQLite |

Live loop (117 test frame): anomaly=86, blur reject=0, score miss=0, false STOP=0, final **STOP**.

Rapor: `results/live_loop/latest/live_loop_report.md`

## Fazlar (özet)

| Faz | Odak |
|-----|------|
| **0** | Simülasyon: PatchCore, adaptive threshold, ensemble, fake GPIO |
| **1** | MVP yazılım (kalibrasyon, dashboard, SQLite) — hâlâ PC |
| **2+** | Jetson / fabrika |

## Lisans notu

MVTec AD dataset kendi lisansına tabidir; arşivi yeniden dağıtmayın.
