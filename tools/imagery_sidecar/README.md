# Imagery classifier sidecar

A small FastAPI service that runs a real EuroSAT land-cover classifier over
bundled Sentinel-2 RGB patches bound to RakshyaNet corridors. Implements
contract **C1** in `docs/CONTRACT-imagery.md` on `http://127.0.0.1:8011`.

It is a separate process with its own virtualenv so that torch never enters
`backend/requirements.txt` (contract C8.1).

## Start it

```powershell
powershell -ExecutionPolicy Bypass -File tools\imagery_sidecar\start.ps1
```

Startup loads the model and pre-warms it with two dummy inferences. Wait for
`sidecar ready` in the log (~20 s cold, mostly CUDA context init) before the
backend calls it — the backend allows 3 s per call with no retry.

## Verified on this machine

| | |
|---|---|
| Model | `nielsr/vit-finetuned-eurosat-kornia` |
| Device | `cuda` — NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| torch | 2.13.0+cu126 |
| Warm inference | ~18-24 ms per tile |

If CUDA is unavailable the service falls back to CPU automatically and reports
`"device": "cpu"` on `/health` and every `/classify` response.

## Demo day: no network needed

The weights are already cached under `D:\rakshyanet-models\hf`. Verified: with
`HF_HUB_OFFLINE=1` the model loads from cache in 1.0 s and classifies correctly.
If the venue wifi is bad, set that variable before starting to skip the Hub
round-trips entirely:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

## Endpoints

`GET /health` → `{"status","model_id","device","warm"}`

`POST /classify` with `{"corridor_id": "...", "incident_type": "flood"}` →
the C1 response body. `404` when no tile is bound to that corridor id.
`water_like` is true only for labels `River` and `SeaLake`.

## Where things live

Everything heavy is on `D:` — C: has under 6 GB free.

- venv: `D:\rakshyanet-models\venv`
- HF weights: `D:\rakshyanet-models\hf` (set via `HF_HOME` in `start.ps1`)
- pip cache / temp: `D:\rakshyanet-models\pipcache`, `D:\rakshyanet-models\tmp`
- EuroSAT source dataset: `D:\rakshyanet-models\data\eurosat`

Bundled tiles and their manifest are in `backend/data/imagery/`.

## Honesty note

The tiles are genuine Sentinel-2 patches from the EuroSAT benchmark
(Helber et al.), captured over Europe. They are bound to Nepali corridor ids
for demonstration and are **not** imagery of those corridors. `acquired_at`
values are scenario timestamps that drive freshness arithmetic. Both caveats
are recorded in `backend/data/imagery/manifest.json`.
