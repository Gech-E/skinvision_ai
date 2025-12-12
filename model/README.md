# Model Artifacts

Place your exported Google Colab model here so the backend can load it for predictions.

## Expected Layout
- `model/model.h5` — default path loaded by the backend
- Optional: another filename if you set `MODEL_PATH` to point to it

## Quick Steps (from Colab)
1) Export your trained Keras/TensorFlow model (H5 or SavedModel).
2) Download the file/directory to your machine.
3) Put it in this folder as `model.h5` (or set `MODEL_PATH` to your filename).
4) Restart the backend (or let auto-reload pick it up).

## Helper Script
Use the lightweight converter to normalize your model into `model.h5`:
```bash
python -m model.model_loader --input /path/to/colab/export --output model/model.h5
```

## Environment Variables
- `MODEL_PATH` — override the model location (defaults to `model/model.h5`).
- `STATIC_DIR` — where prediction images/heatmaps are stored (backend sets/reads this).


