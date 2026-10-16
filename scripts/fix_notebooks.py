"""Fix MLflow tracking URI in all notebooks — switch from mlruns to SQLite."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

notebooks = [
    ROOT / "notebooks" / "02_baselines.ipynb",
    ROOT / "notebooks" / "03_lgbm.ipynb",
    ROOT / "notebooks" / "04_model_comparison.ipynb",
]

DB_PATH = ROOT / "mlflow.db"
NEW_URI_LINE = f'mlflow.set_tracking_uri("sqlite:///{DB_PATH}")\n'

for nb_path in notebooks:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        new_source = []
        for line in cell["source"]:
            if "mlflow.set_tracking_uri" in line and "mlruns" in line:
                new_source.append(NEW_URI_LINE)
                changed = True
                print(f"[{nb_path.name}] replaced:\n  OLD: {line.strip()}\n  NEW: {NEW_URI_LINE.strip()}")
            else:
                new_source.append(line)
        cell["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Saved {nb_path.name} (changed={changed})\n")

print("All notebooks fixed.")
print(f"MLflow DB will be written to: {DB_PATH}")
