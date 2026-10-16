"""
Rewrites Cell 0 of 02_baselines.ipynb with the correct mlflow URI
hardcoded as an absolute path — no ROOT variable ambiguity.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "mlflow.db"
nb_path = ROOT / "notebooks" / "02_baselines.ipynb"

# This is exactly what Cell 1 should contain
NEW_CELL_SOURCE = [
    "import os\n",
    "os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'   # silence the hint\n",
    "\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.express as px\n",
    "import mlflow\n",
    "\n",
    "from m5forecast.evaluation.backtest import Backtester, RollingOriginSplitter\n",
    "from m5forecast.evaluation.metrics import summarise_metrics\n",
    "from m5forecast.models.baselines import SeasonalNaive, ETSForecaster, ARIMAForecaster\n",
    "\n",
    "ROOT = Path(\"..\").resolve()\n",
    "PARQUET = ROOT / \"data\" / \"processed\" / \"m5_long.parquet\"\n",
    "assert PARQUET.exists(), \"Run `make data` first\"\n",
    "\n",
   f'mlflow.set_tracking_uri(r"sqlite:///{DB}")\n',
    "mlflow.set_experiment(\"m5-baselines\")\n",
    "print(\"MLflow OK — tracking to:\", mlflow.get_tracking_uri())\n",
]

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Find and replace the first code cell that imports mlflow
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "mlflow" in "".join(cell["source"]):
        print(f"Replacing cell {i}")
        print("OLD first line:", cell["source"][0].strip())
        cell["source"] = NEW_CELL_SOURCE
        print("NEW last lines:", NEW_CELL_SOURCE[-2].strip(), "|", NEW_CELL_SOURCE[-1].strip())
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSaved. Now in Jupyter:")
print("  1. Close the 02_baselines.ipynb tab entirely")
print("  2. Reopen it from the file browser")
print("  3. Kernel -> Restart Kernel")
print("  4. Run all cells")
