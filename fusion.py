import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scripts import farseeing as fs
from scripts import utils
from scripts.models import get_model_specs
from scripts.model_runner import run_models

# Directories
RES_DIR = Path("results"); RES_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("figs"); FIG_DIR.mkdir(exist_ok=True)
CACHE = RES_DIR / "model_cache"; CACHE.mkdir(exist_ok=True)

WINDOW_SIZES = [3, 5, 7, 10, 15, 30, 60] # seconds
MODEL_SPECS = get_model_specs() # all models
# SEEDS = np.random.RandomState(14).choice(
# 	np.arange(10), size=3, replace=False).tolist()
SEEDS = [0, 1, 2] # for reproducibility
CV_FOLDS = 5
WINDOW_FREQ = 100 #Hz
print(f"Random seeds: {SEEDS}")

# Load train and test data if available, otherwise compute
try:
	subjects = json.load((open(RES_DIR/"subjects.json", "r")))
	TRAIN_SUBJ = subjects['train']
	TEST_SUBJ = subjects['test']
	print("Loaded train/test subjects from subjects.json")
except FileNotFoundError:
	print("No subjects.json found, computing train/test split...")
	TRAIN_SUBJ, TEST_SUBJ = utils.train_test_subjects_split(
		fs, test_size=0.2, random_state=42)
	json.dump({"train": TRAIN_SUBJ.tolist(), "test": TEST_SUBJ.tolist()},
			  open(RES_DIR/"subjects.json", "w"))
	

win_sizes = [10]
whole_df = fs.load()
datasets = {}
for win in win_sizes:
    datasets[win] = utils.split_df(
		whole_df, fs, test_set=TEST_SUBJ,
        window_size=win, segment_test=False,
        thresh=1.4, multiphase=True
)
	
costcv_metrics = []
costcv = get_model_specs(kind="ensemble")
w = 10 # seconds
print(f"Running CostCV window size {w} seconds")
X_tr, X_te, y_tr, y_te = datasets[w]
res = run_models(
	X_tr, X_te, y_tr, y_te,
	model_specs=costcv,
	verbose=True,
	ensemble_models=False,
	ensemble_by_kind=False,
	window_size=w,
    saved_tuned_dir=CACHE
)
res["window_size"] = w
res["fn_factor"] = 2
costcv_metrics.append(res)
print("")
costcv_df = pd.concat(costcv_metrics, ignore_index=True)
costcv_df.to_csv(RES_DIR / "costcv_metrics.csv", index=False)
print(costcv_df)