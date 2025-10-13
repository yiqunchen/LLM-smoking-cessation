# Simple helpers for development. Use `make help` to list targets.

PY=python
MODE?=text-only
MODEL?=gpt-4o
CONFIG?=few-shot-feature-select-balanced
SAMPLE?=100

.PHONY: help install data eval eval-all calibrate clean

help:
	@echo "Targets: install, data, eval, eval-all, calibrate, clean"
	@echo "Vars: MODE=$(MODE) MODEL=$(MODEL) CONFIG=$(CONFIG) SAMPLE=$(SAMPLE)"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) analysis-script/preprocess_data.py

eval:
	$(PY) analysis-script/main_eval.py --mode $(MODE) --model $(MODEL) --prompt-config $(CONFIG) --sample-size $(SAMPLE)

eval-all:
	./run_categorical_eval.sh

calibrate:
	$(PY) analysis-script/calibration_eval.py --mode $(MODE) --model $(MODEL)

clean:
	rm -f evaluation_results_*.json calibrated_results_*.json
	rm -rf plots_*/ calibration_plots_*/

