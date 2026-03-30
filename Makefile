PYTHON ?= python
RUN_NAME ?= campus_safety_v2_fixed
RUN_DIR ?= runs/detect/$(RUN_NAME)

.PHONY: install setup augment train resume backup

install:
	$(PYTHON) -m pip install -r requirements.txt

setup:
	$(PYTHON) code/setup_v2.py

augment:
	$(PYTHON) code/augment_v2.py

train:
	bash code/train_v2.sh

resume:
	yolo detect train resume model=$(RUN_DIR)/weights/last.pt

backup:
	@if [ -z "$(DEST)" ]; then echo "Usage: make backup DEST=/path/to/backup"; exit 1; fi
	$(PYTHON) code/backup_run_artifacts.py --run-dir $(RUN_DIR) --dest "$(DEST)"
