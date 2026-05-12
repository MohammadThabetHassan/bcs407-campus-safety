# BCS407 Campus Safety — Build System
# Run `make <target>` from the repo root.

.PHONY: install setup augment balance analyze stats evaluate generate-report train train-balanced resume backup clean verify

# Default: show help
help:
	@echo "BCS407 Campus Safety — Available targets:"
	@echo ""
	@echo "  install          Install Python dependencies"
	@echo "  setup            Rebuild dataset from source zips (setup_v2.py)"
	@echo "  analyze          Run dataset distribution analysis"
	@echo "  stats            Run detailed bbox statistics (dataset_analysis.py)"
	@echo "  balance          Compute inverse-frequency class weights"
	@echo "  augment          Run augmentation pipeline (no balancing)"
	@echo "  augment-balance  Run augmentation with class balancing"
	@echo "  train            Train original model (v2, unbalanced)"
	@echo "  train-balanced   Train balanced model (v3)"
	@echo "  resume           Resume training from last checkpoint"
	@echo "  evaluate         Evaluate best model on test set"
	@echo "  generate-report  Generate all report figures"
	@echo "  full-pipeline    Run setup → analyze → augment-balance → train-balanced → evaluate → generate-report"
	@echo "  backup           Backup run artifacts"
	@echo "  clean            Remove generated dataset (keep zips)"
	@echo "  verify           Quick sanity check"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install albumentations opencv-python-headless matplotlib numpy pyyaml

setup:
	python code/setup_v2.py

analyze:
	python code/analyze_distribution.py

stats:
	python code/dataset_analysis.py

balance:
	python code/apply_class_weights.py

augment:
	python code/augment_v2.py

augment-balance:
	python code/augment_v2.py --balance-mode equalize --target-count 2500

train:
	bash code/train_v2.sh

train-balanced:
	bash code/train_balanced.sh

resume:
	yolo detect train resume model=runs/detect/campus_safety_v3_balanced/weights/last.pt

evaluate:
	python code/evaluate_model.py

generate-report:
	python code/generate_report_plots.py

full-pipeline: setup analyze augment-balance train-balanced evaluate generate-report
	@echo ""
	@echo "=== FULL PIPELINE COMPLETE ==="
	@echo "Check results/plots/ for all generated figures"
	@echo "Check results/metrics_summary.md for metrics table"

backup:
	python code/backup_run_artifacts.py --dest backups/latest

clean:
	rm -rf dataset/
	@echo "Dataset removed. Raw zips are preserved in repo root."

verify:
	@echo "=== Verification ==="
	@test -f "code/setup_v2.py" && echo "✓ setup_v2.py exists" || echo "✗ setup_v2.py missing"
	@test -f "code/augment_v2.py" && echo "✓ augment_v2.py exists" || echo "✗ augment_v2.py missing"
	@test -f "code/train_balanced.sh" && echo "✓ train_balanced.sh exists" || echo "✗ train_balanced.sh missing"
	@test -f "code/analyze_distribution.py" && echo "✓ analyze_distribution.py exists" || echo "✗ missing"
	@test -f "code/dataset_analysis.py" && echo "✓ dataset_analysis.py exists" || echo "✗ missing"
	@test -f "code/apply_class_weights.py" && echo "✓ apply_class_weights.py exists" || echo "✗ missing"
	@test -f "code/evaluate_model.py" && echo "✓ evaluate_model.py exists" || echo "✗ missing"
	@test -f "code/generate_report_plots.py" && echo "✓ generate_report_plots.py exists" || echo "✗ missing"
	@test -f "code/compute_metrics.py" && echo "✓ compute_metrics.py exists" || echo "✗ missing"
	@test -f "dataset/data.yaml" && echo "✓ data.yaml exists" || echo "✗ data.yaml missing (run: make setup)"
	@ls model/weights/*.pt >/dev/null 2>&1 && echo "✓ Model weights found" || echo "✗ No model weights (run: make train)"