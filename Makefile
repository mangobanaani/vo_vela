# Makefile for L0 Measurement Project
# Author: Pekka Siltala, Aalto University

# Python interpreter
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

# Directories
ANALYSIS_DIR := analysis
RESULTS_DIR := results

# Main analysis scripts
SCRIPTS := $(ANALYSIS_DIR)/measure_L0_bayesian.py \
           $(ANALYSIS_DIR)/systematic_uncertainties.py \
           $(ANALYSIS_DIR)/L0_grid_search_enhanced.py \
           $(ANALYSIS_DIR)/calibrate_vortex_model.py \
           $(ANALYSIS_DIR)/validate_against_literature.py

.PHONY: help venv install activate analysis all clean clean-results clean-all test

# Default target
help:
	@echo "L0 Measurement Project - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make venv            - Create virtual environment"
	@echo "  make install         - Install dependencies in venv"
	@echo "  make activate        - Show command to activate venv"
	@echo "  make analysis        - Run all analysis scripts"
	@echo "  make bayesian        - Run Bayesian L0 measurement only"
	@echo "  make systematics     - Run systematic uncertainties only"
	@echo "  make calibration     - Run vortex model calibration only"
	@echo "  make validation      - Run literature validation only"
	@echo "  make test            - Run test suite"
	@echo "  make all             - Setup venv, install, and run analysis"
	@echo "  make clean           - Remove Python cache files"
	@echo "  make clean-results   - Remove result files"
	@echo "  make clean-all       - Remove everything (venv, results, cache)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)/"
	@echo "To activate: source $(BIN)/activate"

# Install dependencies
install: venv
	@echo "Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "Dependencies installed successfully"

# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source $(BIN)/activate"

# Run all analysis scripts
analysis: install
	@echo "Running all analysis scripts..."
	@echo ""
	@echo "[1/5] Calibrating vortex model..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/calibrate_vortex_model.py
	@echo ""
	@echo "[2/5] Running Bayesian L0 measurement..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/measure_L0_bayesian.py
	@echo ""
	@echo "[3/5] Computing systematic uncertainties..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/systematic_uncertainties.py
	@echo ""
	@echo "[4/5] Running enhanced L0 grid search..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/L0_grid_search_enhanced.py
	@echo ""
	@echo "[5/5] Validating against literature..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/validate_against_literature.py
	@echo ""
	@echo "All analysis complete. Results in $(RESULTS_DIR)/"

# Individual analysis targets
bayesian: install
	@echo "Running Bayesian L0 measurement..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/measure_L0_bayesian.py

systematics: install
	@echo "Computing systematic uncertainties..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/systematic_uncertainties.py

calibration: install
	@echo "Calibrating vortex model..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/calibrate_vortex_model.py

validation: install
	@echo "Validating against literature..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/validate_against_literature.py

grid-search: install
	@echo "Running enhanced L0 grid search..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python $(ANALYSIS_DIR)/L0_grid_search_enhanced.py

# Run tests
test: install
	@echo "Running test suite..."
	cd $(shell pwd) && PYTHONPATH=. $(BIN)/python -m pytest tests/ -v

# Clean Python cache files
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Python cache cleaned"

# Clean result files
clean-results:
	@echo "Removing result files..."
	rm -f $(RESULTS_DIR)/*.json
	rm -f *.json
	@echo "Results cleaned"

# Clean everything including venv
clean-all: clean clean-results
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Complete cleanup done"

# Setup and run everything
all: install analysis
	@echo ""
	@echo "====================================="
	@echo "Project setup and analysis complete!"
	@echo "====================================="
	@echo "Results: $(RESULTS_DIR)/"
