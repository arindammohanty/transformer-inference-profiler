.PHONY: all test dashboard report slides clean emergency-stop

all: test report slides

test:
	@echo "Running test suite..."
	python3 -m pytest tests/ -v

dashboard:
	@echo "Starting Plotly Dash interactive profiler..."
	./run_dashboard.sh

emergency-stop:
	@echo "Invoking Emergency Stop and RAM Recovery..."
	./run_dashboard.sh --emergency-stop

report:
	@echo "Compiling publication-grade LaTeX report..."
	./compile_report.sh

slides:
	@echo "Synthesizing animated LibreOffice presentation deck..."
	./build_slides.sh

clean:
	@echo "Cleaning transient build artifacts..."
	rm -f report/*.aux report/*.log report/*.out report/*.bbl report/*.blg
	rm -rf .pytest_cache __pycache__ */__pycache__

