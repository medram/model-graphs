.PHONY: install run dev clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(VENV)/bin/streamlit run app.py

dev: install
	$(VENV)/bin/streamlit run app.py --server.runOnSave true

clean:
	rm -rf $(VENV)
