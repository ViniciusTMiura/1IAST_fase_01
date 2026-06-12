.PHONY: install lint format clean

install:
	pip install -r requirements.txt

lint:
	flake8 notebooks/ --max-line-length=120 --extend-ignore=E501,W503

format:
	black notebooks/ --line-length=99
	isort notebooks/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete
