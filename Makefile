env:
	virtualenv venv

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python experiment/train.py

predict:
	python experiment/predict.py
