env:
	virtualenv venv

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m experiment.train

predict:
	python -m experiment.predict
