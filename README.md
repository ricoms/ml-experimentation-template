# Machine Learning simple experimentation template

done by: [@ricoms](https://github.com/ricoms)

This repo represents a very simple batch job prediction with machine learning. To increment its maturity some test, machine learning metrics validation, and containerization (docker) could be added.

This repo contains a main package `experiment`. There is also 2 data folders 1) `data`, and 2) `ml` that respectively contains 1) the dataset, a sample json (for testing prediction), and a pickle file containing the split done for experimentation during the Exploratory Data Analysis (EDA), and 2) a machine learning model persisted with [joblib](https://joblib.readthedocs.io/en/latest/index.html).

The suggested order to explore this is:
1. `eda.ipynb` contains a simple Exploratory Data Analysis, simple analysis, and some experimentation preparation.
2. `experiment` package contains a simple `train.py` script which sould allow us to run a cross validation experiment over the provided data.
3. `experiment` package contains as imple `predict.py` script which should allow us to run a prediction over the sample json using the saved machine learning model (best model obtained during training).


## How to use this

Once you have this folder on your machine you will need [Poetry](https://python-poetry.org/). Install it if you don't have it already. You also need [Python](https://www.python.org/) version ">=3.10,<3.11".

I suggest you create a virtual environment for this, if you use `virtualenv` you can just run:

```bash
make env
```
and then activate the virtualen environment running `source venv/bin/activate`.

To install this application run:

```bash
make install
```

After installing Poetry, and succesfully running the command above, you should be able to run the application commands below:

To run an experiment:
```bash
python experiment/train.py
```

To run a prediction:
```bash
poetry run python experiment/predict.py
```

These commands should be run in the order presented here. As the first should generate a machine learning model file inside `ml` folder. The final prediction output of the `data_sample.json` will be generated inside the `ml` folder too, as `final_output.json`.


## Running the commands above

Running the commands above you should see logs similar to below:

```bash
» python experiment/train.py  
INFO:root:reading file: data/dataset_test_ds_v2-Atualizado.csv
INFO:root:saved the model with results below:
INFO:root:    balanced_accuracy_score: 0.566
INFO:root:    recall_score: 0.292

INFO:root:saved the model with results below:
INFO:root:    balanced_accuracy_score: 0.669
INFO:root:    recall_score: 0.500

INFO:root:saved the model with results below:
INFO:root:    balanced_accuracy_score: 0.683
INFO:root:    recall_score: 0.542

INFO:root:    balanced_accuracy_score: 0.536
INFO:root:    recall_score: 0.250

INFO:root:    balanced_accuracy_score: 0.585
INFO:root:    recall_score: 0.333
```


```bash
» poetry run python experiment/predict.py
INFO:root:validating payload file...
INFO:root:getting data from file: data/data_sample.json
INFO:root:getting data from file: ml/clf.joblib
INFO:root:running predictions...
INFO:root:[{0: 0}, {1: 1}]
```