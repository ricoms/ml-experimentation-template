import errno
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from experiment import DateTransformer, data_dtype

logging.basicConfig(level=logging.INFO)


class InputSchema(BaseModel):
    SAFRA: str
    V1: float | None
    V2: float
    V3: float
    V4: float
    V5: float | None
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: str
    V12: str
    CEP: str


def validate_payload(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        data_file = kwargs.get("data_file") if kwargs.get("data_file", False) else args[0]
        logging.info(f"validating payload file...")
        with open(data_file) as f:
            payload = json.load(f)
        for inputs in payload:
            InputSchema(**inputs)
        return func(*args, **kwargs)
    return wrapper
    

def check_file(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        data_file = kwargs.get("data_file") if kwargs.get("data_file", False) else args[0]
        if not data_file.is_file():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(data_file)
            )
        logging.info(f"getting data from file: {data_file}")
        return func(*args, **kwargs)
    return wrapper

@validate_payload
@check_file
def read_payload(data_file: Path) -> pd.DataFrame:
    return pd.read_json(data_file, orient="records", dtype=data_dtype())

@check_file
def load_model(data_file: Path) -> BaseEstimator:
    return joblib.load(data_file)


if __name__ == "__main__":
    project_folder: Path = Path(__file__).parent.parent
    output_folder: Path = project_folder / "ml"
    model_file: Path = project_folder / "ml" / "clf.joblib"
    sample_file: Path = project_folder / "data" / "data_sample.json"
    
    df = read_payload(data_file=sample_file)
    clf = load_model(model_file)
    logging.info("running predictions...")
    predictions = [{i: pred} for i, pred in enumerate(clf.predict(df).tolist())]
    logging.info(predictions)
    with open(output_folder / "final_output.json", 'w') as jsonf:
        jsonf.write(json.dumps(predictions, indent=4))