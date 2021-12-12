from .model import preprocessing
from keras.models import load_model
from fastapi import FastAPI, Response
from pydantic import BaseModel
from .helper import api

import numpy as np
import pickle

app = FastAPI()


class Input(BaseModel):
    text: str


@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Sentiment API Works!", response.status_code)


@app.post("/predict", status_code=200)
def get_prediction(req: Input, response: Response):
    model = load_model('app/model/model.h5')

    with open('app/model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    text, result = preprocessing.convert_input_to_sequences(req.text, 25, model, tokenizer)

    predict = "Negatif"

    if result[0][0] < result[0][1]:
        predict = "Positif"

    return api.builder({
        'preprocess': text,
        'confidence': np.array(result[0]).tolist(),
        'predict': predict},
        response.status_code
    )
