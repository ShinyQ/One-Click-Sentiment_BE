from .model import preprocessing
from keras.models import load_model
from fastapi import FastAPI, Response
from pydantic import BaseModel
from .helper import api
import numpy as np

app = FastAPI()


class Input(BaseModel):
    text: str


@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Sentiment API Works!", response.status_code)


@app.post("/predict", status_code=200)
def get_prediction(req: Input, response: Response):
    model = load_model('app/model/model.h5')
    text, result = preprocessing.convert_input_to_sequences(req.text, 35, model)

    predict = result[0][0]

    if result[0][0] < result[0][1]:
        predict = result[0][1]

    return api.builder({
        'preprocess': text,
        'confidence': np.array(result[0]).tolist(),
        'predict': float(predict)},
        response.status_code
    )
