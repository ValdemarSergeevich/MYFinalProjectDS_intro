import json

import dill

import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

with open('c:/Users/A315-23-R7CZ/final_project_ds_intro/model/event_action_pipe.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    session_id: str
    client_id: str
    visit_time: str
    visit_number: int
    utm_medium: str
    device_category: str
    device_brand: str
    device_browser: str
    geo_country: str
    geo_city: str
    event_action: str

class Prediction(BaseModel):
    session_id: str
    client_id: str
    event_action: int


@app.get('/status')

def status():
    return "I am Ok!"

@app.get('/version')

def version():
    return model['metadata']
#
#
#
@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'client_id': form.client_id,
        'event_action': y[0]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, debug=True)
