from typing import Union

from fastapi import FastAPI

app = FastAPI()

#testing connection
@app.get("/")
def read_root():
    return {"Hello": "World"}