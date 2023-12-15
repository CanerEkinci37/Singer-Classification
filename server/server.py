from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import util

app = FastAPI()


@app.post("/singer_predict")
async def singer_predict(file: UploadFile = File(...)):
    util.load_artifacts()
    return util.classify_image(file)
