from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import util

app = FastAPI()


@app.post("/singer_predict")
async def singer_predict(request: Request):
    util.load_artifacts()
    form_data = await request.form()
    img_path = form_data.get("image_path")
    return {"predicted singer": util.classify_image(img_path)}
