from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from .inference import predict


app = FastAPI()


@app.post("/mask/")
async def create_mask(file: UploadFile = File(...)):
    """An Endpoint to put image and get masked one"""
    preds = await predict(file)
    return FileResponse(preds)
