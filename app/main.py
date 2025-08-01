from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .analyze import analyze_face_image

app = FastAPI()

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        result = analyze_face_image(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
