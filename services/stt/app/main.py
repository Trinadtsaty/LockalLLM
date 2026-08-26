import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.stt_service import STTService



app = FastAPI(
    title="Local STT Service",
    version="1.0.0",
)


stt = STTService()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "stt",
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Файл не передан",
        )

    suffix = os.path.splitext(file.filename)[1]

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            content = await file.read()

            temp_file.write(content)

            temp_path = temp_file.name

        result = stt.transcribe(temp_path)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка распознавания: {str(e)}",
        )

    finally:

        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
