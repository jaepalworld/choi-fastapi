from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import json
import uvicorn
from PIL import Image
import io
import base64
import logging
import os
from datetime import datetime
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과 저장 디렉토리 설정
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 정적 파일 설정
app.mount("/results", StaticFiles(directory="results"), name="results")

COMFY_API_URL = "http://127.0.0.1:8188"

@app.post("/upload-images")
async def upload_images(
    face_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    style_option: str = "natural"  # 기본값 설정
):
    try:
        # 세션 ID 생성
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 이미지 읽기
        face_bytes = await face_image.read()
        product_bytes = await product_image.read()
        
        # 이미지 형식 검증
        try:
            Image.open(io.BytesIO(face_bytes))
            Image.open(io.BytesIO(product_bytes))
        except:
            return {"status": "error", "message": "Invalid image format"}

        # ComfyUI 워크플로우 실행
        face_b64 = base64.b64encode(face_bytes).decode('utf-8')
        product_b64 = base64.b64encode(product_bytes).decode('utf-8')
        
        workflow = {
            "3": {
                "inputs": {
                    "image": face_b64,
                    "choose file to upload": f"face_{session_id}.png"
                },
                "class_type": "LoadImage"
            },
            "4": {
                "inputs": {
                    "image": product_b64,
                    "choose file to upload": f"product_{session_id}.png"
                },
                "class_type": "LoadImage"
            }
            # 추가 노드 설정
        }

        # ComfyUI API 호출
        response = requests.post(
            f"{COMFY_API_URL}/api/queue",
            json={"prompt": workflow}
        )
        
        if response.status_code == 200:
            # 결과 이미지 저장 (예시)
            result_path = RESULTS_DIR / f"result_{session_id}.png"
            
            # 결과 기록 저장
            save_history(session_id, style_option, str(result_path))
            
            return {
                "status": "success", 
                "data": response.json(),
                "session_id": session_id,
                "result_url": f"/results/result_{session_id}.png"
            }
        return {"status": "error", "message": "ComfyUI processing failed"}

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

# 히스토리 데이터를 저장할 리스트 (임시 저장소)
history_data = []

def save_history(session_id: str, style_option: str, result_path: str):
    """히스토리 데이터 저장"""
    history_data.append({
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "style_option": style_option,
        "result_url": f"/results/result_{session_id}.png"
    })

@app.get("/history")
async def get_history():
    """히스토리 조회"""
    try:
        return history_data
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"message": "ComfyUI Test Server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)