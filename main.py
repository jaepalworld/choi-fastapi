import os
import uuid
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import aiohttp
import firebase_admin

from firebase_admin import credentials, initialize_app, storage
import face
import httpx
import logging

# Firebase 초기화 (주석 처리: 필요시 해제)
cred = credentials.Certificate('firebase-adminsdk-credentials.json')
initialize_app(cred, {
    'storageBucket': 'hairai-21bb9.firebasestorage.app'  # 여기를 수정
})
bucket = storage.bucket()

# ComfyUI 서버 주소
COMFYUI_API_URL = "http://127.0.0.1:8188"
COMFYUI_OUTPUT_DIR = "D:/StabilityMatrix-win-x64/Data/Packages/ComfyUI/output"  # ComfyUI 출력 디렉토리 경로 (실제 환경에 맞게 수정 필요)

app = FastAPI(title="AI Face Transformation API")

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS 미들웨어 설정 - 모든 출처 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase에 이미지 업로드 함수
def upload_image_to_firebase(local_image_path, destination_blob_name):
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_image_path)
    blob.make_public()
    return blob.public_url

# 기본 라우트
@app.get("/")
async def read_root():
    return {"message": "AI Face Transformation API에 오신 것을 환영합니다!"}

# 파일 업로드 엔드포인트
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
        # Firebase 연동 주석 처리
        blob = bucket.blob(f'images/{file.filename}')
        blob.upload_from_file(file.file)
        blob.make_public()
        return {"result": blob.public_url}
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 모든 이미지 URL 가져오기 엔드포인트
@app.get("/findAllImageUrl/")
async def find_all_image_url():
    try:
        # Firebase 연동 주석 처리
        blobs = bucket.list_blobs(prefix="images/")
        imageUrlList = []
        for blob in blobs:
            blob.make_public()
            imageUrlList.append(blob.public_url)
        return {"result": imageUrlList}
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 이미지 삭제 엔드포인트
@app.post("/deleteImage")
async def delete_by_image(imageUrl: str):
    try:
        # Firebase 연동 주석 처리
        blobs = bucket.list_blobs(prefix="createdImageUrl/")
        beforeDelete = []
        afterDelete = []
        for blob in blobs:
            beforeDelete.append(blob.public_url)
            blob.make_public()
            if blob.public_url == imageUrl:
                blob.delete()
            else:
                afterDelete.append(blob.public_url)
        return {
            "삭제 전 이미지 리스트": beforeDelete,
            "삭제 후 이미지 리스트": afterDelete
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 얼굴 변환 모델
class TransformRequest(BaseModel):
    original_image: Optional[str] = None
    role_model_image: Optional[str] = None

# 얼굴 변환 엔드포인트 부분의 코드 수정 (main.py 파일의 일부)
@app.post("/api/transform")
async def transform_face(
    original_image: UploadFile = File(...),
    role_model_image: UploadFile = File(...)
):
    try:
        # 파일 크기 검증 (10MB 제한)
        file_size_limit = 5 * 1024 * 1024  # 5MB
        for img_file in [original_image, role_model_image]:
            content = await img_file.read()
            if len(content) > file_size_limit:
                raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다: {img_file.filename}. 10MB 이하의 파일만 허용됩니다.")
            
            # 파일 포인터 위치 초기화
            await img_file.seek(0)
            
            # 파일 타입 검증
            if not img_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"잘못된 파일 형식: {img_file.filename}. 이미지 파일만 허용됩니다.")
        
        # ComfyUI 서버 상태 확인 로그 추가
        print(f"ComfyUI 서버 연결 테스트 중...")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_API_URL}/") as response:
                print(f"ComfyUI 서버 응답: {response.status}")
        
        # 파일 콘텐츠 읽기
        original_content = await original_image.read()
        role_model_content = await role_model_image.read()

        # 임시 파일명 생성
        original_filename = f"original_{uuid.uuid4()}.png"
        role_model_filename = f"rolemodel_{uuid.uuid4()}.png"

        # ComfyUI에 이미지 업로드
        await face.upload_image(original_content, original_filename)
        await face.upload_image(role_model_content, role_model_filename)

        # 워크플로우 직접 작성 (facedefault.json 파일 로드 대신)
        # ComfyUI API 형식으로 워크플로우 작성
        workflow = {
    "1": {
        "inputs": {
            "image": role_model_filename,  # 이미지 매개변수 추가
            "upload": True  # 업로드 플래그 추가
        },
        "class_type": "LoadImage",
        "outputs": {
            "IMAGE": ["IMAGE", 0],
            "MASK": ["MASK", 0]
        },
        "title": "rolemodelimage"
    },
    "2": {
        "inputs": {
            "image": original_filename,  # 이미지 매개변수 추가
            "upload": True  # 업로드 플래그 추가
        },
        "class_type": "LoadImage",
        "outputs": {
            "IMAGE": ["IMAGE", 1],
            "MASK": ["MASK", 1]
        },
        "title": "originalImage"
    },
    "3": {
        "inputs": {
            "input_image": ["1", 0],
            "source_image": ["2", 0],
            "console_log_level": 0,  # 문자열 '0'에서 정수 0으로 변경
            "face_restore_model": "GPEN-BFR-1024.onnx",
            "enabled": True,
            "codeformer_weight": 0.5,
            "input_faces_index": "0",
            "detect_gender_source": "no",
            "detect_gender_input": "no",
            "source_faces_index": "0",
            "facedetection": "YOLOv5l",
            "face_restore_visibility": 1,
            "swap_model": "inswapper_128.onnx"
        },
        "class_type": "ReActorFaceSwap",
        "outputs": {
            "IMAGE": ["IMAGE", 3],
            "FACE_MODEL": ["FACE_MODEL", 0]
        }
    },
    "4": {
        "inputs": {
            "images": ["3", 0]
        },
        "class_type": "PreviewImage",
        "outputs": {}
    },
    "5": {
        "inputs": {
            "images": ["3", 0],
            "filename_prefix": "ComfyUI"
        },
        "class_type": "SaveImage",
        "outputs": {}
    }
}
    

        # 클라이언트 ID 생성
        client_id = str(uuid.uuid4())
        
        # 프롬프트 큐에 추가
        print("워크플로우 전송 중...")
        prompt_id = await face.queue_prompt(workflow, client_id)
        
        # 결과 대기
        print(f"프롬프트 ID: {prompt_id}")
        print("결과 대기 중...")
        await asyncio.sleep(5)  # 초기 대기 시간
        
        # 결과 이미지 가져오기
        output_images = await get_output_images(prompt_id)
        print(f"DEBUG: 출력 이미지 목록: {output_images}")

        if not output_images:
            raise HTTPException(status_code=500, detail="변환 실패: 결과 이미지를 찾을 수 없습니다.")

        # 첫 번째 결과 이미지 URL 사용
        result_image_url = output_images[0]
        print(f"DEBUG: 결과 이미지 URL: {result_image_url}")
        
        # 결과 이미지 URL 사용
        result_image_url = output_images[0]
        print(f"DEBUG: 결과 이미지 URL: {result_image_url}")
        
        # ComfyUI 출력 디렉토리에서 파일 경로 구성
        comfy_output_path = os.path.join(COMFYUI_OUTPUT_DIR, result_image_url)
        print(f"DEBUG: 로컬 파일 경로: {comfy_output_path}")
        
        # Firebase에 업로드 (results/ 폴더에 저장)
        firebase_path = f"results/{str(uuid.uuid4())}.png"
        firebase_url = None
        
        # 파일이 존재하는지 확인
        if os.path.exists(comfy_output_path):
            print(f"Firebase 업로드 시도: {comfy_output_path} -> {firebase_path}")
            try:
                firebase_url = upload_image_to_firebase(comfy_output_path, firebase_path)
                print(f"Firebase 업로드 완료: {firebase_url}")
            except Exception as upload_error:
                print(f"Firebase 업로드 실패: {str(upload_error)}")
        else:
            print(f"경고: 로컬 파일을 찾을 수 없음: {comfy_output_path}")
        
        # 응답 반환 (기존 ComfyUI URL과 함께 Firebase URL도 반환)
        return {
            "status": "success",
            "result_image_url": f"{COMFYUI_API_URL}/view?filename={result_image_url}&subfolder=&type=output",
            "firebase_url": firebase_url,  # Firebase URL 추가
            "message": "얼굴 변환이 완료되었습니다."
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("상세 오류:", error_detail)
        raise HTTPException(status_code=500, detail=f"변환 실패: {str(e)}")

# 워크플로우 로딩 함수
async def load_workflow(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            workflow = json.load(file)
        return workflow
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"워크플로우 파일을 찾을 수 없습니다: '{file_path}'")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"파일 '{file_path}'이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크플로우 로딩 중 오류 발생: {str(e)}")

# 결과 이미지 URL 가져오기 함수
async def get_output_images(prompt_id: str):
    try:
        history_url = f"{COMFYUI_API_URL}/history/{prompt_id}"
        
        async with aiohttp.ClientSession() as session:
            max_attempts = 30  # 최대 30번 시도 (30초)
            for _ in range(max_attempts):
                async with session.get(history_url) as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            # 출력 노드에서 결과 이미지 찾기
                            for node_id, node_output in history[prompt_id].get("outputs", {}).items():
                                if "images" in node_output:
                                    return [img["filename"] for img in node_output["images"]]
                
                # 이미지가 준비되지 않았으면 1초 대기 후 재시도
                await asyncio.sleep(1)
                
        return []
    except Exception as e:
        print(f"결과 이미지 가져오기 실패: {str(e)}")
        return []

# 히스토리 항목 모델 - 결과 이미지만 저장
class HistoryItem(BaseModel):
    id: str  # 히스토리 항목의 고유 ID
    result_image: str  # 생성된 결과 이미지 URL
    timestamp: str  # 생성 시간
    is_saved: bool  # 저장 여부
    user_id: Optional[str] = None  # 사용자 ID (선택사항)

# 히스토리 저장 엔드포인트
@app.post("/api/history")
async def save_history_item(item: HistoryItem):
    try:
        # 실제 환경에서는 데이터베이스에 히스토리 저장
        # 현재는 메모리에 임시 저장 (실제 구현 시에는 데이터베이스 사용 필요)
        if not hasattr(app, "history_storage"):
            app.history_storage = []
        
        app.history_storage.append(item.dict())
        
        return {
            "status": "success", 
            "message": "히스토리 항목이 저장되었습니다.",
            "item_id": item.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 저장 실패: {str(e)}")

# 히스토리 조회 엔드포인트
@app.get("/api/history")
async def get_history(user_id: Optional[str] = None):
    try:
        if not hasattr(app, "history_storage"):
            app.history_storage = []
        
        # 사용자 ID로 필터링 (있는 경우)
        if user_id:
            items = [item for item in app.history_storage if item.get("user_id") == user_id]
        else:
            items = app.history_storage
        
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 조회 실패: {str(e)}")

# 히스토리 항목 삭제 엔드포인트
@app.delete("/api/history/{item_id}")
async def delete_history_item(item_id: str):
    try:
        if not hasattr(app, "history_storage"):
            app.history_storage = []
            return {"status": "success", "message": "히스토리가 비어있습니다."}
        
        # 항목 찾기 및 삭제
        initial_length = len(app.history_storage)
        app.history_storage = [item for item in app.history_storage if item.get("id") != item_id]
        
        if len(app.history_storage) < initial_length:
            return {"status": "success", "message": "히스토리 항목이 삭제되었습니다."}
        else:
            return {"status": "not_found", "message": "해당 ID의 히스토리 항목을 찾을 수 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 항목 삭제 실패: {str(e)}")

# HTTP 예외 핸들러 (응답 형식 일관성 유지)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "code": exc.status_code
        }
    )

# 일반 예외 핸들러
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
            "code": 500
        }
    )

# 워크플로우 변환 함수 - 참고용으로 남겨두지만 현재 사용하지는 않음
async def convert_workflow_to_api_format(workflow_data):
    """
    워크플로우 JSON 형식을 ComfyUI API가 요구하는 형식으로 변환합니다.
    
    Args:
        workflow_data (dict): 원본 워크플로우 데이터 (facedefault.json에서 로드됨)
        
    Returns:
        dict: ComfyUI API 형식의 워크플로우 데이터
        
    Raises:
        ValueError: 워크플로우 데이터가 유효하지 않은 경우
    """
    try:
        api_format = {}
        
        # nodes 배열 검증
        if not isinstance(workflow_data.get("nodes"), list):
            raise ValueError("유효하지 않은 워크플로우 형식: 'nodes' 배열이 필요합니다.")
            
        # 링크 매핑 생성 (성능 최적화)
        links_map = {}
        for link in workflow_data.get("links", []):
            # link: [link_id, source_node_id, source_output_idx, target_node_id, target_input_idx, type]
            links_map[link[0]] = {
                "source_node": str(link[1]),
                "source_output": link[2]
            }

        # 각 노드를 API 형식으로 변환
        for node in workflow_data["nodes"]:
            node_id = str(node.get("id"))
            if not node_id:
                continue
                
            # 기본 노드 구조 생성
            api_node = {
                "class_type": node.get("type", ""),
                "inputs": {},
                "outputs": {},
            }
            
            # 입력 처리
            for input_data in node.get("inputs", []):
                input_name = input_data.get("name")
                link_id = input_data.get("link")
                
                if input_name and link_id and link_id in links_map:
                    link_info = links_map[link_id]
                    api_node["inputs"][input_name] = [
                        link_info["source_node"],
                        link_info["source_output"]
                    ]
                elif input_name:
                    api_node["inputs"][input_name] = None
            
            # 출력 처리
            for output_data in node.get("outputs", []):
                output_name = output_data.get("name")
                if output_name:
                    api_node["outputs"][output_name] = [output_name.upper(), 0]
            
            # 추가 속성 복사
            for field in ["title", "properties", "widgets_values"]:
                if field in node:
                    api_node[field] = node[field]
            
            api_format[node_id] = api_node
            
        return api_format
        
    except Exception as e:
        raise ValueError(f"워크플로우 변환 중 오류 발생: {str(e)}")

@app.get("/test-comfyui")
async def test_comfyui_connection():
    """ComfyUI 서버 연결 테스트"""
    try:
        logger.info("ComfyUI 연결 테스트를 시작합니다...")
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8188/")  # ComfyUI는 8188 포트 사용
            logger.info(f"ComfyUI 응답 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("ComfyUI 연결 성공!")
                return {"status": "success", "message": "ComfyUI 서버에 성공적으로 연결되었습니다!"}
            else:
                logger.error(f"ComfyUI 연결 실패: 상태 코드 {response.status_code}")
                return {"status": "error", "message": f"ComfyUI 연결 실패: 상태 코드 {response.status_code}"}
    except Exception as e:
        error_msg = f"ComfyUI 연결 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)