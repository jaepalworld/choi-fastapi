import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import aiohttp
import firebase_admin

from firebase_admin import credentials, initialize_app, storage
import face
import httpx
import logging
# 배경화면 생성 import다
from pydantic import BaseModel
from typing import Optional
# backclear.py 파일 import
import backclear
import os
import backcreate
import HairStyle
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple


load_dotenv()


# Firebase 초기화 (주석 처리: 필요시 해제)
cred = credentials.Certificate('firebase-adminsdk-credentials.json')
initialize_app(cred, {
    'storageBucket': 'hairai-21bb9.firebasestorage.app'  
})
bucket = storage.bucket()

# ComfyUI 서버 주소
# os.environ['BACKCLEAR_WORKFLOW_PATH'] = 'D:/choi-fastapi/workflow/BackClear.json'
COMFYUI_API_URL = os.getenv('COMFYUI_API_URL')
BACKCLEAR_WORKFLOW_PATH = os.getenv('BACKCLEAR_WORKFLOW_PATH')
COMFYUI_OUTPUT_DIR = os.getenv('COMFYUI_OUTPUT_DIR')
BACKGROUND_WORKFLOW_PATH = os.getenv('BACKGROUND_WORKFLOW_PATH')
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
# 배경화면 생성 요청 모델
class BackgroundRequest(BaseModel):
    day: str
    gender: str
    category: str
    light: str
    info: str

# 헤어스타일 생성 요청 모델
class HairStyleRequest(BaseModel):
    hair_dir: str = "D:\\StabilityMatrix-win-x64\\HairImage"
    hair_count: int = 4
    face_restoration: float = 0.5
    batch_size: int = 4
    filename_prefix: str = "HairConsulting"

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

# 배경 생성 요청 모델 (기존 BackgroundRequest 모델 사용 또는 새로 정의)
class BackCreateRequest(BaseModel):
    positive_prompt: Optional[str] = "best quality, beautiful lighting, highly detailed background"
    negative_prompt: Optional[str] = "lowres, bad anatomy, bad hands, cropped, worst quality, nsfw"

    light_source: Optional[str] = "Top Left Light"
    light_color: Optional[str] = "#FFFFFF"
    light_intensity: Optional[float] = 1.0
    seed: Optional[int] = 2222
#여기서부터 배경화면 제거 엔드포인트
@app.post("/api/backclear")
async def remove_background(
    image: UploadFile = File(...)
):
    try:
        # 파일 크기 검증 (10MB 제한)
        file_size_limit = 10 * 1024 * 1024  # 10MB
        content = await image.read()
        if len(content) > file_size_limit:
            raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다: {image.filename}. 10MB 이하의 파일만 허용됩니다.")
        
        # 파일 포인터 위치 초기화
        await image.seek(0)
        
        # 파일 타입 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"잘못된 파일 형식: {image.filename}. 이미지 파일만 허용됩니다.")
        
        # ComfyUI 서버 상태 확인
        print(f"ComfyUI 서버 연결 테스트 중...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{COMFYUI_API_URL}/") as response:
                    print(f"ComfyUI 서버 응답: {response.status}")
                    if response.status != 200:
                        raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
            except aiohttp.ClientError:
                raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
        
        # 파일 콘텐츠 읽기
        content = await image.read()
        
        # 배경 제거 작업 시작
        print("배경 제거 작업 시작...")
        
        # 처리 시작
        result_image_filename = await backclear.process_background_removal(
            content,
            BACKCLEAR_WORKFLOW_PATH
        )
        
        if not result_image_filename:
            raise HTTPException(status_code=500, detail="배경 제거 실패: 결과 이미지를 찾을 수 없습니다.")
        
        # ComfyUI 출력 디렉토리에서 파일 경로 구성
        # 결과 이미지 파일 경로 확인
        comfy_output_path = os.path.join(COMFYUI_OUTPUT_DIR, result_image_filename)
        print(f"로컬 파일 경로: {comfy_output_path}")

        # 파일이 존재하는지 확인
        if not os.path.exists(comfy_output_path):
            print(f"경고: 로컬 파일을 찾을 수 없음: {comfy_output_path}")
            # 최근 생성된 파일 찾기
            newest_file = None
            newest_time = 0
            for file in os.listdir(COMFYUI_OUTPUT_DIR):
                file_path = os.path.join(COMFYUI_OUTPUT_DIR, file)
                file_time = os.path.getmtime(file_path)
                if file_time > newest_time:
                    newest_time = file_time
                    newest_file = file
            
            if newest_file:
                print(f"최근 생성된 파일 사용: {newest_file}")
                result_image_filename = newest_file
                comfy_output_path = os.path.join(COMFYUI_OUTPUT_DIR, result_image_filename)

        # 결과 URL 구성 및 Firebase 업로드
        result_url = f"{COMFYUI_API_URL}/view?filename={result_image_filename}&subfolder=&type=output"
        firebase_url = None

        if os.path.exists(comfy_output_path):
            try:
                firebase_path = f"removed_bg/{str(uuid.uuid4())}.png"
                firebase_url = upload_image_to_firebase(comfy_output_path, firebase_path)
                print(f"Firebase 업로드 성공: {firebase_url}")
            except Exception as e:
                print(f"Firebase 업로드 실패: {str(e)}")
                # Firebase 업로드 실패해도 ComfyUI URL은 반환

        # 응답 반환
        return {
            "status": "success",
            "result_image_url": result_url,
            "firebase_url": firebase_url,
            "message": "배경 제거가 완료되었습니다."
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("상세 오류:", error_detail)
        raise HTTPException(status_code=500, detail=f"배경 제거 실패: {str(e)}")

# new new new new new 이게 new 배경 생성 엔드포인트
@app.post("/api/background")
async def create_background(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # 파일 크기 검증 (10MB 제한)
        file_size_limit = 10 * 1024 * 1024  # 10MB
        content = await image.read()
        if len(content) > file_size_limit:
            raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다: {image.filename}. 10MB 이하의 파일만 허용됩니다.")
        
        # 파일 포인터 위치 초기화
        await image.seek(0)
        
        # 파일 타입 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"잘못된 파일 형식: {image.filename}. 이미지 파일만 허용됩니다.")
        
        # JSON 프롬프트 파싱
        try:
            parsed_data = json.loads(prompt)
            # 파싱된 데이터 검증
            if not isinstance(parsed_data, dict):
                raise HTTPException(status_code=400, detail="잘못된 프롬프트 형식. 유효한 JSON 객체가 필요합니다.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="잘못된 프롬프트 형식. 유효한 JSON이 필요합니다.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"프롬프트 파싱 오류: {str(e)}")
        
        # ComfyUI 서버 상태 확인
        print(f"ComfyUI 서버 연결 테스트 중...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{COMFYUI_API_URL}/") as response:
                    print(f"ComfyUI 서버 응답: {response.status}")
                    if response.status != 200:
                        raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
            except aiohttp.ClientError:
                raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
        
        # 파일 콘텐츠 읽기
        content = await image.read()
        
        # 배경 생성 작업 시작
        print("배경 생성 작업 시작...")
        
        # 워크플로우 파일 경로 설정
        workflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow', 'BackCreate.json')
        if not os.path.exists(workflow_path):
            workflow_path = os.getenv('BACKGROUND_WORKFLOW_PATH')
            if not workflow_path or not os.path.exists(workflow_path):
                raise HTTPException(status_code=404, detail="워크플로우 파일을 찾을 수 없습니다.")
        
        # 처리 시작
        result_image_filename = await backcreate.process_background_creation(
            content,
            parsed_data,
            workflow_path
        )
        
        if not result_image_filename:
            raise HTTPException(status_code=500, detail="배경 생성 실패: 결과 이미지를 찾을 수 없습니다.")
        
        # ComfyUI 출력 디렉토리에서 파일 경로 구성
        comfy_output_path = os.path.join(COMFYUI_OUTPUT_DIR, result_image_filename)
        print(f"로컬 파일 경로: {comfy_output_path}")
        
        # Firebase에 업로드 (background_results/ 폴더에 저장)
        firebase_path = f"background_created/{str(uuid.uuid4())}.png"
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
        
        # 응답 반환
        return {
            "status": "success",
            "result_image_url": f"{COMFYUI_API_URL}/view?filename={result_image_filename}&subfolder=&type=output",
            "firebase_url": firebase_url,
            "message": "배경 생성이 완료되었습니다."
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("상세 오류:", error_detail)
        raise HTTPException(status_code=500, detail=f"배경 생성 실패: {str(e)}")







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



# 이미지 제공 엔드포인트 추가
@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """이미지 파일을 제공합니다"""
    try:
        # 폴더 경로 정규화
        normalized_output_dir = COMFYUI_OUTPUT_DIR.replace('\\', '/')
        
        # 먼저 정확한 파일명으로 시도
        file_path = os.path.join(normalized_output_dir, filename).replace('\\', '/')
        print(f"요청 파일 경로: {file_path}")
        
        # 파일이 존재하지 않는 경우
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없음: {file_path}")
            
            # 출력 디렉토리의 모든 파일 가져오기
            try:
                all_files = os.listdir(normalized_output_dir)
                print(f"출력 디렉토리 내 파일 수: {len(all_files)}")
                print(f"몇 가지 파일명: {all_files[:5] if all_files else []}")
                
                # 요청된 파일명에 해당하는 패턴의 파일명 추출
                # 1. temp_kcogy_ 패턴 제거
                simple_filename = filename.replace("temp_kcogy_", "")
                
                # 2. 숫자 부분 추출
                import re
                num_match = re.search(r'_(\d+)_', filename)
                if num_match:
                    num_str = num_match.group(1)
                    print(f"추출한 숫자: {num_str}")
                    
                    # 숫자 부분을 패턴으로 맞는 파일 찾기
                    pattern_files = [f for f in all_files if num_str in f]
                    print(f"패턴 일치 파일: {pattern_files}")
                    
                    if pattern_files:
                        filename = pattern_files[0]
                        file_path = os.path.join(normalized_output_dir, filename).replace('\\', '/')
                        print(f"패턴 매칭으로 찾은 파일: {file_path}")
                
                # 3. 패턴 매칭으로 찾지 못했다면 가장 최근 파일 사용
                if not os.path.exists(file_path) and all_files:
                    all_files.sort(key=lambda x: os.path.getmtime(os.path.join(normalized_output_dir, x)), reverse=True)
                    newest_file = all_files[0]
                    file_path = os.path.join(normalized_output_dir, newest_file).replace('\\', '/')
                    print(f"최신 파일 사용: {file_path}")
            except Exception as e:
                print(f"디렉토리 탐색 중 오류: {str(e)}")
        
        # 파일이 존재하는지 최종 확인
        if os.path.exists(file_path):
            return FileResponse(file_path)
        else:
            # ComfyUI 서버에서 직접 가져오기 시도
            try:
                comfy_url = f"{COMFYUI_API_URL}/view?filename={filename}&subfolder=&type=output"
                async with aiohttp.ClientSession() as session:
                    async with session.get(comfy_url) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            return Response(content=image_data, media_type="image/png")
            except Exception as e:
                print(f"ComfyUI에서 직접 가져오기 실패: {str(e)}")
            
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    except Exception as e:
        print(f"이미지 제공 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 제공 실패: {str(e)}")
    
    
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

        # 결과 이미지 검색 - 최신 파일 우선 사용
        normalized_output_dir = COMFYUI_OUTPUT_DIR.replace('\\', '/')
        try:
            # 1. 먼저 ComfyUI가 반환한 파일명으로 시도
            result_image_url = output_images[0]  # ComfyUI에서 반환한 파일명
            comfy_output_path = os.path.join(normalized_output_dir, result_image_url).replace('\\', '/')
            
            # 2. 파일이 없으면 최신 파일 찾기
            if not os.path.exists(comfy_output_path):
                print(f"파일을 찾을 수 없음: {comfy_output_path}")
                
                # 디렉토리의 모든 파일 가져오기
                all_files = os.listdir(normalized_output_dir)
                
                # 파일이 있으면 최신 파일 사용
                if all_files:
                    # 수정 시간 기준으로 정렬
                    all_files.sort(key=lambda x: os.path.getmtime(os.path.join(normalized_output_dir, x)), reverse=True)
                    newest_file = all_files[0]
                    result_image_url = newest_file
                    comfy_output_path = os.path.join(normalized_output_dir, newest_file).replace('\\', '/')
                    print(f"최신 파일 사용: {comfy_output_path}")
        except Exception as e:
            print(f"결과 이미지 검색 중 오류: {str(e)}")

        # Firebase에 업로드
        firebase_url = None
        if os.path.exists(comfy_output_path):
            try:
                firebase_path = f"results/{str(uuid.uuid4())}.png"
                firebase_url = upload_image_to_firebase(comfy_output_path, firebase_path)
                print(f"Firebase 업로드 성공: {firebase_url}")
            except Exception as e:
                print(f"Firebase 업로드 실패: {str(e)}")

        # ComfyUI에서 직접 이미지 다운로드 시도
        try:
            comfy_url = f"{COMFYUI_API_URL}/view?filename={result_image_url}&subfolder=&type=output"
            print(f"ComfyUI에서 이미지 다운로드 시도: {comfy_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(comfy_url) as response:
                    if response.status == 200:
                        # 이미지 다운로드 성공
                        image_data = await response.read()
                        print(f"이미지 다운로드 성공: {len(image_data)} 바이트")
                        
                        # 이미지를 출력 디렉토리에 저장
                        os.makedirs(COMFYUI_OUTPUT_DIR, exist_ok=True)
                        output_path = os.path.join(COMFYUI_OUTPUT_DIR, result_image_url)
                        
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        print(f"이미지 저장 완료: {output_path}")
                        
                        # Firebase 업로드
                        firebase_url = None
                        try:
                            firebase_path = f"results/{str(uuid.uuid4())}.png"
                            firebase_url = upload_image_to_firebase(output_path, firebase_path)
                            print(f"Firebase 업로드 완료: {firebase_url}")
                        except Exception as e:
                            print(f"Firebase 업로드 실패: {str(e)}")
                        
                        # 응답 반환
                        return {
                            "status": "success",
                            "result_image_url": comfy_url,  # 완전한 ComfyUI URL 반환
                            "firebase_url": firebase_url,
                            "message": "얼굴 변환이 완료되었습니다."
                        }
        except Exception as e:
            print(f"ComfyUI에서 이미지 다운로드 실패: {str(e)}")

        # 이전 방식으로 포맷된 응답 유지
        return {
            "status": "success",
            "result_image_url": f"{COMFYUI_API_URL}/view?filename={result_image_url}&subfolder=&type=output",  # 전체 URL 사용
            "firebase_url": firebase_url,
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
async def load_workflow(workflow_path: str) -> dict:
    """
    워크플로우 JSON 파일을 로드하고 ComfyUI API 형식으로 변환합니다.
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as file:
            workflow_data = json.load(file)
        
        # ComfyUI 웹 UI 형식인 경우 (nodes와 links가 있는 경우)
        if "nodes" in workflow_data and "links" in workflow_data:
            print("ComfyUI 웹 UI 형식의 워크플로우를 API 형식으로 변환합니다...")
            
            # 노드 ID를 문자열로 변환하여 API 형식으로 준비
            api_workflow = {}
            
            # 1. 먼저 모든 노드를 기본 구조로 변환
            for node in workflow_data["nodes"]:
                node_id = str(node["id"])
                api_workflow[node_id] = {
                    "class_type": node["type"],
                    "inputs": {},
                    "outputs": {}
                }
                
                # 위젯 값이 있으면 추가
                if "widgets_values" in node:
                    for i, value in enumerate(node["widgets_values"]):
                        widget_name = f"widget_{i}"
                        api_workflow[node_id]["inputs"][widget_name] = value
            
            # 2. 링크 정보로 노드 연결 구성
            for link in workflow_data["links"]:
                # link 형식: [link_id, source_node_id, source_slot, target_node_id, target_slot, ...]
                if len(link) >= 5:  # 최소 5개 요소 필요
                    source_node_id = str(link[1])
                    source_slot = link[2]
                    target_node_id = str(link[3])
                    target_slot = link[4]
                    
                    # 소스 노드와 타겟 노드 찾기
                    if source_node_id in api_workflow and target_node_id in api_workflow:
                        # 타겟 노드의 입력에 소스 노드 연결
                        # 입력 이름 찾기 (노드 구조에 따라 달라질 수 있음)
                        source_node = next((n for n in workflow_data["nodes"] if str(n["id"]) == source_node_id), None)
                        target_node = next((n for n in workflow_data["nodes"] if str(n["id"]) == target_node_id), None)
                        
                        if source_node and target_node and "inputs" in target_node and len(target_node["inputs"]) > target_slot:
                            input_name = target_node["inputs"][target_slot]["name"]
                            api_workflow[target_node_id]["inputs"][input_name] = [source_node_id, source_slot]
            
            return api_workflow
        else:
            # 이미 API 형식인 경우
            return workflow_data
    except Exception as e:
        print(f"워크플로우 로딩 중 오류: {str(e)}")
        raise

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

# 헤어스타일 생성 엔드포인트
@app.post("/api/hairstyle")
async def generate_hairstyle(
    image: UploadFile = File(...),
    params: str = Form("{}"),
):
    try:
        # 파일 크기 검증 (10MB 제한)
        file_size_limit = 10 * 1024 * 1024  # 10MB
        content = await image.read()
        if len(content) > file_size_limit:
            raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다: {image.filename}. 10MB 이하의 파일만 허용됩니다.")
        
        # 파일 포인터 위치 초기화
        await image.seek(0)
        
        # 파일 타입 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"잘못된 파일 형식: {image.filename}. 이미지 파일만 허용됩니다.")
        
        # JSON 파라미터 파싱
        try:
            parsed_params = json.loads(params)
            # 파싱된 데이터 검증
            if not isinstance(parsed_params, dict):
                raise HTTPException(status_code=400, detail="잘못된 파라미터 형식. 유효한 JSON 객체가 필요합니다.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="잘못된 파라미터 형식. 유효한 JSON이 필요합니다.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"파라미터 파싱 오류: {str(e)}")
        
        # ComfyUI 서버 상태 확인
        logger.info(f"ComfyUI 서버 연결 테스트 중...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{COMFYUI_API_URL}/") as response:
                    logger.info(f"ComfyUI 서버 응답: {response.status}")
                    if response.status != 200:
                        raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
            except aiohttp.ClientError:
                raise HTTPException(status_code=503, detail="ComfyUI 서버에 연결할 수 없습니다.")
        
        # 파일 콘텐츠 읽기
        content = await image.read()
        
        # 헤어스타일 생성 작업 시작
        logger.info("헤어스타일 생성 작업 시작...")
        
        # 워크플로우 파일 경로 설정
        workflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow', 'HAIReditFinish.json')
        if not os.path.exists(workflow_path):
            workflow_path = os.getenv('HAIRSTYLE_WORKFLOW_PATH')
            if not workflow_path or not os.path.exists(workflow_path):
                raise HTTPException(status_code=404, detail="워크플로우 파일을 찾을 수 없습니다.")
        
        # 처리 시작 (타임아웃: 30분)
        result_images = await HairStyle.process_hairstyle_generation(
            content,
            parsed_params,
            workflow_path,
            timeout_minutes=30
        )
        
        if not result_images:
            raise HTTPException(status_code=500, detail="헤어스타일 생성 실패: 결과 이미지를 찾을 수 없습니다.")
        
        # 응답 반환
        return {
            "status": "success",
            "results": result_images,
            "message": "헤어스타일 생성이 완료되었습니다."
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error("상세 오류: %s", error_detail)
        raise HTTPException(status_code=500, detail=f"헤어스타일 생성 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)