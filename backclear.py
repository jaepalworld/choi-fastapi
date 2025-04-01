import os
import json
import uuid
import asyncio
import aiohttp
from aiohttp import FormData
from typing import Dict, List, Tuple, Any, Optional, Union

# ComfyUI API 서버 주소
COMFYUI_API_URL = "http://127.0.0.1:8188"

async def upload_image(file_content: bytes, filename: str, image_type: str = "input", overwrite: bool = True) -> Dict[str, Any]:
    """
    ComfyUI에 이미지를 업로드합니다.
    
    Args:
        file_content (bytes): 파일 내용
        filename (str): 파일명
        image_type (str, optional): 이미지 유형. 기본값은 "input".
        overwrite (bool, optional): 덮어쓰기 여부. 기본값은 True.
        
    Returns:
        Dict[str, Any]: 업로드 결과
    """
    url = f"{COMFYUI_API_URL}/upload/image"
    
    data = FormData()
    data.add_field('image', file_content, filename=filename, content_type='image/png')
    data.add_field('type', image_type)
    data.add_field('overwrite', str(overwrite).lower())
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"이미지 업로드 실패: {error_text}")
            
            return await response.json()

async def queue_prompt(workflow: Dict[str, Any], client_id: str = "") -> str:
    """
    ComfyUI에 프롬프트를 큐에 추가하고 프롬프트 ID를 반환합니다.
    
    Args:
        workflow (Dict[str, Any]): 워크플로우 데이터
        client_id (str, optional): 클라이언트 ID. 기본값은 빈 문자열.
        
    Returns:
        str: 프롬프트 ID
    """
    if not client_id:
        client_id = str(uuid.uuid4())
    
    prompt_url = f"{COMFYUI_API_URL}/prompt"
    
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(prompt_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"프롬프트 큐 추가 실패: {error_text}")
            
            result = await response.json()
            return result.get('prompt_id')

async def check_progress(prompt_id: str) -> Dict[str, Any]:
    """
    프롬프트 처리 상태를 확인하고 결과를 반환합니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        
    Returns:
        Dict[str, Any]: 프롬프트 처리 결과
    """
    history_url = f"{COMFYUI_API_URL}/history/{prompt_id}"
    
    async with aiohttp.ClientSession() as session:
        max_attempts = 60  # 최대 60번 시도 (60초)
        for _ in range(max_attempts):
            async with session.get(history_url) as response:
                if response.status == 200:
                    history = await response.json()
                    if prompt_id in history:
                        return history[prompt_id]
            
            # 결과가 없으면 1초 기다린 후 재시도
            await asyncio.sleep(1)
    
    raise Exception(f"프롬프트 {prompt_id}의 처리 결과를 가져오지 못했습니다.")

async def get_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """
    ComfyUI에서 이미지를 가져옵니다.
    
    Args:
        filename (str): 파일명
        subfolder (str, optional): 하위 폴더. 기본값은 빈 문자열.
        folder_type (str, optional): 폴더 유형. 기본값은 "output".
        
    Returns:
        bytes: 이미지 바이너리 데이터
    """
    import urllib.parse
    
    params = {
        'filename': filename,
        'subfolder': subfolder,
        'type': folder_type
    }
    
    url_values = urllib.parse.urlencode(params)
    url = f"{COMFYUI_API_URL}/view?{url_values}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"이미지 다운로드 실패: {error_text}")
            
            return await response.read()

async def load_workflow(workflow_path: str) -> Dict[str, Any]:
    """
    워크플로우 JSON 파일을 로드합니다.
    
    Args:
        workflow_path (str): 워크플로우 파일 경로
        
    Returns:
        Dict[str, Any]: 워크플로우 데이터
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as file:
            workflow = json.load(file)
            return workflow
    except FileNotFoundError:
        raise Exception(f"워크플로우 파일을 찾을 수 없습니다: '{workflow_path}'")
    except json.JSONDecodeError:
        raise Exception(f"파일 '{workflow_path}'이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        raise Exception(f"워크플로우 로딩 중 오류 발생: {str(e)}")

async def process_background_removal(
    image_data: bytes,
    workflow_path: str
) -> Optional[str]:
    """
    배경 제거 처리를 수행합니다.
    
    Args:
        image_data (bytes): 이미지 데이터
        workflow_path (str): 워크플로우 파일 경로
        
    Returns:
        Optional[str]: 결과 이미지 파일명. 실패 시 None.
    """
    try:
        # 워크플로우 로드
        print(f"워크플로우 파일 로드: {workflow_path}")
        workflow = await load_workflow(workflow_path)
        
        # 이미지 파일명 생성
        image_filename = f"input_{uuid.uuid4()}.png"
        print(f"생성된 이미지 파일명: {image_filename}")
        
        # ComfyUI에 이미지 업로드
        print("이미지 업로드 중...")
        await upload_image(image_data, image_filename)
        print("이미지 업로드 완료")
        
        # LoadImage 노드 업데이트 (이미지 파일명 설정)
        if '1' in workflow:
            print(f"LoadImage 노드 업데이트: {image_filename}")
            workflow['1']['inputs']['image'] = image_filename
            # 업로드 플래그 설정 (이미 업로드했으므로 false)
            workflow['1']['inputs']['upload'] = False
        else:
            print("경고: LoadImage 노드를 찾을 수 없음")
        
        # 클라이언트 ID 생성
        client_id = str(uuid.uuid4())
        
        # 워크플로우 내용 디버깅
        print("워크플로우 내용:")
        print(json.dumps(workflow, indent=2))
        
        # 프롬프트 큐에 추가
        print("워크플로우 전송 중...")
        prompt_id = await queue_prompt(workflow, client_id)
        print(f"프롬프트 ID: {prompt_id}")
        
        # 결과 대기
        print("결과 대기 중...")
        result = await check_progress(prompt_id)
        
        # 결과 검사
        if not result:
            print("오류: 결과가 없음")
            return None
            
        print("결과 수신됨, 출력 찾는 중...")
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    output_images.append(img["filename"])
                    print(f"결과 이미지 발견: {img['filename']}")
        
        # 첫 번째 결과 이미지 반환
        if output_images:
            print(f"최종 결과 이미지: {output_images[0]}")
            return output_images[0]
        else:
            print("오류: 결과 이미지 없음")
            return None
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"배경 제거 처리 중 오류 발생: {str(e)}")
        print(f"상세 오류 정보: {error_detail}")
        return None

async def get_output_images(prompt_id: str) -> List[str]:
    """
    프롬프트 ID로부터 결과 이미지 파일명 목록을 가져옵니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        
    Returns:
        List[str]: 결과 이미지 파일명 목록
    """
    try:
        # 결과 가져오기
        result = await check_progress(prompt_id)
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                output_images.extend([img["filename"] for img in node_output["images"]])
        
        return output_images
    
    except Exception as e:
        print(f"결과 이미지 가져오기 실패: {str(e)}")
        return []