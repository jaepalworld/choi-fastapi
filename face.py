import os
import json
import uuid
import websocket
import urllib.request
import urllib.parse
import asyncio
import aiohttp
from aiohttp import FormData
from typing import Dict, List, Tuple, Any, Optional, Union

# ComfyUI API 서버 주소
COMFYUI_API_URL = "http://127.0.0.1:8188"

async def open_websocket_connection() -> Tuple[websocket.WebSocket, str, str]:
    """
    ComfyUI 웹소켓 연결을 열고 웹소켓 객체, 서버 주소, 클라이언트 ID를 반환합니다.
    
    Returns:
        Tuple[websocket.WebSocket, str, str]: (웹소켓 객체, 서버 주소, 클라이언트 ID)
    """
    server_address = COMFYUI_API_URL
    client_id = str(uuid.uuid4())
    
    # 비동기 환경에서 동기 함수 호출을 피하기 위해 asyncio.to_thread 사용
    ws = await asyncio.to_thread(websocket.WebSocket)
    await asyncio.to_thread(ws.connect, f"ws://{server_address.replace('http://', '')}/ws?clientId={client_id}")
    
    return ws, server_address, client_id

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

async def process_face_transformation(
    original_image_path: str,
    role_model_image_path: str,
    workflow_path: str = "facedefault.json"
) -> Optional[str]:
    """
    얼굴 변환 처리를 수행합니다.
    
    Args:
        original_image_path (str): 원본 이미지 경로
        role_model_image_path (str): 롤모델 이미지 경로
        workflow_path (str, optional): 워크플로우 파일 경로. 기본값은 "facedefault.json".
        
    Returns:
        Optional[str]: 결과 이미지 파일명. 실패 시 None.
    """
    try:
        # 워크플로우 로드
        workflow = await load_workflow(workflow_path)
        
        # 이미지 로드 및 업로드
        with open(original_image_path, 'rb') as f:
            original_content = f.read()
        
        with open(role_model_image_path, 'rb') as f:
            role_model_content = f.read()
        
        # 임시 파일명 생성
        original_filename = f"original_{uuid.uuid4()}.png"
        role_model_filename = f"rolemodel_{uuid.uuid4()}.png"
        
        # ComfyUI에 이미지 업로드
        await upload_image(original_content, original_filename)
        await upload_image(role_model_content, role_model_filename)
        
        # 워크플로우 업데이트
        for node in workflow.get("nodes", []):
            if node.get("title") == "originalimage":
                node["widgets_values"][0] = original_filename
            elif node.get("title") == "rolemodelimage":
                node["widgets_values"][0] = role_model_filename
        
        # 웹소켓 연결 및 프롬프트 큐 추가
        client_id = str(uuid.uuid4())
        prompt_id = await queue_prompt(workflow, client_id)
        
        # 결과 대기
        result = await check_progress(prompt_id)
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                output_images.extend([img["filename"] for img in node_output["images"]])
        
        # 첫 번째 결과 이미지 반환
        return output_images[0] if output_images else None
    
    except Exception as e:
        print(f"얼굴 변환 처리 중 오류 발생: {str(e)}")
        return None

async def generate_image_by_prompt(workflow: Dict[str, Any], output_path: str, save_previews: bool = False) -> List[str]:
    """
    워크플로우를 사용하여 이미지를 생성합니다.
    
    Args:
        workflow (Dict[str, Any]): 워크플로우 데이터
        output_path (str): 출력 경로
        save_previews (bool, optional): 미리보기 저장 여부. 기본값은 False.
        
    Returns:
        List[str]: 생성된 이미지 파일 경로 목록
    """
    try:
        # 클라이언트 ID 생성 및 프롬프트 큐 추가
        client_id = str(uuid.uuid4())
        prompt_id = await queue_prompt(workflow, client_id)
        
        # 결과 대기
        result = await check_progress(prompt_id)
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    filename = img["filename"]
                    output_images.append(filename)
                    
                    # 이미지 다운로드 및 저장
                    image_data = await get_image(filename)
                    with open(os.path.join(output_path, filename), 'wb') as f:
                        f.write(image_data)
        
        return output_images
    
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {str(e)}")
        return []

async def modify_workflow(workflow: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    워크플로우를 수정합니다.
    
    Args:
        workflow (Dict[str, Any]): 워크플로우 데이터
        updates (Dict[str, Any]): 업데이트할 노드 및 값 ({node_id: {input_name: input_value}})
        
    Returns:
        Dict[str, Any]: 수정된 워크플로우
    """
    modified_workflow = workflow.copy()
    
    for node_id, node_updates in updates.items():
        if node_id in modified_workflow:
            for input_name, input_value in node_updates.items():
                if "inputs" in modified_workflow[node_id] and input_name in modified_workflow[node_id]["inputs"]:
                    modified_workflow[node_id]["inputs"][input_name] = input_value
    
    return modified_workflow

async def track_progress(prompt_id: str) -> Dict[str, Any]:
    """
    웹소켓을 통해 프롬프트 처리 진행 상황을 추적합니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        
    Returns:
        Dict[str, Any]: 최종 처리 결과
    """
    ws, server_address, client_id = await open_websocket_connection()
    
    try:
        # 첫 번째 메시지 수신 대기
        message = await asyncio.to_thread(ws.recv)
        
        execution_start = False
        execution_complete = False
        
        while not execution_complete:
            message = await asyncio.to_thread(ws.recv)
            data = json.loads(message)
            
            # 실행 시작 메시지 확인
            if data.get("type") == "execution_start" and data.get("data", {}).get("prompt_id") == prompt_id:
                execution_start = True
                print("워크플로우 실행 시작...")
            
            # 노드 실행 상태 메시지 확인
            elif data.get("type") == "executing" and execution_start:
                node_id = data.get("data", {}).get("node")
                print(f"노드 {node_id} 실행 중...")
            
            # 실행 완료 메시지 확인
            elif data.get("type") == "execution_complete" and data.get("data", {}).get("prompt_id") == prompt_id:
                execution_complete = True
                print("워크플로우 실행 완료")
        
        # 실행 완료 후 결과 가져오기
        return await check_progress(prompt_id)
    
    finally:
        # 웹소켓 연결 종료
        await asyncio.to_thread(ws.close)

async def get_workflow_info(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    워크플로우에 관한 정보를 추출합니다.
    
    Args:
        workflow (Dict[str, Any]): 워크플로우 데이터
        
    Returns:
        Dict[str, Any]: 노드 유형 및 입력 정보
    """
    node_info = {}
    
    for node_id, node in workflow.items():
        if isinstance(node, dict) and "class_type" in node:
            node_info[node_id] = {
                "type": node["class_type"],
                "inputs": node.get("inputs", {}),
                "widgets": node.get("widgets_values", [])
            }
    
    return node_info