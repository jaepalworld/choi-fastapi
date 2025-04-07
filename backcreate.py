import os
import json
import uuid
import asyncio
import aiohttp
import logging
from aiohttp import FormData
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

# ComfyUI API 서버 주소
COMFYUI_API_URL = os.getenv('COMFYUI_API_URL', "http://127.0.0.1:8188")
BACKGROUND_WORKFLOW_PATH = os.getenv('BACKGROUND_WORKFLOW_PATH', "D:/choi-fastapi/workflow/BackCreate.json")
COMFYUI_OUTPUT_DIR = os.getenv('COMFYUI_OUTPUT_DIR', "output")

# 노드 ID 상수 정의 (워크플로우 파일에서 식별된 주요 노드)
NODE_IDS = {
    "POSITIVE_PROMPT": "6",    # 긍정 프롬프트 노드 ID
    "NEGATIVE_PROMPT": "7",    # 부정 프롬프트 노드 ID
    "LOAD_IMAGE": "15",        # 이미지 로드 노드 ID
    "KSAMPLER": "3",           # KSampler 노드 ID
    "VAEDECODE": "8",          # VAEDecode 노드 ID
    "SAVE_IMAGE": "9",         # SaveImage 노드 ID
    "VAELOADER": "22",         # VAELoader 노드 ID
    "CHECKPOINT_LOADER": "4",   # CheckpointLoaderSimple 노드 ID
    "BRIAAI_MATTING": "16",     # BRIAAI Matting 노드 ID
    "CONSTRAIN_IMAGE": "23",    # ConstrainImage 노드 ID
}

# 커스텀 예외 클래스 정의
class ComfyUIError(Exception):
    """ComfyUI 관련 기본 예외 클래스"""
    pass

class WorkflowError(ComfyUIError):
    """워크플로우 관련 예외"""
    pass

class ImageProcessingError(ComfyUIError):
    """이미지 처리 관련 예외"""
    pass

class APIError(ComfyUIError):
    """API 통신 관련 예외"""
    pass

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
        
    Raises:
        APIError: API 통신 실패 시
        ImageProcessingError: 이미지 처리 실패 시
    """
    url = f"{COMFYUI_API_URL}/upload/image"
    logger.info(f"이미지 업로드 시작: {filename} (type: {image_type})")
    
    try:
        data = FormData()
        data.add_field('image', file_content, filename=filename, content_type='image/png')
        data.add_field('type', image_type)
        data.add_field('overwrite', str(overwrite).lower())
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"이미지 업로드 실패 (상태 코드: {response.status}): {error_text}")
                    raise APIError(f"이미지 업로드 실패: {error_text}")
                
                result = await response.json()
                logger.info(f"이미지 업로드 성공: {filename}")
                return result
                
    except aiohttp.ClientError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise APIError(f"API 통신 오류: {str(e)}")
    except Exception as e:
        logger.error(f"이미지 업로드 중 예상치 못한 오류: {str(e)}")
        raise ImageProcessingError(f"이미지 업로드 중 오류 발생: {str(e)}")

async def queue_prompt(workflow: Dict[str, Any], client_id: str = "") -> str:
    """
    ComfyUI에 프롬프트를 큐에 추가하고 프롬프트 ID를 반환합니다.
    
    Args:
        workflow (Dict[str, Any]): 워크플로우 데이터
        client_id (str, optional): 클라이언트 ID. 기본값은 빈 문자열.
        
    Returns:
        str: 프롬프트 ID
        
    Raises:
        APIError: API 통신 실패 시
        WorkflowError: 워크플로우 처리 실패 시
    """
    if not client_id:
        client_id = str(uuid.uuid4())
    
    prompt_url = f"{COMFYUI_API_URL}/prompt"
    logger.info(f"프롬프트 큐 추가 시작 (client_id: {client_id})")
    
    try:
        payload = {
            "prompt": workflow,
            "client_id": client_id
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(prompt_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"프롬프트 큐 추가 실패 (상태 코드: {response.status}): {error_text}")
                    raise APIError(f"프롬프트 큐 추가 실패: {error_text}")
                
                result = await response.json()
                prompt_id = result.get('prompt_id')
                
                if not prompt_id:
                    logger.error("프롬프트 ID가 없음")
                    raise WorkflowError("프롬프트 ID를 받지 못했습니다")
                
                logger.info(f"프롬프트 큐 추가 성공 (prompt_id: {prompt_id})")
                return prompt_id
                
    except aiohttp.ClientError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise APIError(f"API 통신 오류: {str(e)}")
    except Exception as e:
        logger.error(f"프롬프트 큐 추가 중 예상치 못한 오류: {str(e)}")
        raise WorkflowError(f"프롬프트 큐 추가 중 오류 발생: {str(e)}")

async def check_progress(prompt_id: str) -> Dict[str, Any]:
    """
    프롬프트 처리 상태를 확인하고 결과를 반환합니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        
    Returns:
        Dict[str, Any]: 프롬프트 처리 결과
        
    Raises:
        APIError: API 통신 실패 시
        WorkflowError: 워크플로우 처리 실패 시
    """
    history_url = f"{COMFYUI_API_URL}/history/{prompt_id}"
    logger.info(f"프롬프트 진행 상태 확인 시작 (prompt_id: {prompt_id})")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 시간 초과 설정을 180초(30분)으로 늘림
            max_attempts = 1800  # 최대 1800번 시도 (30분)
            for attempt in range(max_attempts):
                async with session.get(history_url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"진행 상태 확인 실패 (상태 코드: {response.status}): {error_text}")
                        raise APIError(f"진행 상태 확인 실패: {error_text}")
                    
                    history = await response.json()
                    if prompt_id in history:
                        logger.info(f"프롬프트 처리 완료 (prompt_id: {prompt_id})")
                        return history[prompt_id]
                
                # 진행 상태 로그 (30초마다)
                if attempt % 30 == 0 and attempt > 0:
                    logger.info(f"배경 생성 진행 중... ({attempt}초 경과)")
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5) # 호출간격 늘리는거
            
            logger.error(f"프롬프트 처리 시간 초과 (prompt_id: {prompt_id})")
            raise WorkflowError(f"프롬프트 {prompt_id}의 처리 결과를 가져오지 못했습니다 (시간 초과: 5분)")
            
    except aiohttp.ClientError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise APIError(f"API 통신 오류: {str(e)}")
    except Exception as e:
        logger.error(f"진행 상태 확인 중 예상치 못한 오류: {str(e)}")
        raise WorkflowError(f"진행 상태 확인 중 오류 발생: {str(e)}")

async def get_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """
    ComfyUI에서 이미지를 가져옵니다.
    
    Args:
        filename (str): 파일명
        subfolder (str, optional): 하위 폴더. 기본값은 빈 문자열.
        folder_type (str, optional): 폴더 유형. 기본값은 "output".
        
    Returns:
        bytes: 이미지 바이너리 데이터
        
    Raises:
        APIError: API 통신 실패 시
        ImageProcessingError: 이미지 처리 실패 시
    """
    import urllib.parse
    
    params = {
        'filename': filename,
        'subfolder': subfolder,
        'type': folder_type
    }
    
    url_values = urllib.parse.urlencode(params)
    url = f"{COMFYUI_API_URL}/view?{url_values}"
    logger.info(f"이미지 다운로드 시작: {filename} (folder: {subfolder}, type: {folder_type})")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"이미지 다운로드 실패 (상태 코드: {response.status}): {error_text}")
                    raise APIError(f"이미지 다운로드 실패: {error_text}")
                
                image_data = await response.read()
                if not image_data:
                    logger.error("다운로드된 이미지 데이터가 비어있음")
                    raise ImageProcessingError("다운로드된 이미지 데이터가 비어있습니다")
                
                logger.info(f"이미지 다운로드 성공: {filename}")
                return image_data
                
    except aiohttp.ClientError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise APIError(f"API 통신 오류: {str(e)}")
    except Exception as e:
        logger.error(f"이미지 다운로드 중 예상치 못한 오류: {str(e)}")
        raise ImageProcessingError(f"이미지 다운로드 중 오류 발생: {str(e)}")

async def load_workflow(workflow_path: str) -> Dict[str, Any]:
    """
    워크플로우 JSON 파일을 로드하고 ComfyUI API 형식으로 변환합니다.
    
    Args:
        workflow_path (str): 워크플로우 파일 경로
        
    Returns:
        Dict[str, Any]: 워크플로우 데이터
        
    Raises:
        WorkflowError: 워크플로우 파일 로드 또는 변환 실패 시
    """
    logger.info(f"워크플로우 파일 로드 시작: {workflow_path}")
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as file:
            workflow_data = json.load(file)
            
        # ComfyUI API 형식의 워크플로우를 구성
        if "nodes" in workflow_data:
            logger.info("워크플로우 변환 시작")
            # 노드 정보를 API 형식으로 변환
            api_workflow = {}
            
            # 노드 유형별 위젯 매핑 정의
            widget_mappings = {
                "VAELoader": ["vae_name"],
                "CheckpointLoaderSimple": ["ckpt_name"],
                "BRIAAI Matting": ["version", "fp16", "bg_color", "batch_size"],
                "ConstrainImage|pysssss": ["max_width", "max_height", "min_width", "min_height", "crop_if_required"],
                "KSampler": ["seed", "sampler_name", "steps", "cfg", "scheduler", "denoise"],
                "CLIPTextEncode": ["text"],
                "SaveImage": ["filename_prefix"]
            }
            
            # 1단계: 노드 기본 구조 생성 및 위젯 값 매핑
            for node in workflow_data.get("nodes", []):
                node_id = str(node.get("id"))
                node_type = node.get("type", "")
                
                if not node_id:
                    logger.warning("노드 ID가 없는 노드 발견, 건너뜀")
                    continue
                    
                api_node = {
                    "class_type": node_type,
                    "inputs": {},
                    "outputs": {}
                }
                
                # 위젯 값이 있으면 매핑에 따라 추가
                if "widgets_values" in node and node_type in widget_mappings:
                    widget_names = widget_mappings[node_type]
                    for i, value in enumerate(node["widgets_values"]):
                        if i < len(widget_names):
                            input_name = widget_names[i]
                            api_node["inputs"][input_name] = value
                        else:
                            # 매핑되지 않은 위젯은 기본 이름 사용
                            api_node["inputs"][f"widget_{i}"] = value
                
                api_workflow[node_id] = api_node
            
            # 2단계: 링크 정보 추가
            for link in workflow_data.get("links", []):
                # link: [link_id, source_node_id, source_output_idx, target_node_id, target_input_idx, type]
                if len(link) >= 5:
                    source_node_id = str(link[1])
                    target_node_id = str(link[3])
                    source_output_idx = link[2]
                    target_input_idx = link[4]
                    
                    # 링크 추가
                    if source_node_id in api_workflow and target_node_id in api_workflow:
                        # 타겟 노드와 입력 정보 찾기
                        target_node = next((n for n in workflow_data["nodes"] if str(n["id"]) == target_node_id), None)
                        
                        if target_node and "inputs" in target_node and target_input_idx < len(target_node["inputs"]):
                            input_name = target_node["inputs"][target_input_idx].get("name")
                            if input_name:
                                api_workflow[target_node_id]["inputs"][input_name] = [source_node_id, source_output_idx]
            
            # 3단계: 특정 노드의 중요 입력값 확인 및 설정
            for node_id, node_type in [
                (NODE_IDS["VAEDECODE"], "VAEDecode"),
                (NODE_IDS["SAVE_IMAGE"], "SaveImage"),
                (NODE_IDS["KSAMPLER"], "KSampler"),
                (NODE_IDS["BRIAAI_MATTING"], "BRIAAI Matting"),
                (NODE_IDS["CONSTRAIN_IMAGE"], "ConstrainImage"),
                (NODE_IDS["VAELOADER"], "VAELoader"),
                (NODE_IDS["CHECKPOINT_LOADER"], "CheckpointLoaderSimple")
            ]:
                if node_id in api_workflow:
                    logger.debug(f"{node_type} 노드({node_id}) 설정 확인")
                    # 각 노드 타입별 기본값 설정은 그대로 유지
            
            logger.info("워크플로우 변환 완료")
            return api_workflow
        else:
            # 이미 API 형식인 경우
            logger.info("이미 API 형식의 워크플로우")
            return workflow_data
            
    except FileNotFoundError:
        logger.error(f"워크플로우 파일을 찾을 수 없음: {workflow_path}")
        raise WorkflowError(f"워크플로우 파일을 찾을 수 없습니다: '{workflow_path}'")
    except json.JSONDecodeError:
        logger.error(f"워크플로우 파일이 유효한 JSON 형식이 아님: {workflow_path}")
        raise WorkflowError(f"파일 '{workflow_path}'이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        logger.error(f"워크플로우 로딩 중 예상치 못한 오류: {str(e)}")
        import traceback
        error_detail = traceback.format_exc()
        raise WorkflowError(f"워크플로우 로딩 중 오류 발생: {str(e)}\n{error_detail}")

async def update_workflow_parameters(
    workflow: Dict[str, Any],
    image_filename: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    워크플로우 파라미터를 동적으로 업데이트합니다.
    
    Args:
        workflow (Dict[str, Any]): 원본 워크플로우 데이터
        image_filename (str): 업로드된 이미지 파일명
        params (Dict[str, Any]): 업데이트할 파라미터
            - positive_prompt (str): 긍정 프롬프트 텍스트
            - negative_prompt (str): 부정 프롬프트 텍스트
            - seed (int, optional): 시드값 (기본값: 랜덤)
            - light_position (str, optional): 조명 위치 (기본값: "Top Left Light")
            
    Returns:
        Dict[str, Any]: 업데이트된 워크플로우 데이터
    """
    # 워크플로우의 깊은 복사본 생성
    updated_workflow = json.loads(json.dumps(workflow))
    
    # 1. LoadImage 노드 업데이트 (노드 15)
    if "15" in updated_workflow:
        updated_workflow["15"]["inputs"] = {
            "image": image_filename,
            "upload": False  # 이미 업로드된 이미지
        }
    
    # 2. 긍정 프롬프트 노드 업데이트 (노드 6)
    if "6" in updated_workflow:
        updated_workflow["6"]["inputs"] = {
            "text": params.get("positive_prompt", "best quality, beautiful woman, light and shadow, library"),
            "clip": ["4", 1]  # CLIP from CheckpointLoaderSimple
        }
    
    # 3. 부정 프롬프트 노드 업데이트 (노드 7)
    if "7" in updated_workflow:
        updated_workflow["7"]["inputs"] = {
            "text": params.get("negative_prompt", "lowres, bad anatomy, bad hands, cropped, worst quality, nsfw"),
            "clip": ["4", 1]  # CLIP from CheckpointLoaderSimple
        }
    
    # 4. VAELoader 노드 설정 (노드 22)
    if "22" in updated_workflow:
        updated_workflow["22"]["inputs"] = {
            "vae_name": "vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors"
        }
    
    # 5. CheckpointLoaderSimple 노드 설정 (노드 4)
    if "4" in updated_workflow:
        updated_workflow["4"]["inputs"] = {
            "ckpt_name": "realisticVisionV51_v51VAE.safetensors"
        }
    
    # 6. BRIAAI Matting 노드 설정 (노드 16)
    if "16" in updated_workflow:
        updated_workflow["16"]["inputs"] = {
            "video_frames": ["15", 0],  # Input from LoadImage
            "version": "v1.4",
            "fp16": True,
            "bg_color": "#7F7F7F",
            "batch_size": 3
        }
    
    # 7. ConstrainImage 노드 설정 (노드 23)
    if "23" in updated_workflow:
        updated_workflow["23"]["inputs"] = {
            "images": ["16", 0],  # Input from BRIAAI Matting
            "max_width": 1024,
            "max_height": 1024,
            "min_width": 0,
            "min_height": 0,
            "crop_if_required": "no"
        }
    
    # 8. GetImageSize+ 노드 설정 (노드 27)
    if "27" in updated_workflow:
        updated_workflow["27"]["inputs"] = {
            "image": ["23", 0]  # Input from ConstrainImage
        }
    
    # 9. ICLightConditioning 노드 설정 (노드 19)
    if "19" in updated_workflow:
        updated_workflow["19"]["inputs"] = {
            "positive": ["6", 0],  # Input from CLIPTextEncode (Positive)
            "negative": ["7", 0],  # Input from CLIPTextEncode (Negative)
            "vae": ["22", 0],  # Input from VAELoader
            "foreground": ["20", 0],  # Input from VAEEncode
            "multiplier": 0.18215
        }
    
    # 10. VAEEncode 노드 설정 (노드 20)
    if "20" in updated_workflow:
        updated_workflow["20"]["inputs"] = {
            "pixels": ["23", 0],  # Input from ConstrainImage
            "vae": ["22", 0]  # Input from VAELoader
        }
    
    # 11. LoadAndApplyICLightUnet 노드 설정 (노드 21)
    if "21" in updated_workflow:
        updated_workflow["21"]["inputs"] = {
            "model": ["4", 0],  # Input from CheckpointLoaderSimple
            "model_path": "IC-Light\\iclight_sd15_fc.safetensors"
        }
    
    # 12. LightSource 노드 설정 (노드 25)
    if "25" in updated_workflow:
        updated_workflow["25"]["inputs"] = {
            "width": ["27", 0],  # Input from GetImageSize+
            "height": ["27", 1],  # Input from GetImageSize+
            "light_position": params.get("light_position", "Top Left Light"),
            "multiplier": 1,
            "start_color": "#FFFFFF",
            "end_color": "#000000",
            "intensity": 1.0
        }
    
    # 13. VAEEncode (second one) 노드 설정 (노드 28)
    if "28" in updated_workflow:
        updated_workflow["28"]["inputs"] = {
            "pixels": ["25", 0],  # Input from LightSource
            "vae": ["4", 2]  # Input from CheckpointLoaderSimple
        }
    
    # 14. KSampler 노드 설정 (노드 3)
    if "3" in updated_workflow:
        # 시드값 가져오기 (파라미터에서 또는 랜덤 생성)
        seed = params.get("seed", int(uuid.uuid4().int % 10000000))
        updated_workflow["3"]["inputs"] = {
            "model": ["21", 0],  # Input from LoadAndApplyICLightUnet
            "positive": ["19", 0],  # Input from ICLightConditioning
            "negative": ["19", 1],  # Input from ICLightConditioning
            "latent_image": ["28", 0],  # Input from VAEEncode
            "seed": seed,
            "steps": 10,
            "cfg": 0.9,
            "sampler_name": "euler_ancestral",  # 유효한 샘플러
            "scheduler": "normal",  # 유효한 스케줄러
            "denoise": 0.9
        }
    
    # 15. VAEDecode 노드 설정 (노드 8)
    if "8" in updated_workflow:
        updated_workflow["8"]["inputs"] = {
            "samples": ["3", 0],  # Input from KSampler
            "vae": ["22", 0]  # Input from VAELoader
        }
    
    # 16. SaveImage 노드 설정 (노드 9)
    if "9" in updated_workflow:
        updated_workflow["9"]["inputs"] = {
            "images": ["8", 0],  # Input from VAEDecode
            "filename_prefix": "Result"
        }
    
    return updated_workflow

async def process_background_creation(
    image_data: bytes,
    params: Dict[str, Any],
    workflow_path: str = BACKGROUND_WORKFLOW_PATH
) -> Optional[str]:
    """
    새로운 배경 생성 처리를 수행합니다.
    
    Args:
        image_data (bytes): 입력 이미지 데이터
        params (Dict[str, Any]): 파라미터
        workflow_path (str): 워크플로우 파일 경로
        
    Returns:
        Optional[str]: 결과 이미지 파일명. 실패 시 None.
        
    Raises:
        WorkflowError: 워크플로우 처리 실패 시
        ImageProcessingError: 이미지 처리 실패 시
        APIError: API 통신 실패 시
    """
    logger.info("배경 생성 처리 시작")
    
    try:
        # 워크플로우 로드
        logger.info(f"워크플로우 파일 로드: {workflow_path}")
        workflow = await load_workflow(workflow_path)
        
        # 이미지 파일명 생성
        image_filename = f"input_{uuid.uuid4()}.png"
        logger.info(f"생성된 이미지 파일명: {image_filename}")
        
        # ComfyUI에 이미지 업로드
        logger.info("이미지 업로드 중...")
        await upload_image(image_data, image_filename)
        logger.info("이미지 업로드 완료")
        
        # 워크플로우 파라미터 업데이트
        updated_workflow = await update_workflow_parameters(workflow, image_filename, params)
        
        # 클라이언트 ID 생성
        client_id = str(uuid.uuid4())
        
        # 워크플로우 내용 디버깅
        logger.debug("워크플로우 내용:")
        logger.debug(json.dumps(updated_workflow, indent=2))
        
        # 프롬프트 큐에 추가
        logger.info("워크플로우 전송 중...")
        prompt_id = await queue_prompt(updated_workflow, client_id)
        logger.info(f"프롬프트 ID: {prompt_id}")
        
        # 결과 대기
        logger.info("결과 대기 중...")
        result = await check_progress(prompt_id)
        
        # 결과 검사
        if not result:
            logger.error("오류: 결과가 없음")
            return None
            
        logger.info("결과 수신됨, 출력 찾는 중...")
        
        # 결과 이미지 파일명 추출
        output_images = []
        logger.debug("모든 출력 노드 검사:")
        for node_id, node_output in result.get("outputs", {}).items():
            logger.debug(f"노드 ID: {node_id}, 출력: {node_output}")
            if "images" in node_output:
                for img in node_output["images"]:
                    output_images.append(img["filename"])
                    logger.info(f"결과 이미지 발견: {img['filename']}")

        # 출력 이미지 목록 확인
        if output_images:
            logger.info(f"모든 결과 이미지: {output_images}")
            # SaveImage 노드의 출력(일반적으로 ComfyUI_로 시작)을 우선 사용
            save_image = next((img for img in output_images if img.startswith("ComfyUI_")), None)
            if save_image:
                logger.info(f"SaveImage 출력 사용: {save_image}")
                return save_image
            else:
                logger.info(f"첫 번째 이미지 사용: {output_images[0]}")
                return output_images[0]
        else:
            logger.error("오류: 결과 이미지 없음")
            return None
    
    except WorkflowError as e:
        logger.error(f"워크플로우 처리 오류: {str(e)}")
        raise
    except ImageProcessingError as e:
        logger.error(f"이미지 처리 오류: {str(e)}")
        raise
    except APIError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"배경 생성 처리 중 예상치 못한 오류: {str(e)}")
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"상세 오류 정보: {error_detail}")
        raise WorkflowError(f"배경 생성 처리 중 오류 발생: {str(e)}\n{error_detail}")

async def get_output_images(prompt_id: str) -> List[str]:
    """
    프롬프트 ID로부터 결과 이미지 파일명 목록을 가져옵니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        
    Returns:
        List[str]: 결과 이미지 파일명 목록
        
    Raises:
        APIError: API 통신 실패 시
        WorkflowError: 워크플로우 처리 실패 시
    """
    logger.info(f"결과 이미지 목록 가져오기 시작 (prompt_id: {prompt_id})")
    
    try:
        # 결과 가져오기
        result = await check_progress(prompt_id)
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                output_images.extend([img["filename"] for img in node_output["images"]])
                logger.debug(f"노드 {node_id}에서 {len(node_output['images'])}개의 이미지 발견")
        
        if output_images:
            logger.info(f"총 {len(output_images)}개의 결과 이미지 발견")
            return output_images
        else:
            logger.warning("결과 이미지가 없음")
            return []
    
    except APIError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise
    except WorkflowError as e:
        logger.error(f"워크플로우 처리 오류: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"결과 이미지 목록 가져오기 중 예상치 못한 오류: {str(e)}")
        raise WorkflowError(f"결과 이미지 목록 가져오기 중 오류 발생: {str(e)}")