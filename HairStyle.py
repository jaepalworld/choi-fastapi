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
HAIRSTYLE_WORKFLOW_PATH = os.getenv('HAIRSTYLE_WORKFLOW_PATH', "D:/choi-fastapi/workflow/HAIReditFinish.json")
COMFYUI_OUTPUT_DIR = os.getenv('COMFYUI_OUTPUT_DIR', "output")

# 노드 ID 상수 정의 (HAIReditFinish.json 워크플로우 파일에서 식별된 주요 노드)
NODE_IDS = {
    "FACE_ANALYSIS_MODELS": "12",      # FaceAnalysisModels 노드 ID
    "REACTOR_FACE_SWAP": "8",          # ReActorFaceSwap 노드 ID
    "LOAD_IMAGE_USER": "53",           # 사용자 이미지 로드 노드 ID (userimage)
    "LOAD_IMAGES_DIRECTORY": "57",     # 롤모델 이미지 배치 로드 노드 ID
    "SAVE_IMAGE": "74",                # SaveImage 노드 ID
    "FACE_SEGMENTATION_EYES": "11",    # 눈 세그멘테이션 노드 ID
    "FACE_SEGMENTATION_NOSE": "17",    # 코 세그멘테이션 노드 ID
    "FACE_SEGMENTATION_MOUTH": "15",   # 입 세그멘테이션 노드 ID
    "FACE_SEGMENTATION_FACE": "34",    # 얼굴 세그멘테이션 노드 ID
    "FREQUENCY_SEPARATION": "47",      # 주파수 분리 노드 ID
    "REPEAT_IMAGE_BATCH": "67",        # 이미지 배치 반복 노드 ID
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

async def check_progress(prompt_id: str, timeout_minutes: int = 30) -> Dict[str, Any]:
    """
    프롬프트 처리 상태를 확인하고 결과를 반환합니다.
    
    Args:
        prompt_id (str): 프롬프트 ID
        timeout_minutes (int, optional): 타임아웃 시간(분). 기본값은 30분.
        
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
            # 타임아웃 설정 (분 -> 초)
            max_attempts = timeout_minutes * 60 // 5  # 5초마다 체크, timeout_minutes 분 동안
            for attempt in range(max_attempts):
                try:
                    async with session.get(history_url) as response:
                        if response.status == 404:
                            # 이 시점에서는 아직 처리 중일 수 있으므로 오류로 처리하지 않음
                            logger.debug(f"프롬프트 {prompt_id}의 처리 결과가 아직 없습니다. 대기 중...")
                        elif response.status != 200:
                            error_text = await response.text()
                            logger.error(f"진행 상태 확인 실패 (상태 코드: {response.status}): {error_text}")
                            raise APIError(f"진행 상태 확인 실패: {error_text}")
                        else:
                            history = await response.json()
                            if prompt_id in history:
                                logger.info(f"프롬프트 처리 완료 (prompt_id: {prompt_id})")
                                return history[prompt_id]
                except aiohttp.ClientError as e:
                    # 일시적인 네트워크 오류는 무시하고 재시도
                    logger.warning(f"네트워크 오류 발생, 재시도 중: {str(e)}")
                
                # 진행 상태 로그 (60초마다)
                if attempt % 12 == 0 and attempt > 0:
                    minutes_elapsed = attempt * 5 // 60
                    logger.info(f"헤어스타일 생성 진행 중... ({minutes_elapsed}분 경과)")
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5)  # 5초 대기 후 재시도
            
            logger.error(f"프롬프트 처리 시간 초과 (prompt_id: {prompt_id}, 타임아웃: {timeout_minutes}분)")
            raise WorkflowError(f"프롬프트 {prompt_id}의 처리 결과를 가져오지 못했습니다 (시간 초과: {timeout_minutes}분)")
            
    except aiohttp.ClientError as e:
        logger.error(f"API 통신 오류: {str(e)}")
        raise APIError(f"API 통신 오류: {str(e)}")
    except Exception as e:
        if isinstance(e, WorkflowError) or isinstance(e, APIError):
            raise
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
                "FaceAnalysisModels": ["model", "provider"],
                "ReActorFaceSwap": ["enabled", "swap_model", "facedetection", "face_restore_model", 
                                   "face_restore_visibility", "codeformer_weight", "detect_gender_source", 
                                   "detect_gender_input", "input_faces_index", "source_faces_index"],
                "LoadImage": ["image", "upload", "use_metadata"],
                "LoadImagesFromDirectory": ["directory", "image_load_cap", "start_index"],
                "SaveImage": ["filename_prefix"],
                "FaceSegmentation": ["part", "index", "blur", "dilation"],
                "FrequencySeparation": ["radius"],
                "GrowMaskWithBlur": ["expand", "contract", "normalize", "invert", "blur_radius", "iterations", "sigmoid", "discontiguous_areas"],
                "RepeatImageBatch": ["amount"]
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
                
                # 노드 제목이 있으면 추가
                if "title" in node:
                    api_node["title"] = node["title"]
                
                # 위젯 값이 있으면 매핑에 따라 추가
                if "widgets_values" in node:
                    # 해당 노드 타입의 위젯 매핑 가져오기
                    widget_names = widget_mappings.get(node_type, [])
                    for i, value in enumerate(node["widgets_values"]):
                        if i < len(widget_names):
                            # 매핑된 이름으로 위젯 추가
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
    헤어스타일 워크플로우 파라미터를 동적으로 업데이트합니다.
    
    Args:
        workflow (Dict[str, Any]): 원본 워크플로우 데이터
        image_filename (str): 업로드된 이미지 파일명
        params (Dict[str, Any]): 업데이트할 파라미터
            
    Returns:
        Dict[str, Any]: 업데이트된 워크플로우 데이터
    """
    # 워크플로우의 깊은 복사본 생성
    updated_workflow = json.loads(json.dumps(workflow))
    
    # 기본 파라미터 설정
    hair_dir = params.get("hair_dir", "D:\\StabilityMatrix-win-x64\\HairImage")
    hair_count = params.get("hair_count", 4)
    face_restoration = params.get("face_restoration", 0.5)
    batch_size = params.get("batch_size", 4)
    filename_prefix = params.get("filename_prefix", "HairConsulting")
    
    # 1. FaceAnalysisModels 노드 업데이트 (노드 12)
    if NODE_IDS["FACE_ANALYSIS_MODELS"] in updated_workflow:
        updated_workflow[NODE_IDS["FACE_ANALYSIS_MODELS"]]["inputs"] = {
            "model": "insightface",
            "provider": "CUDA"
        }
        
    # 2. 사용자 이미지 로드 노드 업데이트 (노드 53)
    if NODE_IDS["LOAD_IMAGE_USER"] in updated_workflow:
        updated_workflow[NODE_IDS["LOAD_IMAGE_USER"]]["inputs"] = {
            "image": image_filename,
            "upload": True
        }
        
    # 3. 롤모델 이미지 디렉토리 로드 노드 업데이트 (노드 57)
    if NODE_IDS["LOAD_IMAGES_DIRECTORY"] in updated_workflow:
        updated_workflow[NODE_IDS["LOAD_IMAGES_DIRECTORY"]]["inputs"] = {
            "directory": hair_dir,
            "image_load_cap": hair_count,
            "start_index": 0
        }
        
    # 4. RepeatImageBatch 노드 업데이트 (노드 67)
    if NODE_IDS["REPEAT_IMAGE_BATCH"] in updated_workflow:
        updated_workflow[NODE_IDS["REPEAT_IMAGE_BATCH"]]["inputs"] = {
            "amount": batch_size,
            "image": [NODE_IDS["LOAD_IMAGE_USER"], 0]  # 링크 88: 노드 53 -> 노드 67
        }
        
    # 5. ReActorFaceSwap 노드 업데이트 (노드 8)
    if NODE_IDS["REACTOR_FACE_SWAP"] in updated_workflow:
        updated_workflow[NODE_IDS["REACTOR_FACE_SWAP"]]["inputs"] = {
            "input_image": [NODE_IDS["LOAD_IMAGES_DIRECTORY"], 0],  # 링크 90: 노드 57 -> 노드 8
            "source_image": [NODE_IDS["REPEAT_IMAGE_BATCH"], 0],    # 링크 89: 노드 67 -> 노드 8
            "enabled": True,
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": "GFPGANv1.4.pth",
            "face_restore_visibility": 1.0,
            "codeformer_weight": face_restoration,
            "detect_gender_source": "no",
            "detect_gender_input": "no",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1
        }
        
    # 6. FaceSegmentation 노드 - 눈 (노드 11)
    if NODE_IDS["FACE_SEGMENTATION_EYES"] in updated_workflow:
        updated_workflow[NODE_IDS["FACE_SEGMENTATION_EYES"]]["inputs"] = {
            "analysis_models": [NODE_IDS["FACE_ANALYSIS_MODELS"], 0],  # 링크 14: 노드 12 -> 노드 11
            "image": [NODE_IDS["REACTOR_FACE_SWAP"], 0],                # 링크 15: 노드 8 -> 노드 11
            "area": "eyes",
            "grow": 0,
            "grow_tapered": False,
            "blur": 13,
            "index": 0
        }
        
    # 7. FaceSegmentation 노드 - 코 (노드 17)
    if NODE_IDS["FACE_SEGMENTATION_NOSE"] in updated_workflow:
        updated_workflow[NODE_IDS["FACE_SEGMENTATION_NOSE"]]["inputs"] = {
            "analysis_models": [NODE_IDS["FACE_ANALYSIS_MODELS"], 0],  # 링크 92: 노드 12 -> 노드 17
            "image": [NODE_IDS["REACTOR_FACE_SWAP"], 0],                # 링크 18: 노드 8 -> 노드 17
            "area": "nose",
            "grow": 0,
            "grow_tapered": False,
            "blur": 13,
            "index": 0
        }
        
    # 8. FaceSegmentation 노드 - 입 (노드 15)
    if NODE_IDS["FACE_SEGMENTATION_MOUTH"] in updated_workflow:
        updated_workflow[NODE_IDS["FACE_SEGMENTATION_MOUTH"]]["inputs"] = {
            "analysis_models": [NODE_IDS["FACE_ANALYSIS_MODELS"], 0],  # 링크 41: 노드 12 -> 노드 15
            "image": [NODE_IDS["REACTOR_FACE_SWAP"], 0],                # 링크 19: 노드 8 -> 노드 15
            "area": "mouth",
            "grow": 0,
            "grow_tapered": False,
            "blur": 13,
            "index": 0
        }
        
    # 9. FaceSegmentation 노드 - 얼굴 (노드 34)
    if NODE_IDS["FACE_SEGMENTATION_FACE"] in updated_workflow:
        updated_workflow[NODE_IDS["FACE_SEGMENTATION_FACE"]]["inputs"] = {
            "analysis_models": [NODE_IDS["FACE_ANALYSIS_MODELS"], 0],  # 링크 53: 노드 12 -> 노드 34
            "image": [NODE_IDS["REACTOR_FACE_SWAP"], 0],                # 링크 47: 노드 8 -> 노드 34
            "area": "face+forehead (if available)",
            "grow": 0,
            "grow_tapered": False,
            "blur": 13,
            "index": 0
        }
        
    # 10. GrowMaskWithBlur 노드 - 눈 (노드 25)
    if "25" in updated_workflow:
        updated_workflow["25"]["inputs"] = {
            "mask": [NODE_IDS["FACE_SEGMENTATION_EYES"], 0],  # 링크 29: 노드 11 -> 노드 25
            "expand": 80,
            "contract": 0,
            "normalize": True,
            "invert": False,
            "blur_radius": 10.0,
            "iterations": 1,
            "sigmoid": 0.83,
            "discontiguous_areas": False
        }
        
    # 11. GrowMaskWithBlur 노드 - 코 (노드 27)
    if "27" in updated_workflow:
        updated_workflow["27"]["inputs"] = {
            "mask": [NODE_IDS["FACE_SEGMENTATION_NOSE"], 0],  # 링크 33: 노드 17 -> 노드 27
            "expand": 80,
            "contract": 0,
            "normalize": True,
            "invert": False,
            "blur_radius": 10.0,
            "iterations": 1,
            "sigmoid": 0.83,
            "discontiguous_areas": False
        }
        
    # 12. GrowMaskWithBlur 노드 - 입 (노드 29)
    if "29" in updated_workflow:
        updated_workflow["29"]["inputs"] = {
            "mask": [NODE_IDS["FACE_SEGMENTATION_MOUTH"], 0],  # 링크 39: 노드 15 -> 노드 29
            "expand": 80,
            "contract": 0,
            "normalize": True,
            "invert": False,
            "blur_radius": 10.0,
            "iterations": 1,
            "sigmoid": 0.83,
            "discontiguous_areas": False
        }
        
    # 13. GrowMaskWithBlur 노드 - 얼굴 (노드 32)
    if "32" in updated_workflow:
        updated_workflow["32"]["inputs"] = {
            "mask": [NODE_IDS["FACE_SEGMENTATION_FACE"], 0],  # 링크 42: 노드 34 -> 노드 32
            "expand": 80,
            "contract": 0,
            "normalize": True,
            "invert": False,
            "blur_radius": 10.0,
            "iterations": 1,
            "sigmoid": 0.83,
            "discontiguous_areas": False
        }
        
    # 14. ImageCompositeMasked 노드 - 눈 (노드 18)
    if "18" in updated_workflow:
        updated_workflow["18"]["inputs"] = {
            "destination": [NODE_IDS["REACTOR_FACE_SWAP"], 0],  # 링크 36: 노드 8 -> 노드 18
            "source": [NODE_IDS["REACTOR_FACE_SWAP"], 0],       # 링크 21: 노드 8 -> 노드 18
            "mask": ["25", 0],                                   # 링크 28: 노드 25 -> 노드 18
            "x": 0,
            "y": 0,
            "resize_source": False
        }
        
    # 15. ImageCompositeMasked 노드 - 코 (노드 26)
    if "26" in updated_workflow:
        updated_workflow["26"]["inputs"] = {
            "destination": ["18", 0],                        # 링크 66: 노드 18 -> 노드 26
            "source": [NODE_IDS["REACTOR_FACE_SWAP"], 0],    # 링크 34: 노드 8 -> 노드 26
            "mask": ["27", 0],                                # 링크 30: 노드 27 -> 노드 26
            "x": 0,
            "y": 0,
            "resize_source": False
        }
        
    # 16. ImageCompositeMasked 노드 - 입 (노드 28)
    if "28" in updated_workflow:
        updated_workflow["28"]["inputs"] = {
            "destination": ["18", 0],                        # 링크 67: 노드 18 -> 노드 28
            "source": [NODE_IDS["REACTOR_FACE_SWAP"], 0],    # 링크 35: 노드 8 -> 노드 28
            "mask": ["29", 0],                                # 링크 31: 노드 29 -> 노드 28
            "x": 0,
            "y": 0,
            "resize_source": False
        }
        
    # 17. ImageCompositeMasked 노드 - 얼굴 (노드 33)
    if "33" in updated_workflow:
        updated_workflow["33"]["inputs"] = {
            "destination": ["18", 0],                        # 링크 68: 노드 18 -> 노드 33
            "source": [NODE_IDS["REACTOR_FACE_SWAP"], 0],    # 링크 47: 노드 8 -> 노드 33
            "mask": ["32", 0],                                # 링크 43: 노드 32 -> 노드 33
            "x": 0,
            "y": 0,
            "resize_source": False
        }
        
    # 18. FrequencySeparation 노드 (노드 39)
    if "39" in updated_workflow:
        updated_workflow["39"]["inputs"] = {
            "image": ["28", 0],       # 링크 63: 노드 28 -> 노드 39
            "blur_radius": 3          # 스크린샷에서 확인한 값
        }
        
    # 19. FrequencySeparation 노드 (노드 42)
    if "42" in updated_workflow:
        updated_workflow["42"]["inputs"] = {
            "image": ["28", 0],       # 링크 64: 노드 28 -> 노드 42
            "blur_radius": 10         # 스크린샷에서 확인한 값
        }
        
    # 20. FrequencyCombination 노드 (노드 45)
    if "45" in updated_workflow:
        updated_workflow["45"]["inputs"] = {
            "high_freq": ["42", 0],   # 링크 58: 노드 42 -> 노드 45
            "low_freq": ["39", 1]     # 링크 57: 노드 39 -> 노드 45
        }
        
    # 21. FrequencySeparation 노드 (노드 47) - 주파수 분리
    if NODE_IDS["FREQUENCY_SEPARATION"] in updated_workflow:
        updated_workflow[NODE_IDS["FREQUENCY_SEPARATION"]]["inputs"] = {
            "image": [NODE_IDS["REACTOR_FACE_SWAP"], 0],  # 링크 69: 노드 8 -> 노드 47
            "blur_radius": 3                              # 스크린샷에서 확인한 값 (parameter 이름 주의!)
        }
        
    # 22. SaveImage 노드 (노드 74)
    if NODE_IDS["SAVE_IMAGE"] in updated_workflow:
        updated_workflow[NODE_IDS["SAVE_IMAGE"]]["inputs"] = {
            "filename_prefix": filename_prefix,
            "images": [NODE_IDS["FREQUENCY_SEPARATION"], 1]  # 링크 96: 노드 47(low_freq 출력) -> 노드 74
        }
    
    return updated_workflow

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

async def process_hairstyle_generation(
    image_data: bytes,
    params: Dict[str, Any],
    workflow_path: str = HAIRSTYLE_WORKFLOW_PATH,
    timeout_minutes: int = 30
) -> List[Dict[str, Any]]:
    """
    헤어스타일 생성 처리를 수행합니다.
    
    Args:
        image_data (bytes): 입력 이미지 데이터
        params (Dict[str, Any]): 파라미터
        workflow_path (str): 워크플로우 파일 경로
        timeout_minutes (int): 최대 대기 시간(분)
        
    Returns:
        List[Dict[str, Any]]: 결과 이미지 정보 목록 (각 항목은 filename, url, firebase_url 포함)
        
    Raises:
        WorkflowError: 워크플로우 처리 실패 시
        ImageProcessingError: 이미지 처리 실패 시
        APIError: API 통신 실패 시
    """
    logger.info("헤어스타일 생성 처리 시작")
    
    try:
        # 워크플로우 로드
        logger.info(f"워크플로우 파일 로드: {workflow_path}")
        workflow = await load_workflow(workflow_path)
        
        # 이미지 파일명 생성 및 업로드
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
        
        # 프롬프트 큐에 추가
        logger.info("워크플로우 전송 중...")
        prompt_id = await queue_prompt(updated_workflow, client_id)
        logger.info(f"프롬프트 ID: {prompt_id}")
        
        # 결과 대기 (타임아웃 설정)
        logger.info(f"결과 대기 중... (최대 {timeout_minutes}분)")
        result = await check_progress(prompt_id, timeout_minutes)
        
        # 결과 검사
        if not result:
            logger.error("오류: 결과가 없음")
            return []
            
        logger.info("결과 수신됨, 출력 이미지 찾는 중...")
        
        # 결과 이미지 파일명 추출
        output_images = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    filename = img.get("filename")
                    if filename:
                        output_images.append(filename)
                        logger.info(f"결과 이미지 발견: {filename}")
        
        # Firebase에 결과 업로드 및 결과 정보 구성
        result_info = []
        for filename in output_images:
            # 이미지 URL 구성
            image_url = f"{COMFYUI_API_URL}/view?filename={filename}&subfolder=&type=output"
            
            # 파일 경로 구성
            file_path = os.path.join(COMFYUI_OUTPUT_DIR, filename)
            
            # Firebase 업로드 시도
            firebase_url = None
            try:
                if os.path.exists(file_path):
                    # Firebase Storage에 업로드 함수 호출 (외부에서 정의된 함수 사용 가정)
                    from main import upload_image_to_firebase
                    firebase_path = f"hairstyles/{uuid.uuid4().hex}_{filename}"
                    firebase_url = upload_image_to_firebase(file_path, firebase_path)
                    logger.info(f"Firebase 업로드 성공: {firebase_url}")
            except Exception as e:
                logger.warning(f"Firebase 업로드 실패: {str(e)}")
            
            # 결과 정보 추가
            result_info.append({
                "filename": filename,
                "url": image_url,
                "firebase_url": firebase_url
            })
        
        if result_info:
            logger.info(f"헤어스타일 생성 완료: {len(result_info)}개의 결과")
            return result_info
        else:
            logger.error("오류: 결과 이미지 정보 없음")
            return []
        
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
        logger.error(f"헤어스타일 생성 처리 중 예상치 못한 오류: {str(e)}")
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"상세 오류 정보: {error_detail}")
        raise WorkflowError(f"헤어스타일 생성 처리 중 오류 발생: {str(e)}\n{error_detail}")

# 메모리 효율적인 이미지 처리 함수
async def process_image_efficiently(image_data: bytes, max_size: int = 1024) -> bytes:
    """
    메모리 효율적으로 이미지를 처리합니다 (크기 조정 등).
    
    Args:
        image_data (bytes): 원본 이미지 데이터
        max_size (int): 최대 이미지 크기 (픽셀)
        
    Returns:
        bytes: 처리된 이미지 데이터
    """
    try:
        # Pillow를 사용한 이미지 처리
        from io import BytesIO
        from PIL import Image
        
        # 이미지 로드
        with BytesIO(image_data) as buffer:
            img = Image.open(buffer)
            img.load()  # 이미지 데이터 로드
        
        # 이미지 크기 확인 및 조정
        width, height = img.size
        if width > max_size or height > max_size:
            # 비율 유지하며 크기 조정
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # 이미지 리사이징
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 처리된 이미지 저장
            output_buffer = BytesIO()
            if img.mode == 'RGBA':
                # PNG 형식으로 저장 (알파 채널 유지)
                img.save(output_buffer, format="PNG")
            else:
                # JPEG 형식으로 저장 (크기 최적화)
                img.save(output_buffer, format="JPEG", quality=95)
            
            processed_data = output_buffer.getvalue()
            return processed_data
        
        # 크기 조정이 필요 없는 경우 원본 반환
        return image_data
        
    except Exception as e:
        logger.error(f"이미지 효율적 처리 중 오류: {str(e)}")
        # 오류 발생 시 원본 이미지 반환
        return image_data