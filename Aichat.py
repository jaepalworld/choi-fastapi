import json
import logging
import asyncio
import httpx
import firebase_admin
from firebase_admin import firestore
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API 엔드포인트
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

# 시스템 프롬프트 
SYSTEM_PROMPT = """당신은 친절하고 전문적인 HairAI 고객 상담 AI입니다. 
당신의 역할은 HairAI 서비스에 관한 사용자의 질문에 정확하고 간결하게 답변하는 것입니다.

HairAI 서비스 정보:
- HairAI는 AI를 활용한 헤어스타일 추천 및 가상 체험 서비스입니다.
- 주요 기능: 얼굴 분석, 헤어스타일 추천, 헤어스타일 시뮬레이션, 인물 사진 배경 제거, 얼굴 변환 기능 제공
- 사용자는 자신의 사진을 업로드하여 다양한 헤어스타일을 가상으로 체험해볼 수 있습니다.
- 서비스는 웹과 모바일 앱 모두 지원합니다.
- 무료 회원가입 후 10번의 무료 사용권을 제공하며, 이후에는 유료 구독이 필요합니다.

응답 지침:
1. 항상 공손하고 친절한 어조를 유지하세요.
2. 사용자의 질문을 정확히 이해하고 직접적인 답변을 제공하세요.
3. 모르는 질문에는 솔직히 모른다고 답변하고, 추가 정보를 요청하세요.
4. 응답은 간결하게 유지하되, 필요한 정보는 모두 포함하세요.
5. 기술적인 질문에는 전문적이지만 이해하기 쉬운 설명을 제공하세요.

제약사항:
- HairAI와 관련 없는 주제는 제한적으로 대응하고, 서비스와 관련된 문의로 안내해주세요.
- 부적절한 요청이나 개인정보를 요구하는 질문은 정중히 거절하세요.
- 상담 AI로서 실시간 사용자 데이터나 계정 정보는 접근할 수 없음을 알려주세요.
"""

# 연결된 클라이언트를 관리하는 ConnectionManager 클래스
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}  # 사용자별 세션 정보 저장

    async def connect(self, websocket: WebSocket, client_id: str, already_accepted: bool = False):
        # already_accepted 매개변수 추가: 이미 accept된 경우에는 다시 accept하지 않음
        if not already_accepted:
            await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_sessions[client_id] = {"history": []}
        logger.info(f"클라이언트 연결됨: {client_id}")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self.user_sessions.pop(client_id, None)
        logger.info(f"클라이언트 연결 종료: {client_id}")

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    def get_session(self, client_id: str) -> Dict[str, Any]:
        if client_id not in self.user_sessions:
            self.user_sessions[client_id] = {"history": []}
        return self.user_sessions[client_id]


# ConnectionManager 인스턴스 생성
manager = ConnectionManager()


# Ollama API를 호출하여 AI 응답을 생성하는 함수
async def generate_ai_response(message: str, client_id: str, user_id: str, chat_id: str, db):
    session = manager.get_session(client_id)
    history = session.get("history", [])
    
    # 프롬프트 구성
    prompt = format_prompt(message, history)
    
    # AI가 입력 중임을 알림
    typing_notification = json.dumps({
        "type": "ai_typing",
        "userId": user_id,
        "timestamp": datetime.now().isoformat()
    })
    await manager.send_message(client_id, typing_notification)
    
    try:
        # Ollama API 호출 (스트리밍 모드)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True,
                },
                timeout=60.0,
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 텍스트 초기화
            full_response = ""
            
            # 스트리밍 응답 처리
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 응답 토큰 처리
                    if "response" in data:
                        token = data["response"]
                        full_response += token
                        
                        # 클라이언트에 토큰 전송
                        ai_message = {
                            "text": full_response,
                            "userId": user_id,
                            "userName": "AI 상담원",
                            "type": "ai",
                            "timestamp": datetime.now().isoformat(),
                            "tabType": "ai",
                            "chatId": chat_id
                        }
                        await manager.send_message(client_id, json.dumps(ai_message))
                    
                    # 응답이 완료되면 종료
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"JSON 파싱 오류: {line}")
                    continue
            
            # 응답 저장
            if full_response:
                # 대화 기록 업데이트
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": full_response})
                session["history"] = history[-10:]  # 최근 5개 대화만 유지
                
                # Firebase에 저장
                await save_to_firebase(user_id, "ai", full_response, chat_id, db)
                
                logger.info(f"AI 응답 생성 완료 (길이: {len(full_response)})")
                return full_response
                
    except Exception as e:
        error_msg = f"AI 응답 생성 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        
        # 오류 메시지 전송
        error_response = {
            "text": "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "userId": user_id,
            "userName": "AI 상담원",
            "type": "ai",
            "timestamp": datetime.now().isoformat(),
            "tabType": "ai",
            "chatId": chat_id
        }
        await manager.send_message(client_id, json.dumps(error_response))
        
        # Firebase에 오류 저장
        await save_to_firebase(
            user_id, 
            "ai", 
            "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", 
            chat_id,
            db
        )
        
        return None


# 프롬프트 포맷팅 함수
def format_prompt(message: str, history: List[Dict[str, str]]) -> str:
    # 시스템 프롬프트와 대화 기록 포맷팅
    formatted_prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    
    # 대화 기록 추가
    for entry in history:
        if entry["role"] == "user":
            formatted_prompt += f"{entry['content']} [/INST] "
        elif entry["role"] == "assistant":
            formatted_prompt += f"{entry['content']} </s><s>[INST] "
    
    # 현재 사용자 메시지 추가
    formatted_prompt += f"{message} [/INST] "
    
    return formatted_prompt


# Firebase에 메시지 저장 함수
async def save_to_firebase(user_id: str, message_type: str, message: str, chat_id: str, db):
    try:
        # 저장할 데이터 준비
        data = {
            "userId": user_id,
            "userName": "AI 상담원" if message_type == "ai" else "사용자",
            "text": message,
            "type": message_type,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "tabType": "ai",
            "chatId": chat_id
        }
        
        # aiChats 컬렉션에 저장
        await asyncio.to_thread(
            db.collection("aiChats").add,
            data
        )
        
        logger.info(f"Firebase에 메시지 저장 완료 (사용자: {user_id}, 타입: {message_type})")
    except Exception as e:
        logger.error(f"Firebase 저장 중 오류 발생: {str(e)}")


# WebSocket 핸들러
async def websocket_endpoint(websocket: WebSocket, db, already_accepted=False):
    client_id = None
    user_id = None
    chat_id = None
    
    try:
        logger.info("WebSocket 핸들러 시작")
        
        # 첫 메시지로 사용자 정보 수신
        data = await websocket.receive_text()
        logger.info(f"첫 메시지 수신: {data[:100]}...")  # 로그에 데이터 일부만 표시
        
        try:
            user_info = json.loads(data)
            
            # 사용자 정보 추출
            client_id = user_info.get("userId", str(id(websocket)))
            user_id = user_info.get("userId")
            
            logger.info(f"사용자 연결: client_id={client_id}, user_id={user_id}")
            
            # 연결 관리자에 등록 - already_accepted 매개변수 전달
            await manager.connect(websocket, client_id, already_accepted=already_accepted)
            
            # 필요시 추가 연결 정보 전송
            success_msg = {
                "type": "user_registered",
                "message": "사용자 등록 완료"
            }
            await websocket.send_text(json.dumps(success_msg))
            
        except json.JSONDecodeError:
            logger.error(f"첫 메시지 파싱 오류: {data}")
            error_msg = {
                "type": "error",
                "message": "잘못된 메시지 형식입니다. JSON 형식이 필요합니다."
            }
            await websocket.send_text(json.dumps(error_msg))
            return
        
        # 이후 메시지 처리
        while True:
            data = await websocket.receive_text()
            logger.info(f"메시지 수신: {data[:100]}...")  # 로그에 데이터 일부만 표시
            
            try:
                message_data = json.loads(data)
                
                # 메시지 유형 체크
                if message_data.get("type") == "user":
                    user_message = message_data.get("text", "")
                    chat_id = message_data.get("chatId", f"chat_{user_id}_{datetime.now().timestamp()}")
                    
                    logger.info(f"사용자 메시지 수신: {user_message[:50]}...")
                    
                    # AI 응답 생성
                    asyncio.create_task(generate_ai_response(
                        user_message, client_id, user_id, chat_id, db
                    ))
            
            except json.JSONDecodeError:
                logger.error(f"메시지 파싱 오류: {data}")
                error_msg = {
                    "type": "error",
                    "message": "잘못된 메시지 형식입니다. JSON 형식이 필요합니다."
                }
                await websocket.send_text(json.dumps(error_msg))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket 연결 종료: {client_id}")
        if client_id:
            manager.disconnect(client_id)
    
    except Exception as e:
        logger.error(f"WebSocket 처리 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        if client_id:
            manager.disconnect(client_id)