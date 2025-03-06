from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
from datetime import datetime
import json
import asyncio


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기본 루트 경로 추가 
@app.get("/")
async def root():
    return {"message": "켜졌다리다리다리 백만불짜리 다리"}

# 모델과 토크나이저 로드
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # 관리자 연결을 추적하기 위한 딕셔너리 추가
        self.admin_connections: Dict[str, WebSocket] = {}
        # 사용자 ID를 웹소켓에 매핑하는 딕셔너리 추가
        self.user_connections: Dict[str, WebSocket] = {}
        # chatId와 userId 매핑을 저장하는 딕셔너리 추가
        self.chat_user_map: Dict[str, str] = {}
       
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
       
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # 연결이 끊긴 웹소켓에 해당하는 관리자/사용자 매핑 제거
        admin_to_remove = None
        for admin_id, ws in self.admin_connections.items():
            if ws == websocket:
                admin_to_remove = admin_id
                break
                
        if admin_to_remove:
            del self.admin_connections[admin_to_remove]
            
        user_to_remove = None
        for user_id, ws in self.user_connections.items():
            if ws == websocket:
                user_to_remove = user_id
                break
                
        if user_to_remove:
            del self.user_connections[user_to_remove]
            
    # 관리자 연결 등록 메서드 추가
    def register_admin(self, admin_id: str, websocket: WebSocket):
        self.admin_connections[admin_id] = websocket
        
    # 사용자 연결 등록 메서드 추가
    def register_user(self, user_id: str, websocket: WebSocket):
        self.user_connections[user_id] = websocket
       
    # 메시지 브로드캐스트 함수 수정
    async def broadcast(self, message: dict):
        # 모든 연결에 메시지 전송 (기존 코드)
        for connection in self.active_connections:
            await connection.send_json(message)
            
    # 특정 사용자에게만 메시지 전송하는 함수 추가
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.user_connections:
            await self.user_connections[user_id].send_json(message)
            
    # 모든 관리자에게 메시지 전송하는 함수 추가
    async def send_to_admins(self, message: dict):
        for admin_ws in self.admin_connections.values():
            await admin_ws.send_json(message)
            
    # 개인 채팅 메시지 전달 함수 수정
    async def handle_personal_chat(self, message: dict):
        print("Handling personal chat message:", message)
        print("Current chat_user_map:", self.chat_user_map)
        print("Current user_connections:", list(self.user_connections.keys()))
        
        # 사용자가 보낸 메시지인 경우
        if message['type'] == 'user':
            print("User message received, sending to admins")
            
            # chatId가 있는 경우 chat-user 매핑 저장 (이 부분 중요!)
            if 'chatId' in message and 'userId' in message:
                self.chat_user_map[message['chatId']] = message['userId']
                print(f"Updated chat_user_map: {message['chatId']} -> {message['userId']}")
            
            # 모든 관리자에게 메시지 전달
            await self.send_to_admins(message)
            # 메시지를 보낸 사용자에게도 본인 메시지 확인용으로 전달
            if 'userId' in message and message['userId'] in self.user_connections:
                await self.send_to_user(message['userId'], message)
                
        # 관리자가 보낸 메시지인 경우 ('agent' 또는 'admin' 타입 모두 처리)
        elif message['type'] in ['agent', 'admin']:
            print(f"Admin message received: {message}")
            
            # chatId와 userId 찾기
            target_user_id = None
            
            # 메시지에 userId가 있는지 확인
            if 'userId' in message and message['userId']:
                target_user_id = message['userId']
                print(f"Found userId in message: {target_user_id}")
                
            # 메시지에 chatId가 있고 매핑 테이블에 있는지 확인
            elif 'chatId' in message and message['chatId'] in self.chat_user_map:
                target_user_id = self.chat_user_map[message['chatId']]
                message['userId'] = target_user_id  # 메시지에 userId 필드 추가
                print(f"Found userId from chat_user_map: {target_user_id} for chatId: {message['chatId']}")
            
            # 사용자 찾기 실패
            if not target_user_id:
                print(f"Cannot find target user for message: {message}")
                # 다른 관리자들에게는 메시지 공유
                await self.send_to_admins(message)
                return
                
            # 사용자에게 메시지 전송
            if target_user_id in self.user_connections:
                print(f"Sending message to user: {target_user_id}")
                await self.send_to_user(target_user_id, message)
            else:
                print(f"User {target_user_id} is not connected")
                
            # 다른 관리자들에게도 메시지 공유
            await self.send_to_admins(message)

    async def generate_response(self, text: str) -> str:
        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,  # temperature 사용을 위해 True로 설정
                    temperature=0.7,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id,  # pad_token_id 설정
                    attention_mask=inputs["attention_mask"]  # attention_mask 추가
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 기본 인사말 설정
            if "안녕" in text:
                return "안녕하세요! AI 뷰티의 아름다움을 공유하는 AI 챗봇입니다. 무엇을 도와드릴까요?"
               
            return response.strip()
           
        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
            return "죄송합니다. 다시 한 번 말씀해 주시겠어요?"

    async def send_ai_response(self, user_message: dict):
        try:
            user_text = user_message['text']
            response = await self.generate_response(user_text)
            
            ai_message = {
                'text': response,
                'userId': user_message['userId'],  # 요청한 사용자 ID 사용
                'userName': 'AI 상담원',
                'type': 'agent',
                'timestamp': datetime.now().isoformat(),
                'tabType': 'ai'
            }
            
            # AI 응답은 해당 사용자에게만 전송
            if user_message['userId'] in self.user_connections:
                await self.send_to_user(user_message['userId'], ai_message)
            
        except Exception as e:
            print(f"응답 처리 중 오류: {e}")
            error_message = {
                'text': "죄송합니다. 잠시 후 다시 시도해 주세요.",
                'userId': user_message['userId'],
                'userName': 'AI 상담원',
                'type': 'agent',
                'timestamp': datetime.now().isoformat(),
                'tabType': 'ai'
            }
            if user_message['userId'] in self.user_connections:
                await self.send_to_user(user_message['userId'], error_message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            print("Received data:", data)  # 디버깅용 로그 추가
            
            # 사용자/관리자 연결 처리
            if 'type' in data:
                if data['type'] == 'user_connect':  # 사용자 연결 확인
                    manager.register_user(data['userId'], websocket)
                    print(f"User connected: {data['userId']}")
                    print(f"Current user connections: {list(manager.user_connections.keys())}")
                    continue
                elif data['type'] == 'admin_connect':  # 관리자 연결 확인
                    admin_id = data.get('adminId', data.get('chatId'))  # adminId나 chatId 중 하나 사용
                    manager.register_admin(admin_id, websocket)
                    print(f"Admin connected: {admin_id}")
                    print(f"Current admin connections: {list(manager.admin_connections.keys())}")
                    continue
            
            # 개인 상담 메시지 처리 (더 자세한 로깅 추가)
            if 'tabType' in data and data['tabType'] == 'personal':
                print(f"Processing personal chat message: {data}")
                await manager.handle_personal_chat(data)
                continue
                
            # AI 상담 메시지 처리
            if 'tabType' in data and data['tabType'] == 'ai' and data['type'] == 'user':
                await manager.send_ai_response(data)
               
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error detail: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)