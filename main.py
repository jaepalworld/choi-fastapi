from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
from datetime import datetime


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
   return {"message": "Welcome to Chat API"}

# 모델과 토크나이저 로드
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class ConnectionManager:
   def __init__(self):
       self.active_connections: List[WebSocket] = []
       
   async def connect(self, websocket: WebSocket):
       await websocket.accept()
       self.active_connections.append(websocket)
       
   def disconnect(self, websocket: WebSocket):
       self.active_connections.remove(websocket)
       
   async def broadcast(self, message: dict):
       for connection in self.active_connections:
           await connection.send_json(message)

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
               'userId': 'ai_agent',
               'userName': 'AI 상담원',
               'type': 'agent',
               'timestamp': datetime.now().isoformat(),
               'tabType': 'ai'
           }
           
           await self.broadcast(ai_message)
           
       except Exception as e:
           print(f"응답 처리 중 오류: {e}")
           error_message = {
               'text': "죄송합니다. 잠시 후 다시 시도해 주세요.",
               'userId': 'ai_agent',
               'userName': 'AI 상담원',
               'type': 'agent',
               'timestamp': datetime.now().isoformat(),
               'tabType': 'ai'
           }
           await self.broadcast(error_message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
   await manager.connect(websocket)
   try:
       while True:
           data = await websocket.receive_json()
           await manager.broadcast(data)
           
           if data['tabType'] == 'ai' and data['type'] == 'user':
               await manager.send_ai_response(data)
               
   except Exception as e:
       print(f"WebSocket 오류: {e}")
       manager.disconnect(websocket)

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)