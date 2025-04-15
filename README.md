# HairAI 백엔드 서비스

HairAI는 AI 기반 헤어스타일 추천 및 가상 체험 서비스의 백엔드 시스템입니다. FastAPI를 기반으로 개발되었으며, 
ComfyUI를 활용한 고급 이미지 처리 기능과 Firebase 연동을 통한 데이터 관리 및 실시간 AI 상담 기능을 제공합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [시스템 요구사항](#시스템-요구사항)
- [설치 및 설정](#설치-및-설정)
- [프로젝트 구조](#프로젝트-구조)
- [API 문서](#api-문서)
- [개발 가이드](#개발-가이드)
- [문제 해결](#문제-해결)
- [라이선스](#라이선스)

## 주요 기능

### 💇‍♀️ 헤어스타일 변환
- 사용자 얼굴 이미지 분석 및 헤어스타일 추천
- 얼굴형에 맞는 헤어스타일 시뮬레이션
- 주파수 분리 기술을 활용한 자연스러운 합성

### 👤 얼굴 변환
- ReActor 기술을 활용한 고품질 얼굴 교체
- 얼굴 특징 보존 및 자연스러운 블렌딩

### 🖼️ 배경 처리
- **배경 제거**: RMBG 기술을 활용한 정확한 배경 제거
- **배경 생성**: 생성형 AI를 활용한 맞춤형 배경 생성
  - 다양한 테마 및 스타일 지원
  - 조명 효과 및 색상 커스터마이징

### 💬 AI 상담 시스템
- Llama3 기반 전문 헤어 상담 AI
- 실시간 WebSocket 통신
- Firebase를 활용한 채팅 기록 저장 및 관리

## 기술 스택

- **백엔드**: FastAPI, Python 3.8+
- **AI 모델**: ComfyUI (SD 1.5 기반), Llama3 (8B 모델)
- **데이터베이스**: Firebase Firestore
- **스토리지**: Firebase Storage
- **통신**: WebSocket, REST API
- **배포**: Docker (선택 사항)

## 시스템 요구사항

- **OS**: Windows 10+ 또는 Linux (Ubuntu 20.04+ 권장)
- **RAM**: 최소 16GB (32GB 이상 권장)
- **GPU**: NVIDIA GPU (최소 8GB VRAM)
- **스토리지**: 최소 20GB 여유 공간
- **네트워크**: 안정적인 인터넷 연결

## 설치 및 설정

### 1. 사전 준비

- Python 3.8 이상 설치
- ComfyUI 설치 및 설정 (https://github.com/comfyanonymous/ComfyUI)
- Firebase 프로젝트 생성 및 API 키 발급
- Ollama 설치 (https://ollama.ai)

### 2. 저장소 복제 및 의존성 설치

```bash
# 저장소 복제
git clone https://github.com/yourusername/hairai-backend.git
cd hairai-backend

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 내용을 추가합니다:

```
COMFYUI_API_URL=http://127.0.0.1:8188
COMFYUI_OUTPUT_DIR=output
BACKCLEAR_WORKFLOW_PATH=workflow/BackClear.json
BACKGROUND_WORKFLOW_PATH=workflow/BackCreate.json
HAIRSTYLE_WORKFLOW_PATH=workflow/HAIReditFinish.json
```

### 4. Firebase 설정

1. Firebase 콘솔에서 서비스 계정 키를 다운로드
2. 다운로드한 키 파일을 `firebase-adminsdk-credentials.json`으로 저장하여 프로젝트 루트에 위치

### 5. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 프로젝트 구조

```
hairai-backend/
├── main.py               # FastAPI 애플리케이션 메인 파일
├── Aichat.py             # AI 상담 서비스 모듈
├── HairStyle.py          # 헤어스타일 변환 모듈
├── face.py               # 얼굴 변환 모듈
├── backclear.py          # 배경 제거 모듈
├── backcreate.py         # 배경 생성 모듈
├── requirements.txt      # 의존성 목록
├── .env                  # 환경 변수 (gitignore에 포함됨)
├── .gitignore            # Git 무시 파일 목록
├── firebase-adminsdk-credentials.json  # Firebase 인증 키 (비공개)
└── workflow/             # ComfyUI 워크플로우 JSON 파일
    ├── BackClear.json    # 배경 제거 워크플로우
    ├── BackCreate.json   # 배경 생성 워크플로우
    ├── HAIReditFinish.json  # 헤어스타일 변환 워크플로우
    └── facedefault.json  # 얼굴 변환 워크플로우
```

## API 문서

서버를 실행한 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 주요 엔드포인트

#### 헤어스타일 변환
```http
POST /api/hairstyle
```
- **요청**: 멀티파트 폼 데이터 (이미지 파일 + JSON 파라미터)
- **응답**: 변환된 이미지 URL 목록

#### 얼굴 변환
```http
POST /api/transform
```
- **요청**: 멀티파트 폼 데이터 (원본 이미지 + 롤모델 이미지)
- **응답**: 변환된 이미지 URL

#### 배경 제거
```http
POST /api/backclear
```
- **요청**: 멀티파트 폼 데이터 (이미지 파일)
- **응답**: 배경이 제거된 이미지 URL

#### 배경 생성
```http
POST /api/background
```
- **요청**: 멀티파트 폼 데이터 (이미지 파일 + 프롬프트 JSON)
- **응답**: 새 배경이 적용된 이미지 URL

#### AI 상담
```http
WebSocket /ws
```
- 실시간 대화식 AI 상담 인터페이스

## 개발 가이드

### 코드 스타일
- PEP 8 가이드라인 준수
- 함수 및 클래스에 docstring 추가
- 타입 힌트 사용

### 모듈 구조
각 기능은 독립적인 모듈로 구성되어 있어 확장성과 유지보수성을 높였습니다:

1. **main.py**: API 엔드포인트 및 서버 설정
2. **Aichat.py**: WebSocket 기반 AI 상담 시스템
3. **HairStyle.py**: 헤어스타일 변환 로직
4. **face.py**: 얼굴 변환 로직
5. **backclear.py/backcreate.py**: 배경 처리 로직

### 워크플로우 커스터마이징
ComfyUI 워크플로우 파일(.json)을 수정하여 이미지 처리 파이프라인을 조정할 수 있습니다:

1. ComfyUI 웹 인터페이스에서 워크플로우 수정
2. JSON 파일로 내보내기
3. 해당 파일을 workflow/ 디렉토리에 저장
4. 환경 변수에서 경로 업데이트

## 문제 해결

### 일반적인 문제
- **ConnectionError**: ComfyUI 서버가 실행 중인지 확인
- **메모리 부족**: 배치 크기 및 이미지 해상도 조정
- **느린 처리 속도**: 
  - GPU 메모리 확인
  - 워크플로우 최적화
  - 배치 처리 사용

### 로깅
문제 진단을 위해 logs/ 디렉토리의 로그 파일을 확인하세요:
```bash
tail -f logs/api.log
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

© 2025 HairAI Team. All Rights Reserved.
