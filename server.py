from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn # 모델 클래스 정의를 위해 필요
import numpy as np

# ---------------------------------------------
# 1. 모델 클래스 정의 (저장 시점과 동일해야 함)
# 예: 간단한 Linear Layer를 가진 모델이라고 가정
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 입력 데이터의 피처(Feature) 개수에 맞게 수정해야 합니다.
        self.fc = nn.Linear(in_features=4, out_features=1) 

    def forward(self, x):
        return self.fc(x)
# ---------------------------------------------


app = FastAPI()

# ⚠️ 서버 시작 시 한 번만 로드하는 것이 중요합니다!
# 모델 클래스 인스턴스 생성
model = SimpleModel()

# 학습된 가중치 로드
try:
    # 'my_pytorch_model.pt' 파일을 사용한다고 가정
    model.load_state_dict(torch.load("my_pytorch_model.pt", map_location=torch.device('cpu')))
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    # 서버 실행 실패 시 처리
    
# ⭐️ 추론(Inference) 모드로 전환:
# Dropout, BatchNorm 등이 비활성화되어 일관된 예측을 보장합니다.
model.eval()

# 3. 입력 데이터 형식 정의 (Pydantic)
# 예: 4개의 입력 피처(x1, x2, x3, x4)를 받는다고 가정
class InputData(BaseModel):
    features: list[float] # 길이가 4인 리스트를 기대

# 4. API 엔드포인트 정의
@app.post("/predict")
def predict(data: InputData):
    
    # ⭐️ 1. 입력 데이터를 PyTorch 텐서(Tensor)로 변환
    # (data.features는 [x1, x2, x3, x4] 형태의 리스트)
    input_tensor = torch.tensor(data.features, dtype=torch.float32).unsqueeze(0)
    # .unsqueeze(0): (4) -> (1, 4) 형태로 배치 차원(Batch Dimension) 추가
    
    # ⭐️ 2. 예측 수행 (Gradient 계산 비활성화)
    with torch.no_grad():
        output = model(input_tensor)
        
    # 3. 결과 후처리
    # 예: 결과를 NumPy로 변환하고, 파이썬 기본 float 형태로 추출
    prediction_result = output.squeeze().cpu().numpy().item()
    
    # 4. JSON 응답 반환
    return {
        "prediction": prediction_result,
        "status": "success"
    }

# 실행: uvicorn main:app --reload