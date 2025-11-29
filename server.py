from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

# 1. 설정
MODEL_PATH = "AACommu_model_best.pt"
TOKENIZER_NAME = "klue/bert-base" # ⭐️ 학습 때 쓴 모델명과 일치해야 함

app = FastAPI()

# 2. 모델 클래스 정의
class MyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ⭐️ 여기서도 klue/bert-base를 로드해야 사이즈 오류가 안 납니다.
        self.bert = BertModel.from_pretrained(TOKENIZER_NAME)
        self.out = nn.Linear(768, 2) # 분류 모델 (예: 0 또는 1 예측)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(output.pooler_output)

# 3. 로딩
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = MyLanguageModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# map_location 추가하여 안전하게 로드
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 4. 입력 데이터
class PromptRequest(BaseModel):
    text: str

# ⭐️ 엔드포인트 이름을 기능에 맞게 수정 (예: 분류 예측)
@app.post("/predict/classification")
def predict_classification(req: PromptRequest):
    # A. 토크나이징 (attention_mask도 함께 받음)
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device) # ⭐️ 필수 인자

    # B. 추론
    with torch.no_grad():
        # forward 함수에 인자 2개 모두 전달
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # C. 결과 처리 (분류 모델이므로 가장 높은 점수의 클래스 선택)
    # outputs shape: [1, 2] -> 단어 생성이 아니라 클래스 확률임
    predicted_class_id = torch.argmax(outputs, dim=1).item()
    
    return {
        "input_text": req.text,
        "predicted_class": predicted_class_id, # 0 또는 1
        "message": "분류가 완료되었습니다."
    }
class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


@app.get("/")
async def health_check():
    return HealthResponse(status='ok')