# server.py
import torch
import re
import os
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. AI 모델 및 커스텀 가중치 로드
# ==========================================
print("모델을 초기화 중입니다...")

# 1-1. 기본 모델 아키텍처 및 토크나이저 로드
base_model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

# 껍데기(아키텍처)만 먼저 로드합니다.
model = GPT2LMHeadModel.from_pretrained(base_model_name)

# 1-2. 학습된 가중치(*.pt) 덮어씌우기
# [주의] 여기에 실제 .pt 파일의 경로를 입력하세요.
MODEL_PATH = "./aac_kogpt2_model.pt"  # 예: "/home/hyeonungyu8/AACommu_model/best_model.pt"

if os.path.exists(MODEL_PATH):
    print(f"커스텀 가중치 파일({MODEL_PATH})을 로드합니다.")
    # map_location='cpu': GPU에서 학습했더라도 CPU 서버에서 돌릴 수 있게 호환성 확보
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # 만약 저장할 때 'model_state_dict' 키로 감싸서 저장했다면 아래 주석 해제
    # if 'model_state_dict' in state_dict:
    #     state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict)
    print("가중치 로드 성공!")
else:
    print(f"[경고] {MODEL_PATH} 파일을 찾을 수 없습니다. 기본 SKT 모델을 사용합니다.")

# 추론 모드 전환 (Dropout 등을 비활성화하여 결과 일관성 유지)
model.eval()

# ==========================================
# 2. 추천 알고리즘 로직 (특수문자/괄호 완벽 제거)
# ==========================================
def generate_next_chunks(category: str, context_question: str, current_answer_list: List[str]) -> List[str]:
    
    current_context_string = " ".join(current_answer_list)
    
    # 프롬프트 구성
    if context_question:
        prompt = f"Q:{context_question} C:{category} A:{current_context_string}"
    else:
        prompt = f"C:{category} A:{current_context_string}"

    if current_context_string and not prompt.endswith(" "):
        prompt += " "

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=3,       # 짧게 생성 (청크)
            num_beams=10,           # 빔 서치 (점수 기반)
            num_return_sequences=3, # 상위 3개 출력
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )

    candidates = []
    
    # [수정됨] 제거할 특수문자 패턴 정의
    # 대괄호[], 중괄호{}, 소괄호(), 꺽쇠<>, 따옴표"', 문장부호.,!? 등 모두 포함
    clean_pattern = r'[.,!?\[\]\{\}\(\)<>\"\'\`~;:]'

    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_part = decoded_text[len(prompt):].strip()
        
        if generated_part:
            # 1. 정의한 패턴에 해당하는 특수문자 모두 제거 (빈 문자열로 치환)
            clean_chunk = re.sub(clean_pattern, '', generated_part)
            
            # 2. 앞뒤 공백 정리
            clean_chunk = clean_chunk.strip()

            # 3. 유효성 검사 (빈 문자열이 아니고 중복이 아닐 때만)
            if clean_chunk and clean_chunk not in candidates:
                candidates.append(clean_chunk)

    # 모델이 적절한 단어를 찾지 못했을 경우의 기본값
    if not candidates:
        return ["네", "아니요", "잠시만요"]
        
    return candidates[:3]

# ==========================================
# 3. 서버 API 정의
# ==========================================
app = FastAPI()

class RequestData(BaseModel):
    category: str
    stt_text: str 
    history: List[str]

@app.post("/predict")
async def predict(data: RequestData):
    results = generate_next_chunks(data.category, data.stt_text, data.history)
    return {"recommendations": results}