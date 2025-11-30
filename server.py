# server.py
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. AI 모델 로드
# ==========================================
print("모델을 로드 중입니다... 잠시만 기다려주세요.")
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained(model_name)
print("모델 로드 완료!")

# ==========================================
# 2. 추천 알고리즘 로직 수정 (Beam Search & Short Generation)
# ==========================================
def generate_next_chunks(category: str, context_question: str, current_answer_list: List[str]) -> List[str]:
    """
    category: 장소/상황 (예: "카페")
    context_question: 상대방의 질문 (예: "주문하시겠어요?")
    current_answer_list: 사용자가 지금까지 선택한 청크들의 리스트 (예: ["아이스", "아메리카노"])
    """
    
    # 1. 리스트로 들어온 히스토리를 하나의 문장으로 합침
    # (한국어 띄어쓰기 고려: 청크 사이에 공백 추가)
    current_context_string = " ".join(current_answer_list)
    
    # 2. 프롬프트 구성 (KoGPT2가 이해하기 쉬운 구조로 변경)
    # Q: 질문, C: 상황, A: 답변 시작 부분
    if context_question:
        prompt = f"Q:{context_question} C:{category} A:{current_context_string}"
    else:
        prompt = f"C:{category} A:{current_context_string}"

    # 답변이 비어있지 않다면 자연스러운 연결을 위해 공백을 추가할 수도 있음
    if current_context_string and not prompt.endswith(" "):
        prompt += " "

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # 3. 핵심 알고리즘: Beam Search (점수 기반 추천)
    # 문장을 끝까지 만드는 게 아니라, '다음에 올 가장 확률 높은 단어'를 찾습니다.
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=3,    # 핵심: 3~4 토큰(단어 1~2개 분량)만 짧게 생성
            num_beams=10,        # 10개의 후보 경로를 탐색 (Scoring)
            num_return_sequences=3, # 점수가 가장 높은 상위 3개를 리턴
            no_repeat_ngram_size=2, # 동일한 단어 반복 방지
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )

    candidates = []
    for output in outputs:
        # 전체 문장에서 프롬프트(입력값)를 제외한 '새로 생성된 부분'만 추출
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        # prompt 길이만큼 잘라내기
        generated_part = decoded_text[len(prompt):].strip()
        
        # 생성된 텍스트가 비어있지 않다면 추가
        # (짧게 생성했으므로 generated_part 자체가 하나의 청크가 될 가능성이 높음)
        if generated_part:
            # 혹시 여러 단어가 생성되었다면 첫 어절만 가져오기 or 전체 사용
            # 여기서는 '청크' 개념이므로 짧은 구(Phrase) 전체를 사용
            clean_chunk = generated_part.split('.')[0] # 문장 끝 점(.)이 나오면 제거
            if clean_chunk not in candidates:
                candidates.append(clean_chunk)

    # 중복 제거 후 3개가 안 될 경우를 대비해 빈 값 처리 혹은 기본값
    if not candidates:
        return ["네", "아니요", "잠시만요"]
        
    # 최대 3개까지만 반환
    return candidates[:3]

# ==========================================
# 3. 서버 API 정의
# ==========================================
app = FastAPI()

class RequestData(BaseModel):
    category: str
    stt_text: str 
    history: List[str] # 사용자 입력 히스토리를 리스트로 받음

@app.post("/predict")
async def predict(data: RequestData):
    results = generate_next_chunks(data.category, data.stt_text, data.history)
    return {"recommendations": results}