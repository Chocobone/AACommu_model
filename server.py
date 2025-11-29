# server.py
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. AI 모델 로드 (서버 켤 때 한 번만 실행됨)
# ==========================================
print("모델을 로드 중입니다... 잠시만 기다려주세요.")
# SKT의 KoGPT-2 모델 사용 (한국어 성능 우수)
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained(model_name)
print("모델 로드 완료!")

# ==========================================
# 2. 추천 알고리즘 로직 (핵심)
# ==========================================
def generate_next_chunks(context_question: str, current_answer: str) -> List[str]:
    """
    context_question: 상대방의 질문 (예: "점심 뭐 먹을래?")
    current_answer: 내가 지금까지 입력한 답변 (예: "나는 오늘")
    """
    
    # 모델에게 줄 프롬프트 구성 (질문과 답변을 이어붙임)
    # Q와 A라는 태그를 붙여서 모델이 대화 상황임을 인지하게 유도
    if context_question:
        prompt = f"Q: {context_question}\nA: {current_answer}"
    else:
        prompt = f"A: {current_answer}"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # -------------------------------------------------
    # 핵심 알고리즘: Top-k & Top-p Sampling
    # 확률이 높은 단어 하나만 뽑는 게 아니라, 
    # 그럴싸한 후보 여러 개를 다양하게 뽑아내는 기법입니다.
    # -------------------------------------------------
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 5,  # 현재 길이보다 5토큰(약 2~3어절) 더 예측
            do_sample=True,      # 확률적 샘플링 사용 (매번 조금씩 다른 추천)
            top_k=50,            # 확률 상위 50개 후보 중에서만 선택
            top_p=0.95,          # 누적 확률 95% 내의 후보만 선택 (이상한 단어 제외)
            temperature=0.8,     # 창의성 조절 (낮을수록 정확, 높을수록 창의적)
            num_return_sequences=3, # 서로 다른 후보 3개를 생성해라
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=2.0  # 했던 말 또 하지 않게 페널티 부여
        )

    candidates = []
    for output in outputs:
        # 모델이 생성한 전체 문장에서 프롬프트 부분을 제거하고 '새로 생성된 부분'만 추출
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_part = decoded_text[len(prompt):].strip()
        
        # 첫 번째 공백이나 문장 부호를 기준으로 '청크'를 자름
        # 예: "김치찌개 먹고 싶어" -> "김치찌개"
        chunks = generated_part.split()
        if chunks:
            # 첫 번째 덩어리만 추천 후보로 등록
            first_chunk = chunks[0]
            if first_chunk not in candidates: # 중복 제거
                candidates.append(first_chunk)

    # 만약 모델이 아무것도 추천 못했다면 기본값 제공 (에러 방지)
    if not candidates:
        return ["...", "음,", "저기"]

    return candidates

# ==========================================
# 3. 서버 API 정의 (FastAPI)
# ==========================================
app = FastAPI()

class RequestData(BaseModel):
    partner_speech: str  # 상대방 말
    my_speech: str       # 내 현재 입력

@app.post("/recommend")
async def recommend(data: RequestData):
    # 위에서 만든 알고리즘 실행
    results = generate_next_chunks(data.partner_speech, data.my_speech)
    return {"recommendations": results}