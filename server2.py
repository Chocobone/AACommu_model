# server.py
import torch
import re
import os
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. 모델 및 토크나이저 로드 (학습 환경과 동일하게 맞춤)
# ==========================================
print("시스템을 초기화 중입니다...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# [중요 1] 학습 때 저장한 '토크나이저 폴더'를 로드해야 합니다.
# (그래야 <usr>, <sys> 토큰을 인식합니다)
TOKENIZER_PATH = "./aac_tokenizer" 
MODEL_PATH = "./aac_kogpt2_model.pt"

if os.path.exists(TOKENIZER_PATH):
    print(f"저장된 토크나이저를 로드합니다: {TOKENIZER_PATH}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
else:
    print("🚨 [경로 오류] './aac_tokenizer' 폴더가 없습니다! 학습 코드를 다시 실행해 토크나이저를 저장해주세요.")
    # 임시로 기본 로드 (성능 저하 원인)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')

# 1-2. 모델 아키텍처 로드
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# [중요 2] 모델의 임베딩 크기를 토크나이저에 맞춰 늘려줍니다.
model.resize_token_embeddings(len(tokenizer))

# 1-3. 학습된 가중치 로드
if os.path.exists(MODEL_PATH):
    print(f"커스텀 가중치 파일({MODEL_PATH})을 로드합니다.")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 가중치 로드 성공!")
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}")
        print("토크나이저 크기와 모델 크기가 맞지 않을 수 있습니다.")
else:
    print(f"[경고] {MODEL_PATH} 파일을 찾을 수 없습니다.")

model.to(device)
model.eval()

# ==========================================
# 2. 추천 알고리즘 (프롬프트 수정됨)
# ==========================================
def generate_next_chunks(category: str, context_question: str, current_answer_list: List[str]) -> List[str]:
    
    # 사용자 히스토리 합치기 (예: "아이스", "아메리카노" -> "아이스 아메리카노")
    current_context_string = " ".join(current_answer_list)
    
    # [중요 3] 프롬프트 형식을 학습 데이터와 100% 동일하게 변경
    # 학습 포맷: <usr>상대방말<sys>나의말
    # category(장소)는 학습에 안 썼으므로 제거하거나, 텍스트에 자연스럽게 녹여야 함.
    # 여기서는 학습 데이터 포맷을 엄격히 따릅니다.
    
    if current_context_string:
        # 이미 대답을 하고 있는 중이라면
        prompt = f"<usr>{context_question}<sys>{current_context_string}"
    else:
        # 대답을 시작하는 단계라면
        prompt = f"<usr>{context_question}<sys>"

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=5,       # 단어 1~2개 추천이므로 짧게 설정
            num_beams=5,            # 빔 서치
            num_return_sequences=3, # 상위 3개
            repetition_penalty=2.0, # 반복 방지 (ngram_size보다 자연스러움)
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    candidates = []
    
    # 특수문자 제거 패턴 (점(.)은 살려둘지 고민 필요하지만, 일단 제거)
    clean_pattern = r'[\[\]\{\}\(\)<>\"\'\`~;:]' # 마침표(.) 물음표(?) 제외

    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=False)
        
        # <sys> 태그 뒤의 내용만 추출
        if "<sys>" in decoded_text:
            generated_part = decoded_text.split("<sys>")[1]
            
            # 입력했던 히스토리(prefix) 제거
            # 예: 입력 "아이스", 생성 "아이스 아메리카노" -> " 아메리카노" 추출
            if current_context_string and current_context_string in generated_part:
                generated_part = generated_part.replace(current_context_string, "", 1)
        else:
            generated_part = decoded_text

        # </s> 등 특수 토큰 제거
        generated_part = generated_part.replace("</s>", "").replace("<pad>", "").strip()
        
        # 정규식으로 특수문자 제거
        clean_chunk = re.sub(clean_pattern, '', generated_part).strip()

        # 첫 어절만 가져오기 (단어 단위 추천을 위해)
        # 문장이 통째로 나오면 버튼에 넣기 힘드므로 공백 기준으로 자름
        if clean_chunk:
            first_word = clean_chunk.split(' ')[0]
            
            # 유효성 검사
            if first_word and first_word not in candidates:
                candidates.append(first_word)

    # 결과가 없으면 기본값
    if not candidates:
        return ["네", "아니요", "잠시만요"]
        
    return candidates[:3]

# ==========================================
# 3. 서버 API
# ==========================================
app = FastAPI()

class RequestData(BaseModel):
    category: str
    stt_text: str 
    history: List[str]

@app.post("/predict")
async def predict(data: RequestData):
    results = generate_next_chunks(data.category, data.stt_text, data.history)
    print(f"Input: {data.stt_text} / Hist: {data.history} -> Out: {results}")
    return {"recommendations": results}