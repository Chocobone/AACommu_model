# server.py
import torch
import re
import os
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. AI 모델 및 커스텀 토크나이저 로드
# ==========================================
print("모델 시스템을 초기화 중입니다...")

# [수정됨] 1-1. 로컬 폴더(aac_tokenizer)에서 토크나이저 로드
TOKENIZER_PATH = "./aac_tokenizer"  # 업로드한 파일들이 들어있는 폴더 경로

if os.path.exists(TOKENIZER_PATH):
    print(f"커스텀 토크나이저를 {TOKENIZER_PATH}에서 로드합니다.")
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            TOKENIZER_PATH,
            # KoGPT2 필수 특수 토큰 설정 (config 파일에 있어도 명시적으로 지정하는 것이 안전함)
            bos_token='</s>', 
            eos_token='</s>', 
            unk_token='<unk>',
            pad_token='<pad>', 
            mask_token='<mask>'
        )
    except Exception as e:
        print(f"토크나이저 로드 중 오류 발생: {e}")
        print("기본 SKT 토크나이저로 대체합니다.")
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
else:
    print(f"[경고] {TOKENIZER_PATH} 경로를 찾을 수 없습니다. 기본 SKT 토크나이저를 사용합니다.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')


# 1-2. 모델 로드 및 가중치 적용
base_model_name = "skt/kogpt2-base-v2"
model = GPT2LMHeadModel.from_pretrained(base_model_name)

# 커스텀 가중치 파일 경로
MODEL_PATH = "./model_state_dict.pt" 

if os.path.exists(MODEL_PATH):
    print(f"커스텀 가중치 파일({MODEL_PATH})을 로드합니다.")
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    print("가중치 로드 성공!")
else:
    print(f"[경고] {MODEL_PATH} 파일을 찾을 수 없습니다. 기본 모델을 사용합니다.")

# [중요] 토크나이저에 새로운 단어가 추가되어 vocab 크기가 바뀌었을 경우를 대비해 모델 임베딩 크기 조정
if len(tokenizer) != model.transformer.wte.weight.shape[0]:
    print(f"토크나이저 크기({len(tokenizer)})와 모델 크기({model.transformer.wte.weight.shape[0]})를 맞춥니다.")
    model.resize_token_embeddings(len(tokenizer))

model.eval()

# ==========================================
# 2. 추천 알고리즘 (이전과 동일)
# ==========================================
def generate_next_chunks(category: str, context_question: str, current_answer_list: List[str]) -> List[str]:
    
    current_context_string = " ".join(current_answer_list)
    
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
            max_new_tokens=3,       
            num_beams=10,           
            num_return_sequences=3, 
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )

    candidates = []
    # 특수문자 제거 패턴
    clean_pattern = r'[.,!?\[\]\{\}\(\)<>\"\'\`~;:]'

    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_part = decoded_text[len(prompt):].strip()
        
        if generated_part:
            clean_chunk = re.sub(clean_pattern, '', generated_part)
            clean_chunk = clean_chunk.strip()

            if clean_chunk and clean_chunk not in candidates:
                candidates.append(clean_chunk)

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