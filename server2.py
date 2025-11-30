# server.py
import torch
import re
import os
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. 모델 로드 (GPT + BERT)
# ==========================================
print("시스템 초기화 중... (GPU 메모리 확보 필요)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- (1) KoGPT2 로드 (생성 담당) ---
GPT_MODEL_PATH = "./aac_kogpt2_model.pt"
TOKENIZER_PATH = "./aac_tokenizer"

if os.path.exists(TOKENIZER_PATH):
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
else:
    # 토크나이저 폴더가 없으면 기본 로드 (성능 저하 주의)
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')

gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
gpt_model.resize_token_embeddings(len(gpt_tokenizer))

if os.path.exists(GPT_MODEL_PATH):
    gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
    print("✅ GPT 모델 로드 완료")
else:
    print("⚠️ GPT 모델 파일이 없습니다. 기본 모델로 동작합니다.")

gpt_model.to(device)
gpt_model.eval()

# --- (2) BERT 로드 (검증/채점 담당) ---
BERT_MODEL_PATH = "./aac_bert_model.pt"
bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

bert_model = BertClassifier()
if os.path.exists(BERT_MODEL_PATH):
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    print("✅ BERT 모델 로드 완료")
else:
    print("⚠️ BERT 모델 파일이 없습니다. 검증 기능이 작동하지 않을 수 있습니다.")

bert_model.to(device)
bert_model.eval()

# ==========================================
# 2. 로직: 생성(Generate) -> 검증(Rank)
# ==========================================

def get_best_candidates(stt_question: str, current_history: List[str]) -> List[str]:
    # 1. 문맥 정리
    current_context = " ".join(current_history)
    
    # 프롬프트 생성 (학습 포맷 준수)
    if current_context:
        prompt = f"<usr>{stt_question}<sys>{current_context}"
    else:
        prompt = f"<usr>{stt_question}<sys>"

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 2. GPT가 후보 15개 생성 (많이 뽑아서 BERT에게 넘김)
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids,
            max_new_tokens=6,       # 조금 더 길게 봐서 문맥 파악
            num_beams=15,           # 빔 개수 증가
            num_return_sequences=15, # 후보 15개 리턴
            repetition_penalty=2.5, # 반복 강력 억제
            do_sample=True,         # 샘플링 허용 (다양성 확보)
            temperature=0.7,        # 창의성 약간 억제 (정확도 위주)
            top_k=50,
            eos_token_id=gpt_tokenizer.eos_token_id,
            pad_token_id=gpt_tokenizer.pad_token_id
        )

    # 3. 후보군 1차 필터링 (특수문자 제거 등)
    raw_candidates = []
    clean_pattern = r'[\[\]\{\}\(\)<>\"\'\`~;:]' # 점(.)은 살려둠(문장 끝 판단용)

    for output in outputs:
        decoded = gpt_tokenizer.decode(output, skip_special_tokens=False)
        if "<sys>" in decoded:
            generated = decoded.split("<sys>")[1]
            if current_context and current_context in generated:
                generated = generated.replace(current_context, "", 1)
        else:
            generated = decoded

        # 태그 및 특수문자 정리
        generated = generated.replace("</s>", "").replace("<pad>", "").strip()
        generated = re.sub(clean_pattern, '', generated).strip()
        
        # 첫 어절 추출 (단어 단위 추천)
        if generated:
            first_word = generated.split(' ')[0]
            # 이미 선택한 단어거나, 너무 짧은 조사 등은 제외 가능
            if first_word and first_word not in raw_candidates:
                raw_candidates.append(first_word)

    # 후보가 너무 적으면 기본값 추가
    if len(raw_candidates) < 3:
        raw_candidates.extend(["네", "아니요", "감사합니다"])
    
    # 중복 제거
    raw_candidates = list(dict.fromkeys(raw_candidates))

    # =========================================================
    # 4. [핵심] BERT로 후보 채점 (Re-ranking)
    # =========================================================
    scored_candidates = []
    
    with torch.no_grad():
        for cand in raw_candidates:
            # BERT에게 물어봄: "질문(Q)에 대해, 기존문장+이단어(A)가 적절하니?"
            # 예: Q="드시고 가시나요?", A="아 다행이다" -> BERT Score 낮음
            # 예: Q="드시고 가시나요?", A="네" -> BERT Score 높음
            
            full_answer = f"{current_context} {cand}".strip()
            
            # BERT 입력 포맷
            text = f"{stt_question} [SEP] {full_answer}"
            inputs = bert_tokenizer(
                text, return_tensors='pt', truncation=True, max_length=128, padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = bert_model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            score = probs[0][1].item() # '적절함(Label 1)'일 확률
            
            scored_candidates.append((cand, score))

    # 5. 점수 높은 순으로 정렬
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"BERT 채점 결과: {scored_candidates[:5]}") # 로그 확인용

    # 상위 3개 단어만 반환
    final_result = [item[0] for item in scored_candidates[:3]]
    return final_result


# ==========================================
# 3. API 서버
# ==========================================
app = FastAPI()

class RequestData(BaseModel):
    category: str
    stt_text: str 
    history: List[str]

@app.post("/predict")
async def predict(data: RequestData):
    results = get_best_candidates(data.stt_text, data.history)
    return {"recommendations": results}