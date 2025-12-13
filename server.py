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
GPT_MODEL_PATH = "./aac_kogpt2_dir_tag_model.pt"
TOKENIZER_PATH = "./aac_tokenizer"

# 1. 토크나이저 로드 및 보정
if os.path.exists(TOKENIZER_PATH):
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
else:
    # 토크나이저 폴더가 없으면 기본 모델 로드
    print("⚠️ 저장된 토크나이저가 없어 기본 모델을 로드합니다.")
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')
    
    # [핵심 수정] 저장된 모델(51202)과 크기를 맞추기 위해 특수 토큰 강제 추가
    # 차이 나는 2개는 보통 <usr>, <sys> 입니다.
    special_tokens = ['<usr>', '<sys>']
    gpt_tokenizer.add_tokens(special_tokens)
    print(f"   👉 특수 토큰 추가됨: {special_tokens} (Size: {len(gpt_tokenizer)})")

# 2. 모델 로드 및 리사이징
gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 토크나이저 크기(51202 예상)에 맞춰 모델 임베딩 늘리기
gpt_model.resize_token_embeddings(len(gpt_tokenizer))

# 3. 가중치 적용 (안전 장치 포함)
if os.path.exists(GPT_MODEL_PATH):
    try:
        # weights_only=False는 파이토치 버전 호환성을 위해 추가될 수 있음
        state_dict = torch.load(GPT_MODEL_PATH, map_location=device)
        gpt_model.load_state_dict(state_dict)
        print("✅ GPT 모델 로드 완료")
    except RuntimeError as e:
        # 만약 그래도 크기가 안 맞으면, 모델 파일의 크기에 맞춰 강제 리사이징 시도
        print(f"⚠️ 모델 크기 불일치 감지! 강제 조정 시도... ({e})")
        # 에러 메시지에서 타겟 사이즈 추출 (예: 51202)
        # 임시로 51202로 맞추고 재시도
        gpt_model.resize_token_embeddings(51202) 
        gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
        print("✅ GPT 모델 강제 로드 성공 (주의: 토크나이저와 매핑이 어긋날 수 있음)")
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

# server.py의 get_best_candidates 함수를 이걸로 덮어쓰세요.

def get_best_candidates(category: str, stt_question: str, current_history: List[str]) -> List[str]:
    # 1. 문맥 정리
    current_context = " ".join(current_history)
    
    # 프롬프트 생성
    if current_context:
        prompt = f"<usr>{stt_question}<sys>{current_context}"
    else:
        prompt = f"<usr>{stt_question}<sys>"

    # 입력 텐서 생성
    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 2. GPT 생성
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids,
            max_new_tokens=8,
            num_beams=15,
            num_return_sequences=15, 
            repetition_penalty=3.0,
            do_sample=True,          
            temperature=0.7,         
            top_k=50,
            eos_token_id=gpt_tokenizer.eos_token_id,
            pad_token_id=gpt_tokenizer.pad_token_id
        )

    # 3. 1차 필터링 (불량 토큰 제거)
    raw_candidates = []
    
    # 조사를 걸러내기 위한 리스트 (이걸로 시작하면 버림)
    bad_starts = ["을", "를", "이", "가", "은", "는", "로", "에", "서", "고", "지", "만", "요"]
    # 한 글자라도 살려야 하는 단어들
    valid_singles = ["네", "물", "컵", "약", "밥", "면", "국", "돈"]

    for output in outputs:
        # [수정] output 텐서를 리스트로 변환하여 decode 에러 방지
        decoded = gpt_tokenizer.decode(output.tolist(), skip_special_tokens=False)
        
        # <sys> 뒤의 내용 추출
        if "<sys>" in decoded:
            try:
                generated = decoded.split("<sys>")[1]
            except IndexError:
                generated = decoded
            
            # 문맥이 포함되어 있다면 제거
            if current_context and current_context in generated:
                generated = generated.replace(current_context, "", 1)
        else:
            generated = decoded

        # 특수문자 및 태그 제거
        clean_pattern = r'[\[\]\{\}\(\)<>\"\'\`~;:,.!?]' 
        generated = generated.replace("</s>", "").replace("<pad>", "").strip()
        generated = re.sub(clean_pattern, '', generated).strip()
        
        if not generated:
            continue

        # 첫 어절만 추출
        first_word = generated.split(' ')[0]
        
        # 규칙 1: 너무 짧은데 의미 없는 말 제거
        if len(first_word) == 1 and first_word not in valid_singles:
            continue
            
        # 규칙 2: 조사로 시작하는 말 제거
        is_bad_start = False
        for bad in bad_starts:
            if first_word.startswith(bad):
                is_bad_start = True
                break
        if is_bad_start:
            continue
            
        # 규칙 3: 중복 제거 및 리스트 추가
        if first_word and first_word not in raw_candidates:
            raw_candidates.append(first_word)

    # 4. BERT 채점 (Re-ranking)
    scored_candidates = []
    
    # 후보가 아예 없으면 기본값 리턴
    if not raw_candidates:
        return ["네", "아니요", "감사합니다"]

    with torch.no_grad():
        for cand in raw_candidates:
            full_answer = f"{current_context} {cand}".strip()
            
            # 질문 + (현재문맥 + 후보단어) 쌍으로 검증
            text = f"{stt_question} [SEP] {full_answer}"
            
            inputs = bert_tokenizer(
                text, return_tensors='pt', truncation=True, max_length=128, padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = bert_model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            score = probs[0][1].item() 
            
            # 규칙 4: BERT 점수가 0.4점 미만이면 과감히 버림
            if score >= 0.4:
                scored_candidates.append((cand, score))

    # 5. 최종 정렬 및 반환
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"BERT 최종 후보(Score 0.4↑): {scored_candidates[:5]}") 

    # 상위 3개 추출
    final_result = [item[0] for item in scored_candidates[:3]]
    
    # 규칙 5: 결과가 3개 미만이면 기본 단어로 채움 (안전장치)
    defaults = ["네", "아니요", "감사합니다", "잠시만요"]
    for d in defaults:
        if len(final_result) < 3:
            if d not in final_result:
                final_result.append(d)
        else:
            break

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
    results = get_best_candidates(data.category, data.stt_text, data.history)
    return {"recommendations": results}