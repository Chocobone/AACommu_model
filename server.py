# server.py (데이터 수정 없이 성능 극대화 버전)
import torch
import re
import os
import random
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ==========================================
# 1. 모델 로드 (기존과 동일)
# ==========================================
print("🚀 AI 서버(Inference Boost Mode) 가동 중...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 정의
class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

# 경로 설정
GPT_MODEL_PATH = "./aac_kogpt2_model.pt"
TOKENIZER_PATH = "./aac_tokenizer"
BERT_MODEL_PATH = "./aac_bert_model.pt"

# 로드 로직
if os.path.exists(TOKENIZER_PATH):
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
else:
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
gpt_model.resize_token_embeddings(len(gpt_tokenizer))
if os.path.exists(GPT_MODEL_PATH):
    gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
gpt_model.to(device); gpt_model.eval()

bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
bert_model = BertClassifier()
if os.path.exists(BERT_MODEL_PATH):
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
bert_model.to(device); bert_model.eval()

# ==========================================
# 2. [치트키] 정답 단어장 (Dictionary Injection)
# ==========================================
# 학습 데이터가 부족하니, 여기서 '지식'을 보충해줍니다.
# AI가 이 단어를 뱉으면 점수를 확 올려주고, 못 뱉으면 강제로 넣어줍니다.

PREFERRED_WORDS = {
    "카페": [
        "아메리카노", "라떼", "카페모카", "바닐라", "아이스티", "에이드", "스무디", 
        "티", "주스", "케이크", "베이글", "마카롱", "샌드위치", "쿠키",
        "얼음", "시럽", "휘핑", "샷", "사이즈", "테이크아웃", "매장", "진동벨", "쿠폰", "적립"
    ],
    "식당": [
        "김치찌개", "된장찌개", "볶음밥", "덮밥", "돈까스", "우동", "라면", "김밥",
        "반찬", "공기밥", "물", "김치", "단무지", "메뉴판", "주문", "계산", "포장",
        "1인분", "2인분", "매운거", "안맵게", "앞치마"
    ],
    "편의점": [
        "봉투", "영수증", "담배", "라이터", "교통카드", "충전", "도시락", "김밥",
        "음료수", "물", "맥주", "소주", "과자", "라면", "행사", "1+1", "2+1"
    ],
    "공통": [
        "네", "아니요", "좋아요", "싫어요", "감사합니다", "잠시만요", "얼마에요?", 
        "주세요", "해주세요", "없어요", "있어요"
    ]
}

BAD_STARTS = ["을", "를", "이", "가", "은", "는", "로", "에", "서", "고", "지", "만", "요", "도", "의", "면"]

# ==========================================
# 3. 추천 로직 (부스팅 + 필터링 + 주입)
# ==========================================

def get_best_candidates(category: str, stt_question: str, current_history: List[str]) -> List[str]:
    current_context = " ".join(current_history)
    
    # 1. 프롬프트 생성
    if current_context:
        prompt = f"<usr>{stt_question}<sys>{current_context}"
    else:
        prompt = f"<usr>{stt_question}<sys>"

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 2. GPT 생성 (온도를 높여서 다양성 확보)
    # temperature 1.0 = 완전 창의적 (실수도 많이 함) -> 우리가 필터링할 거니까 괜찮음
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids,
            max_new_tokens=8,
            num_beams=15,             # 빔 개수를 늘려 후보를 많이 확보
            num_return_sequences=15,
            repetition_penalty=1.3,
            do_sample=True,
            temperature=1.0,          # [중요] 창의성 최대화 (복숭아 탈출용)
            top_k=50,
            top_p=0.95,
            eos_token_id=gpt_tokenizer.eos_token_id,
            pad_token_id=gpt_tokenizer.pad_token_id
        )

    # 3. 정답 단어장 로드
    target_vocab = PREFERRED_WORDS.get(category, PREFERRED_WORDS["카페"]) + PREFERRED_WORDS["공통"]

    # 4. 후보 수집 및 1차 필터링
    candidates_pool = set() # 중복 방지용 Set
    
    for output in outputs:
        decoded = gpt_tokenizer.decode(output, skip_special_tokens=False)
        if "<sys>" in decoded:
            generated = decoded.split("<sys>")[1]
            if current_context and current_context in generated:
                generated = generated.replace(current_context, "", 1)
        else:
            generated = decoded

        clean_pattern = r'[\[\]\{\}\(\)<>\"\'\`~;:,.!?]'
        generated = generated.replace("</s>", "").replace("<pad>", "").strip()
        generated = re.sub(clean_pattern, '', generated).strip()
        
        if not generated: continue
        first_word = generated.split(' ')[0]
        
        # 조사로 시작하는 쓰레기 데이터 제거
        is_bad = False
        for bad in BAD_STARTS:
            if first_word.startswith(bad): is_bad = True; break
        if is_bad: continue

        # 너무 긴 단어(5글자 이상)는 노이즈일 확률 높음 (단, 메뉴명은 제외)
        if len(first_word) > 5 and first_word not in target_vocab: continue
        
        candidates_pool.add(first_word)

    # list로 변환
    raw_candidates = list(candidates_pool)

    # =========================================================
    # [치트키 1] 강제 주입 (Injection)
    # =========================================================
    # GPT가 멍청해서 좋은 단어를 하나도 못 뱉었을 경우를 대비해
    # DB에서 상황에 맞는 단어 5개를 랜덤으로 뽑아서 후보에 슬쩍 끼워넣습니다.
    
    # 문장이 짧을 때(초반)만 메뉴 주입
    if len(current_history) < 2: 
        injected = random.sample(target_vocab, k=min(5, len(target_vocab)))
        for item in injected:
            if item not in current_history: # 이미 말한거 제외
                raw_candidates.append(item)

    # =========================================================
    # [치트키 2] BERT 채점 + 점수 조작 (Score Boosting)
    # =========================================================
    scored_candidates = []
    
    with torch.no_grad():
        for cand in raw_candidates:
            # 1. 기본 BERT 점수 계산
            full_answer = f"{current_context} {cand}".strip()
            text = f"{stt_question} [SEP] {full_answer}"
            
            inputs = bert_tokenizer(
                text, return_tensors='pt', truncation=True, max_length=128, padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = bert_model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            base_score = probs[0][1].item() 
            
            # 2. [점수 조작] 우리가 좋아하는 단어면 가산점 부여!
            final_score = base_score
            if cand in target_vocab:
                final_score += 0.3  # 가산점 0.3점 (엄청 큰 점수)
            
            # 3. 커트라인 통과 여부 (가산점 덕분에 메뉴 이름은 쉽게 통과함)
            # 이상한 단어("예약석으로")는 사전에 없으니 가산점 못 받고 탈락
            if final_score >= 0.35: 
                scored_candidates.append((cand, final_score))

    # 점수 순 정렬
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🔹 [{category}] Q: {stt_question} -> Top: {scored_candidates[:5]}")

    # 최종 3개 선정
    final_result = [item[0] for item in scored_candidates[:3]]
    
    # 안전장치 (결과 부족 시 채우기)
    defaults = ["네", "아니요", "감사합니다", "잠시만요"]
    for d in defaults:
        if len(final_result) < 3:
            if d not in final_result: final_result.append(d)
        else: break
            
    return final_result


# ==========================================
# 4. API 실행
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