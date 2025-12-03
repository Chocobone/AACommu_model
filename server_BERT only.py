# server.py
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
# 1. 설정 및 클래스 정의
# ==========================================
print("🚀 AAC AI 서버 초기화 중... (GPU 메모리 확인)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   - 사용 장치: {device}")

# BERT 모델 클래스 (학습 코드와 동일한 구조여야 함)
class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

# ==========================================
# 2. 모델 및 토크나이저 로드
# ==========================================

# --- (A) KoGPT2 (생성 모델) 로드 ---
GPT_MODEL_PATH = "./aac_kogpt2_model.pt"
TOKENIZER_PATH = "./aac_tokenizer"

if os.path.exists(TOKENIZER_PATH):
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    print("   ✅ GPT 토크나이저 로드 완료")
else:
    print("   ⚠️ [주의] 저장된 토크나이저가 없어 기본 모델을 사용합니다.")
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')

gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
gpt_model.resize_token_embeddings(len(gpt_tokenizer))

if os.path.exists(GPT_MODEL_PATH):
    # map_location: GPU 학습 모델을 CPU 서버에서도 돌릴 수 있게 함
    gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
    print("   ✅ GPT 가중치 로드 완료")
else:
    print("   ⚠️ GPT 가중치 파일이 없습니다! 기본 모델로 동작합니다.")

gpt_model.to(device)
gpt_model.eval()

# --- (B) BERT (검증 모델) 로드 ---
BERT_MODEL_PATH = "./aac_bert_model.pt"
bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
bert_model = BertClassifier()

if os.path.exists(BERT_MODEL_PATH):
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    print("   ✅ BERT 가중치 로드 완료")
else:
    print("   ⚠️ BERT 가중치 파일이 없습니다! 검증 기능이 제한됩니다.")

bert_model.to(device)
bert_model.eval()

print("🏁 시스템 준비 완료!\n")


# ==========================================
# 3. 데이터 및 유틸리티
# ==========================================

# [메뉴 주입] AI가 생성하지 못하더라도 후보에 강제로 섞어줄 단어들
COMMON_CAFE_MENUS = [
    "아메리카노", "아이스 아메리카노", "라떼", "바닐라라떼", "카페모카", "카푸치노", 
    "초코라떼", "녹차라떼", "아이스티", "복숭아 아이스티", "레몬에이드", "자몽에이드",
    "스무디", "요거트 스무디", "망고 스무디", "딸기 스무디",
    "케이크", "치즈케이크", "베이글", "허니브레드", "마카롱", "샌드위치",
    "휘핑", "얼음", "시럽", "사이즈", "샷", "테이크아웃", "매장"
]

# 불량 토큰 필터링 리스트
BAD_STARTS = ["을", "를", "이", "가", "은", "는", "로", "에", "서", "고", "지", "만", "요", "도", "의"]
VALID_SINGLES = ["네", "물", "컵", "약", "밥", "면", "국", "돈", "핫", "아", "예", "좀"]

# ==========================================
# 4. 핵심 추천 로직 (Generate -> Inject -> Filter -> Rank)
# ==========================================

def get_best_candidates(stt_question: str, current_history: List[str]) -> List[str]:
    # 1. 문맥 정리
    current_context = " ".join(current_history)
    
    # 프롬프트 생성 (<usr>질문<sys>답변)
    if current_context:
        prompt = f"<usr>{stt_question}<sys>{current_context}"
    else:
        prompt = f"<usr>{stt_question}<sys>"

    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 2. GPT 생성 (다양성 확보를 위한 파라미터 튜닝)
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids,
            max_new_tokens=8,        # 단어 파편화를 막기 위해 넉넉히
            num_beams=10,            # 빔 서치
            num_return_sequences=10, # 후보 10개 생성
            repetition_penalty=1.5,  # 적당한 반복 억제
            do_sample=True,          # 샘플링 활성화
            temperature=0.9,         # 창의성 높임 (다양한 단어 시도)
            top_k=50,
            top_p=0.92,
            eos_token_id=gpt_tokenizer.eos_token_id,
            pad_token_id=gpt_tokenizer.pad_token_id
        )

    # 3. 1차 후보 수집 및 필터링
    raw_candidates = []
    
    for output in outputs:
        decoded = gpt_tokenizer.decode(output, skip_special_tokens=False)
        
        # <sys> 태그 뒤의 내용만 추출
        if "<sys>" in decoded:
            generated = decoded.split("<sys>")[1]
            if current_context and current_context in generated:
                generated = generated.replace(current_context, "", 1)
        else:
            generated = decoded

        # 특수문자 제거
        clean_pattern = r'[\[\]\{\}\(\)<>\"\'\`~;:,.!?]'
        generated = generated.replace("</s>", "").replace("<pad>", "").strip()
        generated = re.sub(clean_pattern, '', generated).strip()
        
        if not generated: continue

        # 첫 어절만 추출 (단어 단위 추천)
        first_word = generated.split(' ')[0]
        
        # (필터링 1) 한 글자인데 의미 없는 말 제거
        if len(first_word) == 1 and first_word not in VALID_SINGLES: continue
        
        # (필터링 2) 조사로 시작하는 말 제거
        is_bad_start = False
        for bad in BAD_STARTS:
            if first_word.startswith(bad):
                is_bad_start = True
                break
        if is_bad_start: continue
            
        if first_word and first_word not in raw_candidates:
            raw_candidates.append(first_word)

    # 4. [메뉴 주입] 문맥이 짧을 때(초반) 메뉴 리스트 강제 주입
    # -> AI가 복숭아만 말하는 것 방지
    if len(current_history) < 3:
        # 메뉴 리스트에서 랜덤하게 5개 뽑아서 후보군에 추가
        injected_items = random.sample(COMMON_CAFE_MENUS, k=5)
        for item in injected_items:
            if item not in raw_candidates and item not in current_history:
                raw_candidates.append(item)

    # BERT 연산 부하 방지를 위해 최대 20개까지만 검사
    if len(raw_candidates) > 20:
        raw_candidates = raw_candidates[:20]

    # 5. BERT 채점 (Re-ranking)
    scored_candidates = []
    
    if not raw_candidates:
        # 후보가 아예 없으면 기본값 리턴
        return ["네", "아니요", "주문할게요"]

    with torch.no_grad():
        for cand in raw_candidates:
            # 질문 + (현재까지만든문장 + 후보단어) 조합의 적절성 평가
            full_answer = f"{current_context} {cand}".strip()
            text = f"{stt_question} [SEP] {full_answer}"
            
            inputs = bert_tokenizer(
                text, return_tensors='pt', truncation=True, max_length=128, padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = bert_model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            score = probs[0][1].item() # Label 1 (적절함) 확률
            
            # BERT 점수 커트라인 (0.25점 이상만 통과)
            if score >= 0.25:
                scored_candidates.append((cand, score))

    # 점수 높은 순 정렬
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 로그 출력 (디버깅용)
    print(f"🔹 Q: {stt_question} / History: {current_history}")
    print(f"🔹 Top Candidates: {scored_candidates[:5]}")

    # 상위 3개 추출
    final_result = [item[0] for item in scored_candidates[:3]]
    
    # 6. 안전장치 (결과가 3개 미만일 때 기본값 채우기)
    defaults = ["네", "아니요", "주문할게요", "잠시만요", "감사합니다"]
    for d in defaults:
        if len(final_result) < 3:
            if d not in final_result:
                final_result.append(d)
        else:
            break
            
    return final_result


# ==========================================
# 5. API 엔드포인트
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

# 실행 명령: uvicorn server:app --reload