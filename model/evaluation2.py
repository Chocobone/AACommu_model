import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import re

# ==========================================
# 0. 설정 및 모델 정의 (server2.py와 동일하게)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일 경로 (사용자 환경에 맞게 수정)
GPT_MODEL_PATH = "./AAC_KoGPT2_best.pt"
TOKENIZER_PATH = "./aac_tokenizer"
BERT_MODEL_PATH = "./AACommu_model_best.pt"

# 테스트용 데이터 (실제 평가 시 더 많이 추가하세요)
TEST_DATA = [
    {"category": "카페", "q": "주문하시겠어요?", "a": "아이스 아메리카노 주세요"},
    {"category": "카페", "q": "드시고 가시나요?", "a": "네 먹고 갈게요"},
    {"category": "카페", "q": "할인 카드 있으세요?", "a": "아니요 없어요"},
    {"category": "카페", "q": "영수증 드릴까요?", "a": "네 주세요"},
    {"category": "카페", "q": "진동벨로 알려드릴게요", "a": "감사합니다"},
    {"category": "식당", "q": "몇 분이세요?", "a": "두 명이에요"},
]

# BERT 모델 클래스 정의 (server2.py와 일치해야 로드 가능)
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

# ==========================================
# 1. 모델 로드
# ==========================================
print("🔄 모델 로드 중...")

# (1) GPT 로드
try:
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
except:
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
gpt_model.resize_token_embeddings(len(gpt_tokenizer))
if torch.cuda.is_available():
    gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
else:
    gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=torch.device('cpu')))
gpt_model.to(device).eval()

# (2) BERT 로드
bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
bert_model = BertClassifier().to(device)
if torch.cuda.is_available():
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
else:
    bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=torch.device('cpu')))
bert_model.eval()

# (3) 의미 유사도 모델 (KoSBERT)
print("📥 의미 유사도 모델 다운로드 중... (최초 1회)")
similarity_model = SentenceTransformer('jhgan/ko-sbert-multitask')

# ==========================================
# 2. 평가 함수 구현
# ==========================================

def get_candidates(category, question):
    """GPT 모델로 후보 생성 (server2.py 로직 간소화)"""
    prompt = f"<LOC_{category}><usr>{question}<sys>"
    input_ids = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = gpt_model.generate(
            input_ids,
            max_new_tokens=10,
            num_beams=10,        # Beam Search로 다양한 후보 탐색
            num_return_sequences=5,
            early_stopping=True,
            pad_token_id=gpt_tokenizer.pad_token_id,
            eos_token_id=gpt_tokenizer.eos_token_id
        )
    
    candidates = []
    for out in outputs:
        decoded = gpt_tokenizer.decode(out, skip_special_tokens=False)
        if "<sys>" in decoded:
            ans = decoded.split("<sys>")[1].replace("</s>", "").strip()
            # 첫 어절만 추출 (AAC 버튼용)
            first_word = ans.split()[0] if ans else ""
            clean_word = re.sub(r'[^\w]', '', first_word) # 특수문자 제거
            if clean_word and clean_word not in candidates:
                candidates.append(clean_word)
    return candidates[:5] # 상위 5개 반환

# ==========================================
# 3. 메인 평가 루프
# ==========================================
print("\n🚀 평가 시작...\n")

# 지표 집계용 변수
top3_hits = 0
top5_hits = 0
similarity_scores = []
bert_correct = 0
bert_total = 0

for i, item in enumerate(tqdm(TEST_DATA)):
    category = item['category']
    q = item['q']
    true_a = item['a']
    true_word = true_a.split()[0] # 정답의 첫 어절 (핵심)

    # 1. GPT 후보 생성
    preds = get_candidates(category, q)
    
    # [지표 1] Top-K Hit Rate 계산
    # 정답 단어가 추천 목록에 포함되어 있거나, 포함 관계인 경우
    is_hit_3 = False
    is_hit_5 = False
    
    for idx, pred in enumerate(preds):
        # '네' vs '네(먹고갈게요)' 처럼 포함되면 정답 처리 (유연한 평가)
        if (pred in true_word) or (true_word in pred):
            if idx < 3: is_hit_3 = True
            if idx < 5: is_hit_5 = True
    
    if is_hit_3: top3_hits += 1
    if is_hit_5: top5_hits += 1

    # [지표 2] Semantic Similarity (의미 유사도)
    # 모델이 내뱉은 1순위 예측값 vs 정답 전체 문장 유사도
    best_pred = preds[0] if preds else ""
    sim_score = util.pytorch_cos_sim(
        similarity_model.encode(true_word), 
        similarity_model.encode(best_pred)
    ).item()
    similarity_scores.append(sim_score)

    # [지표 3] BERT 분류 정확도 (Re-ranking 성능)
    # 정답(Positive)과 오답(Negative)을 잘 구분하는지 테스트
    # Case A: 정답을 넣었을 때 1(적절)이 나와야 함
    text_pos = f"{q} [SEP] {true_a}"
    inputs = bert_tokenizer(text_pos, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
    with torch.no_grad():
        out = bert_model(inputs['input_ids'], inputs['attention_mask'])
        prob_pos = torch.softmax(out, dim=1)[0][1].item()
    
    # Case B: 엉뚱한 답을 넣었을 때 0(부적절)이 나와야 함
    text_neg = f"{q} [SEP] 엉뚱한소리"
    inputs = bert_tokenizer(text_neg, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
    with torch.no_grad():
        out = bert_model(inputs['input_ids'], inputs['attention_mask'])
        prob_neg = torch.softmax(out, dim=1)[0][1].item()

    if prob_pos > 0.5: bert_correct += 1 # 정답 맞춤
    bert_total += 1
    if prob_neg < 0.5: bert_correct += 1 # 오답 걸러냄
    bert_total += 1

# ==========================================
# 4. 결과 출력
# ==========================================
print("\n" + "="*30)
print("📊 최종 성능 평가 결과")
print("="*30)

# 1. Top-K Hit Rate
score_top3 = top3_hits / len(TEST_DATA) * 100
score_top5 = top5_hits / len(TEST_DATA) * 100
print(f"✅ [사용성] Top-3 Hit Rate: {score_top3:.1f}%  <-- (목표: 50% 이상)")
print(f"✅ [사용성] Top-5 Hit Rate: {score_top5:.1f}%  <-- (보조 지표)")

# 2. Semantic Similarity
avg_sim = sum(similarity_scores) / len(similarity_scores)
print(f"✅ [유연성] 의미 유사도    : {avg_sim:.3f}   <-- (목표: 0.75 이상, 1.0 만점)")

# 3. BERT Accuracy
bert_acc = bert_correct / bert_total * 100
print(f"✅ [신뢰성] 필터링 정확도  : {bert_acc:.1f}%   <-- (목표: 90% 이상)")
print("="*30)