# pip install sentence-transformers scikit-learn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import random
import re
from tools import AACDataProcessor

# --- [설정] ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT_MODEL_PATH = "./aac_kogpt2_dir_tag_model.pt"
BERT_MODEL_PATH = "./aac_bert_model.pt"
TOKENIZER_PATH = "./aac_dir_tag_tokenizer" # 토크나이저 저장 경로
# 테스트용 가상의 정답 데이터셋 (실제로는 load_data 등을 통해 파일에서 불러오세요)
# TEST_DATA = [
#     {"place": "카페", "q": "주문하시겠어요?", "a": "아이스 아메리카노 주세요"},
#     {"place": "카페", "q": "드시고 가시나요?", "a": "네 먹고 갈게요"},
#     {"place": "카페", "q": "할인 카드 있으세요?", "a": "아니요 없어요"},
#     {"place": "카페", "q": "영수증 드릴까요?", "a": "네 주세요"},
#     {"place": "카페", "q": "진동벨로 알려드릴게요", "a": "감사합니다"}
# ]
DIR_CATEGORY_MAP = {
    "TL_01": "카페",   # TL_01... 폴더 안에 있는 건 무조건 <LOC_카페>
    # "TL_02": "식당", # (예시) 나중에 추가 가능
    # "TL_03": "편의점"
}

TEST_DATA_PATH = "/local_datasets/AACommu/Validation/02.라벨링데이터"

processor = AACDataProcessor(TEST_DATA_PATH, DIR_CATEGORY_MAP)
TEST_DATA = processor.load_data()

# ====================================================
# 1. 전략: Top-K Hit Rate (생성 모델 추천 적중률)
# ====================================================
def evaluate_top_k_hit_rate(k=3):
    print(f"\n🚀 [Strategy 1] Top-{k} Hit Rate 평가 시작...")
    
    # 모델 로드 (생성용)
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    hits = 0
    total = len(TEST_DATA)

    for item in tqdm(TEST_DATA, desc="Top-K 평가 중"):
        place = item['place']
        q_text = item['q']
        target_a = item['a']
        target_first_word = target_a.split()[0] # 정답의 첫 어절 (핵심)

        # 입력 포맷 구성
        tag = f"<LOC_{place}>"
        input_text = f"{tag}<usr>{q_text}<sys>"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            # server2.py와 유사하게 빔서치로 다양한 후보 생성
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                num_beams=10,
                num_return_sequences=10,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

        candidates = []
        for out in outputs:
            decoded = tokenizer.decode(out, skip_special_tokens=False)
            if "<sys>" in decoded:
                gen_part = decoded.split("<sys>")[1].replace("</s>", "").strip()
                # 첫 어절 추출
                first_word = gen_part.split()[0] if gen_part else ""
                clean_word = re.sub(r'[^\w]', '', first_word) # 특수문자 제거
                if clean_word and clean_word not in candidates:
                    candidates.append(clean_word)
        
        # 상위 K개 자르기
        top_candidates = candidates[:k]
        
        # 정답 체크: 정답 단어가 후보 리스트에 포함되거나, 후보가 정답에 포함되면 성공
        is_hit = False
        for cand in top_candidates:
            if cand in target_first_word or target_first_word in cand:
                is_hit = True
                break
        
        if is_hit:
            hits += 1
    
    score = (hits / total) * 100
    print(f"📊 Top-{k} Hit Rate: {score:.2f}%")
    return score

# ====================================================
# 2. 전략: BERT Classification Metrics (분류 정확도)
# ====================================================
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

def evaluate_bert_accuracy():
    print(f"\n🚀 [Strategy 2] BERT 적합성 판별 정확도 평가 시작...")
    
    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = BertClassifier().to(device)
    try:
        model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ BERT 모델 로드 실패: {e}")
        return

    # 테스트 데이터 생성 (Positive + Negative)
    inputs = []
    labels = []
    
    for item in TEST_DATA:
        # Positive (정답)
        inputs.append((item['q'], item['a']))
        labels.append(1)
        
        # Negative (랜덤 오답 - 예시로 고정된 오답 사용)
        inputs.append((item['q'], "엉뚱한 대답입니다"))
        labels.append(0)

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for (q, a), label in zip(inputs, labels):
            text = f"{q} [SEP] {a}"
            encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = encoded['input_ids'].to(device)
            mask = encoded['attention_mask'].to(device)
            
            outputs = model(input_ids, mask)
            pred = torch.argmax(outputs, dim=1).item()
            
            preds_list.append(pred)
            labels_list.append(label)

    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list)
    
    print(f"📊 Accuracy: {acc:.4f} (높을수록 좋음)")
    print(f"📊 F1-Score: {f1:.4f}")

# ====================================================
# 3. 전략: Perplexity (PPL) - 언어 모델 성능
# ====================================================
def evaluate_ppl():
    print(f"\n🚀 [Strategy 3] Perplexity(PPL) 평가 시작...")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    nlls = [] # Negative Log Likelihoods

    # PPL은 긴 문장을 얼마나 잘 예측하느냐를 봅니다
    for item in tqdm(TEST_DATA, desc="PPL 계산 중"):
        place = item['place']
        q_text = item['q']
        a_text = item['a']
        
        # 전체 시퀀스: 태그 + 질문 + 답변
        full_text = f"<LOC_{place}><usr>{q_text}<sys>{a_text}</s>"
        
        encodings = tokenizer(full_text, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"📊 Perplexity (PPL): {ppl.item():.2f} (낮을수록 좋음)")

# ====================================================
# 4. 전략: Semantic Similarity (의미 유사도)
# ====================================================
def evaluate_semantic_similarity():
    print(f"\n🚀 [Strategy 4] Semantic Similarity 평가 시작...")
    
    # KoSBERT 모델 로드 (없으면 자동 다운로드됨)
    embedder = SentenceTransformer('jhgan/ko-sbert-multitask')
    
    # 생성 모델 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    similarities = []

    for item in tqdm(TEST_DATA, desc="유사도 측정 중"):
        place = item['place']
        q_text = item['q']
        target_a = item['a']
        
        # 모델이 생성한 답변 1순위 가져오기
        tag = f"<LOC_{place}>"
        input_text = f"{tag}<usr>{q_text}<sys>"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=10, num_beams=1) # Greedy하게 1개만
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            if "<sys>" in generated:
                pred_a = generated.split("<sys>")[1].replace("</s>", "").strip()
            else:
                pred_a = ""
        
        if not pred_a: pred_a = " " # 빈 문자열 방지

        # 코사인 유사도 계산
        emb_target = embedder.encode(target_a, convert_to_tensor=True)
        emb_pred = embedder.encode(pred_a, convert_to_tensor=True)
        
        score = util.pytorch_cos_sim(emb_target, emb_pred).item()
        similarities.append(score)
        
        # print(f"Q: {q_text} | 정답: {target_a} | 예측: {pred_a} | 점수: {score:.4f}")

    avg_sim = sum(similarities) / len(similarities)
    print(f"📊 Average Cosine Similarity: {avg_sim:.4f} (1.0에 가까울수록 좋음)")

# --- [메인 실행] ---
if __name__ == "__main__":
    # 필요한 평가만 주석 해제하여 실행하세요
    evaluate_top_k_hit_rate(k=3)     # 전략 1 (추천)
    evaluate_bert_accuracy()         # 전략 2 (추천 - 수치 높게 나옴)
    evaluate_ppl()                 # 전략 3 (학술적)
    evaluate_semantic_similarity()   # 전략 4 (의미적 일치)