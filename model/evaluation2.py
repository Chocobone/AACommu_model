import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import re
import os
import random  # [추가] 랜덤 샘플링용
import numpy as np
from tools import AACDataProcessor

# ==========================================
# 1. 환경 설정
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")

GPT_MODEL_PATH = "./AAC_KoGPT2_best.pt"
TOKENIZER_PATH = "./aac_tokenizer" 
BERT_MODEL_PATH = "./AACommu_model_best.pt"

TEST_DATA_PATH = "/local_datasets/AACommu/Validation/02.라벨링데이터"

DIR_CATEGORY_MAP = {
    "VL_01": "카페",
    # "VL_02": "식당", # 필요시 주석 해제
}

# 데이터 로드
processor = AACDataProcessor(TEST_DATA_PATH, DIR_CATEGORY_MAP)
df_test = processor.load_data()

if df_test.empty:
    print("\n⚠️ [경고] 로드된 데이터가 없습니다. 경로를 확인하거나 tools.py의 검색 로직을 확인하세요.")
    TEST_DATA = []
    ALL_ANSWERS = []
else:
    TEST_DATA = df_test.to_dict('records')
    # [수정] BERT 오답 생성을 위해 전체 정답 리스트 미리 추출
    ALL_ANSWERS = [item['a'] for item in TEST_DATA if item['a']]
    print(f"✅ 총 {len(TEST_DATA)}개의 평가 데이터를 로드했습니다.")

# ==========================================
# 2. 클래스 정의
# ==========================================
class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# ==========================================
# 3. 모델 로드 함수
# ==========================================
def load_models():
    print("\n🔄 모델 로딩 중...")
    
    # --- (1) GPT 로드 ---
    if os.path.exists(TOKENIZER_PATH):
        gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    else:
        print("⚠️ 저장된 토크나이저 폴더가 없습니다. 기본 모델을 기반으로 복구합니다.")
        gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')
        
        special_tokens = [f"<LOC_{loc}>" for loc in DIR_CATEGORY_MAP.values()]
        if "<LOC_기타>" not in special_tokens:
            special_tokens.append("<LOC_기타>")
        gpt_tokenizer.add_tokens(special_tokens)

    gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    
    # 임베딩 크기 조정
    target_vocab_size = 51201 
    if len(gpt_tokenizer) != target_vocab_size:
        print(f"ℹ️ 토크나이저({len(gpt_tokenizer)})와 타겟({target_vocab_size}) 불일치 -> 강제 조정")
    
    gpt_model.resize_token_embeddings(target_vocab_size)
    
    if os.path.exists(GPT_MODEL_PATH):
        try:
            state_dict = torch.load(GPT_MODEL_PATH, map_location=device)
            gpt_model.load_state_dict(state_dict)
            print("✅ GPT 모델 로드 성공")
        except RuntimeError as e:
            print(f"❌ GPT 모델 로드 실패: {e}")
            return None, None, None, None
    else:
        print(f"❌ GPT 모델 파일이 없습니다: {GPT_MODEL_PATH}")
        return None, None, None, None

    gpt_model.to(device)
    gpt_model.eval()

    # --- (2) BERT 로드 ---
    bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    bert_model = BertClassifier("klue/bert-base")
    
    if os.path.exists(BERT_MODEL_PATH):
        try:
            bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
            print("✅ BERT 모델 로드 성공")
        except Exception as e:
            print(f"❌ BERT 모델 로드 실패: {e}")
            bert_model = None
    else:
        print(f"⚠️ BERT 모델 파일이 없습니다: {BERT_MODEL_PATH}")
        bert_model = None
    
    if bert_model:
        bert_model.to(device)
        bert_model.eval()

    return gpt_model, gpt_tokenizer, bert_model, bert_tokenizer

# ==========================================
# 4. 평가 실행
# ==========================================
def run_evaluation():
    if not TEST_DATA:
        print("❌ 평가할 데이터가 없습니다. 종료합니다.")
        return

    gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()
    if gpt_model is None: return

    try:
        sim_model = SentenceTransformer('jhgan/ko-sbert-multitask')
    except:
        print("⚠️ sentence-transformers 로드 실패. 유사도 평가 건너뜀.")
        sim_model = None

    print("\n🚀 평가 시작...\n")

    top3_hits = 0
    top5_hits = 0
    sim_scores = []
    
    bert_preds = []
    bert_labels = []

    # [디버깅] 처음 5개 샘플의 생성 결과 출력을 위한 카운터
    debug_count = 0 
    debug_limit = 5 

    for item in tqdm(TEST_DATA, desc="Processing"):
        tag = item['place_tag']  
        q_text = item['q']
        target_a = item['a']
        
        # --- [A] GPT 후보 생성 ---
        input_text = f"{tag}<usr>{q_text}<sys>"
        input_ids = gpt_tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = gpt_model.generate(
                input_ids,
                max_new_tokens=20,     # [수정] 토큰 길이 조금 늘림
                num_beams=5,           # [수정] Beam Search 약간 축소 (속도 향상)
                num_return_sequences=5,
                repetition_penalty=1.5, # [수정] 페널티 약간 완화 (너무 높으면 말문 막힘)
                pad_token_id=gpt_tokenizer.pad_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
                early_stopping=True
            )

        candidates = []
        for out in outputs:
            decoded = gpt_tokenizer.decode(out, skip_special_tokens=False)
            if "<sys>" in decoded:
                gen_part = decoded.split("<sys>")[1].replace("</s>", "").strip()
                # 특수문자 일부 제거하되, 문장 형태는 유지
                gen_part = re.sub(r'[^\w\s가-힣]', ' ', gen_part).strip()
                if gen_part and gen_part not in candidates:
                    candidates.append(gen_part)
        
        # [디버깅 출력] 생성된 문장이 무엇인지 확인
        if debug_count < debug_limit:
            print(f"\n[Debug {debug_count+1}]")
            print(f"  Q   : {q_text}")
            print(f"  Ans : {target_a}")
            print(f"  Gen : {candidates[0] if candidates else 'FAILED'}")
            debug_count += 1

        # --- [B] Top-K Hit Rate (개선됨) ---
        preds_top5 = candidates[:5]
        hit_found_3 = False
        hit_found_5 = False
        
        for i, pred in enumerate(preds_top5):
            # [수정] 단순히 첫 단어 일치가 아니라, 정답이 생성 문장에 포함되거나 그 반대의 경우 체크
            # 공백 제거 후 비교 (띄어쓰기 문제 완화)
            pred_clean = pred.replace(" ", "")
            target_clean = target_a.replace(" ", "")
            
            # 정답이 너무 짧은 경우(1글자) 제외하고 포함 관계 확인
            is_match = (target_clean in pred_clean) or (pred_clean in target_clean)
            
            if is_match:
                if i < 3: hit_found_3 = True
                if i < 5: hit_found_5 = True
        
        if hit_found_3: top3_hits += 1
        if hit_found_5: top5_hits += 1

        # --- [C] Semantic Similarity ---
        if sim_model:
            best_pred = preds_top5[0] if preds_top5 else ""
            emb1 = sim_model.encode(target_a, convert_to_tensor=True)
            emb2 = sim_model.encode(best_pred, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item()
            sim_scores.append(score)

        # --- [D] BERT Accuracy (수정됨: Random Negative Sampling) ---
        if bert_model and ALL_ANSWERS:
            # (1) 정답 데이터
            text_pos = f"{q_text} [SEP] {target_a}"
            enc_pos = bert_tokenizer(text_pos, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
            with torch.no_grad():
                out_pos = bert_model(enc_pos['input_ids'], enc_pos['attention_mask'])
                bert_preds.append(torch.argmax(out_pos, dim=1).item())
                bert_labels.append(1)

            # (2) 오답 데이터 (랜덤 추출)
            # 현재 정답과 다른 답변을 찾을 때까지 랜덤 선택
            while True:
                random_wrong = random.choice(ALL_ANSWERS)
                if random_wrong != target_a:
                    break
            
            text_neg = f"{q_text} [SEP] {random_wrong}"
            enc_neg = bert_tokenizer(text_neg, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
            with torch.no_grad():
                out_neg = bert_model(enc_neg['input_ids'], enc_neg['attention_mask'])
                bert_preds.append(torch.argmax(out_neg, dim=1).item())
                bert_labels.append(0)

    # ==========================================
    # 5. 결과 출력
    # ==========================================
    print("\n" + "="*40)
    print("📊 [AAC System Quantitative Evaluation]")
    print("="*40)
    
    total = len(TEST_DATA)
    if total > 0:
        print(f"1️⃣  Top-3 Hit Rate : {top3_hits / total * 100:.2f}%")
        print(f"    Top-5 Hit Rate : {top5_hits / total * 100:.2f}%")
    
    if sim_scores:
        print(f"2️⃣  Semantic Similarity : {sum(sim_scores)/len(sim_scores):.4f}")
    
    if bert_preds:
        print(f"3️⃣  BERT Accuracy       : {accuracy_score(bert_labels, bert_preds)*100:.2f}%")
        print(f"    BERT F1-Score       : {f1_score(bert_labels, bert_preds):.4f}")
    
    print("="*40)

if __name__ == "__main__":
    run_evaluation()