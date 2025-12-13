import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import re
import os
import numpy as np
from tools import AACDataProcessor

# ==========================================
# 1. 환경 설정 (server.py 경로 기준)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")

# 파일 경로 (학습 코드에서 지정한 경로와 일치해야 함)
GPT_MODEL_PATH = "./AAC_KoGPT2_best.pt"
TOKENIZER_PATH = "./aac_tokenizer" 
BERT_MODEL_PATH = "./AACommu_model_best.pt"

TEST_DATA_PATH = "/local_datasets/AACommu/Validation/02.라벨링데이터"

processor = AACDataProcessor(TEST_DATA_PATH, DIR_CATEGORY_MAP)
TEST_DATA = processor.load_data()

# ==========================================
# 2. 클래스 정의 (AAC_BERT.py와 100% 일치시킴)
# ==========================================
class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2) # 0, 1

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
        # tokenizer.json 등이 들어있는 폴더 경로 로드
        gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    else:
        print("⚠️ 저장된 토크나이저 폴더가 없습니다. 기본 모델을 로드합니다.")
        gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')

    gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    # [핵심] 학습 때 토큰을 추가했으므로 임베딩 크기 조절 필수
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))
    
    if os.path.exists(GPT_MODEL_PATH):
        try:
            gpt_model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
            print("✅ GPT 모델 로드 성공")
        except Exception as e:
            print(f"❌ GPT 모델 로드 실패 (크기 불일치 등): {e}")
            return None, None, None, None
    else:
        print(f"❌ GPT 모델 파일이 없습니다: {GPT_MODEL_PATH}")
        return None, None, None, None

    gpt_model.to(device)
    gpt_model.eval()

    # --- (2) BERT 로드 ---
    bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    # [핵심] 초기화 시 model_name 인자 전달 (AAC_BERT.py와 동일하게)
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
    # 1. 모델 로드
    gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()
    if gpt_model is None: return

    # 2. 유사도 모델 (SBERT)
    try:
        sim_model = SentenceTransformer('jhgan/ko-sbert-multitask')
    except:
        print("⚠️ sentence-transformers가 설치되지 않아 유사도 평가는 건너뜁니다.")
        sim_model = None

    # 3. 테스트 데이터 (원하는 만큼 추가하세요)
    # 실제 학습에 쓰지 않은 문장들 위주로 구성하면 좋습니다.
    # TEST_DATA = [
    #     {"place": "카페", "q": "주문하시겠어요?", "a": "아이스 아메리카노 주세요"},
    #     {"place": "카페", "q": "드시고 가시나요?", "a": "네 먹고 갈게요"},
    #     {"place": "카페", "q": "영수증 드릴까요?", "a": "아니요 괜찮아요"},
    #     {"place": "카페", "q": "할인 카드 있으세요?", "a": "없어요"},
    #     {"place": "카페", "q": "진동벨로 알려드릴게요", "a": "감사합니다 수고하세요"},
    #     {"place": "카페", "q": "사이즈는 어떻게 해드릴까요?", "a": "큰 걸로 주세요"},
    # ]

    print("\n🚀 평가 시작...\n")

    top3_hits = 0
    top5_hits = 0
    sim_scores = []
    
    bert_preds = []
    bert_labels = []

    for item in tqdm(TEST_DATA, desc="Processing"):
        place = item['place']
        q_text = item['q']
        target_a = item['a']
        target_first_word = target_a.split()[0] # 정답의 첫 어절

        # --- [A] GPT 후보 생성 ---
        tag = f"<LOC_{place}>"
        input_text = f"{tag}<usr>{q_text}<sys>"
        input_ids = gpt_tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = gpt_model.generate(
                input_ids,
                max_new_tokens=10,
                num_beams=10,
                num_return_sequences=10,
                repetition_penalty=2.0,
                pad_token_id=gpt_tokenizer.pad_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
                early_stopping=True
            )

        candidates = []
        for out in outputs:
            decoded = gpt_tokenizer.decode(out, skip_special_tokens=False)
            if "<sys>" in decoded:
                gen_part = decoded.split("<sys>")[1].replace("</s>", "").strip()
                # 특수문자 제거
                gen_part = re.sub(r'[^\w\s]', '', gen_part)
                first_word = gen_part.split()[0] if gen_part else ""
                
                if first_word and first_word not in candidates:
                    candidates.append(first_word)
        
        # --- [B] Top-K Hit Rate 측정 ---
        preds_top5 = candidates[:5]
        
        hit_found_3 = False
        hit_found_5 = False
        
        for i, pred in enumerate(preds_top5):
            # 정답이 예측에 포함되거나(부분일치), 예측이 정답에 포함되면 인정
            if (target_first_word in pred) or (pred in target_first_word):
                if i < 3: hit_found_3 = True
                if i < 5: hit_found_5 = True
        
        if hit_found_3: top3_hits += 1
        if hit_found_5: top5_hits += 1

        # --- [C] Semantic Similarity 측정 ---
        if sim_model:
            best_pred = preds_top5[0] if preds_top5 else ""
            emb1 = sim_model.encode(target_a, convert_to_tensor=True)
            emb2 = sim_model.encode(best_pred, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item()
            sim_scores.append(score)

        # --- [D] BERT 분류 정확도 측정 ---
        if bert_model:
            # 1. 정답 쌍 (Label 1)
            text_pos = f"{q_text} [SEP] {target_a}"
            enc_pos = bert_tokenizer(text_pos, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
            with torch.no_grad():
                out_pos = bert_model(enc_pos['input_ids'], enc_pos['attention_mask'])
                pred_pos = torch.argmax(out_pos, dim=1).item()
                bert_preds.append(pred_pos)
                bert_labels.append(1)

            # 2. 오답 쌍 (Label 0)
            text_neg = f"{q_text} [SEP] 엉뚱한소리"
            enc_neg = bert_tokenizer(text_neg, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
            with torch.no_grad():
                out_neg = bert_model(enc_neg['input_ids'], enc_neg['attention_mask'])
                pred_neg = torch.argmax(out_neg, dim=1).item()
                bert_preds.append(pred_neg)
                bert_labels.append(0)

    # ==========================================
    # 5. 최종 결과 출력
    # ==========================================
    print("\n" + "="*40)
    print("📊 [AAC System Quantitative Evaluation]")
    print("="*40)
    
    # 1. Top-K Hit Rate
    if len(TEST_DATA) > 0:
        acc3 = top3_hits / len(TEST_DATA) * 100
        acc5 = top5_hits / len(TEST_DATA) * 100
        print(f"1️⃣  Top-3 Hit Rate (Recall@3) : {acc3:.2f}%")
        print(f"    Top-5 Hit Rate (Recall@5) : {acc5:.2f}%")
    
    # 2. Semantic Similarity
    if sim_scores:
        avg_sim = sum(sim_scores) / len(sim_scores)
        print(f"2️⃣  Semantic Similarity       : {avg_sim:.4f} (Max 1.0)")
    
    # 3. BERT Accuracy
    if bert_preds:
        bert_acc = accuracy_score(bert_labels, bert_preds)
        bert_f1 = f1_score(bert_labels, bert_preds)
        print(f"3️⃣  BERT Re-ranking Accuracy  : {bert_acc*100:.2f}%")
        print(f"    BERT F1-Score             : {bert_f1:.4f}")
    else:
        print("3️⃣  BERT 모델이 없어 평가하지 않음")
    
    print("="*40)

if __name__ == "__main__":
    # 라이브러리 설치 안내
    try:
        import sentence_transformers
    except ImportError:
        print("⚠️ 필요 라이브러리 설치: pip install sentence-transformers scikit-learn")
    
    run_evaluation()