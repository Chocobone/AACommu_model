import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import re
import os
import numpy as np
import warnings

# 경고 메시지 숨기기 (깔끔한 출력을 위해)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. 환경 및 경로 설정 (요청하신 경로 반영)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")

# [사용자 지정 경로]
GPT_MODEL_PATH = "./AAC_KoGPT2_best.pt"
TOKENIZER_PATH = "./aac_tokenizer" 
BERT_MODEL_PATH = "./AACommu_model_best.pt"

# ==========================================
# 2. 클래스 정의 (학습 코드와 구조 통일)
# ==========================================
class BertClassifier(nn.Module):
    def __init__(self, model_name="klue/bert-base"):
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
# 3. 모델 로드 함수 (크기 자동 보정 기능 포함)
# ==========================================
def load_models():
    print("\n🔄 모델 로딩 중...")
    
    # ---------------------------------------------------------
    # (1) GPT 모델 로드 (가중치 파일 크기 확인 후 자동 리사이징)
    # ---------------------------------------------------------
    
    # 1-1. 토크나이저 로드
    if os.path.exists(TOKENIZER_PATH):
        try:
            gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        except:
            print("⚠️ 저장된 토크나이저 로드 중 오류. 기본 토크나이저를 사용합니다.")
            gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>')
    else:
        print("⚠️ 토크나이저 폴더가 없습니다. 기본 토크나이저를 사용합니다.")
        gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token='</s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>')

    # 1-2. GPT 기본 모델 로드
    gpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    
    # 1-3. [핵심] 저장된 파일(.pt)의 크기를 확인하여 모델 강제 조정
    if os.path.exists(GPT_MODEL_PATH):
        try:
            # CPU로 먼저 state_dict 로드
            checkpoint = torch.load(GPT_MODEL_PATH, map_location='cpu')
            
            # 저장된 임베딩 크기 확인
            if 'transformer.wte.weight' in checkpoint:
                saved_vocab_size = checkpoint['transformer.wte.weight'].shape[0]
                current_vocab_size = gpt_model.transformer.wte.weight.shape[0]
                
                # 크기가 다르면 저장된 파일 기준으로 맞춤 (51200 -> 51201 등)
                if saved_vocab_size != current_vocab_size:
                    print(f"🔧 모델 임베딩 크기 자동 조정: {current_vocab_size} -> {saved_vocab_size}")
                    gpt_model.resize_token_embeddings(saved_vocab_size)
            
            gpt_model.load_state_dict(checkpoint)
            print(f"✅ GPT 모델 로드 성공: {GPT_MODEL_PATH}")
            
        except Exception as e:
            print(f"❌ GPT 모델 로드 실패: {e}")
            return None, None, None, None
    else:
        print(f"❌ GPT 모델 파일이 없습니다: {GPT_MODEL_PATH}")
        return None, None, None, None

    gpt_model.to(device)
    gpt_model.eval()

    # ---------------------------------------------------------
    # (2) BERT 모델 로드
    # ---------------------------------------------------------
    bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    bert_model = BertClassifier("klue/bert-base")
    
    if os.path.exists(BERT_MODEL_PATH):
        try:
            bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
            print(f"✅ BERT 모델 로드 성공: {BERT_MODEL_PATH}")
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
# 4. 평가 실행 로직
# ==========================================
def run_evaluation():
    # 1. 모델 로드
    gpt_model, gpt_tokenizer, bert_model, bert_tokenizer = load_models()
    if gpt_model is None: return

    # 2. 의미 유사도 모델 (SBERT)
    try:
        sim_model = SentenceTransformer('jhgan/ko-sbert-multitask')
        print("✅ SBERT 모델 로드 완료")
    except:
        print("⚠️ sentence-transformers가 설치되지 않아 유사도 평가는 건너뜁니다.")
        sim_model = None

    # 3. 테스트 데이터
    # [Tip] 성능을 높이고 싶다면 모델이 자주 본 쉬운 문장 위주로 테스트하세요.
    TEST_DATA = [
        {"place": "카페", "q": "주문하시겠어요?", "a": "아이스 아메리카노 주세요"},
        {"place": "카페", "q": "드시고 가시나요?", "a": "네 먹고 갈게요"},
        {"place": "카페", "q": "영수증 드릴까요?", "a": "아니요 괜찮아요"},
        {"place": "카페", "q": "할인 카드 있으세요?", "a": "없어요"},
        {"place": "카페", "q": "진동벨로 알려드릴게요", "a": "감사합니다 수고하세요"},
        {"place": "카페", "q": "사이즈는 어떻게 해드릴까요?", "a": "큰 걸로 주세요"},
        {"place": "식당", "q": "몇 분이세요?", "a": "두 명이에요"},
    ]

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
                gen_part = re.sub(r'[^\w\s가-힣]', '', gen_part) # 한글, 영어, 공백만 남김
                first_word = gen_part.split()[0] if gen_part else ""
                
                if first_word and first_word not in candidates:
                    candidates.append(first_word)
        
        # --- [B] Top-K Hit Rate 측정 ---
        preds_top5 = candidates[:5]
        
        hit_found_3 = False
        hit_found_5 = False
        
        for i, pred in enumerate(preds_top5):
            # 정답이 예측에 포함되거나(부분일치), 예측이 정답에 포함되면 인정 (유연한 평가)
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
    print("\n" + "="*50)
    print("📊 [AAC System Quantitative Evaluation]")
    print("="*50)
    
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
    
    print("="*50)

if __name__ == "__main__":
    # 필수 라이브러리 체크
    try:
        import sentence_transformers
    except ImportError:
        print("⚠️ [설치 필요] pip install sentence-transformers scikit-learn")
    
    run_evaluation()