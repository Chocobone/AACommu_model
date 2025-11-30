import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# --- 1. 기본 설정 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "klue/bert-base"
MAX_LEN = 128
BATCH_SIZE = 32  # BERT는 메모리를 많이 먹으므로 16~32 조절 필요
EPOCHS = 3
LR = 2e-5

# --- 2. 데이터 처리 및 Negative Sampling ---
def create_bert_dataset(data_dir):
    data_path = Path(data_dir)
    
    # 1. TL_01 (식당/카페) 디렉토리만 필터링
    target_dirs = [
        "TL_01.식당카페_01.입장_및_이용안내",
        "TL_01.식당카페_02.자리안내",
        "TL_01.식당카페_03.메뉴추천",
        "TL_01.식당카페_04.메뉴주문",
        "TL_01.식당카페_05.식음료서빙",
        "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
    ]
    
    print(f"🎯 학습 대상 디렉토리 ({len(target_dirs)}개):")
    for d in target_dirs:
        print(f"  - {d.name}")

    # 2. JSON 파일 수집 및 파싱
    raw_pairs = []
    json_files = []
    for d in target_dirs:
        json_files.extend(list(d.rglob('*.json')))
        
    print(f"📂 총 {len(json_files)}개의 JSON 파일을 분석합니다.")

    for json_path in tqdm(json_files, desc="데이터 추출 중"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'video' not in data: continue
            
            interactions = data['video'].get('interactions', [])
            for interaction in interactions:
                # 손님 말 (정답/Human)
                h_txt = ""
                if 'human_event' in interaction:
                    utts = interaction['human_event'].get('utterances', [])
                    if utts: h_txt = utts[0].get('utterance_cap', '').strip()

                # 직원 말 (질문/Robot)
                r_txt = ""
                if 'robot_response' in interaction:
                    resps = interaction['robot_response']
                    if resps: r_txt = resps[0].get('answer', '').strip()
                
                # 데이터 유효성 검사
                if h_txt and r_txt:
                    raw_pairs.append({'q': r_txt, 'a': h_txt})
                    
        except Exception as e:
            continue

    print(f"✅ 원본 대화 쌍 {len(raw_pairs)}개 추출 완료.")
    if not raw_pairs: return pd.DataFrame()

    # 3. Negative Sampling (정답 1개 + 오답 1개 생성)
    processed_data = []
    all_answers = [p['a'] for p in raw_pairs] # 랜덤 추출용 전체 답변 리스트
    
    print("데이터셋 생성 (Positive + Negative) 진행 중...")
    
    for p in raw_pairs:
        # (1) Positive Sample (Label 1) : 진짜 문맥
        processed_data.append({
            'text_a': p['q'], 
            'text_b': p['a'], 
            'label': 1
        })
        
        # (2) Negative Sample (Label 0) : 가짜 문맥
        # 전체 답변 중 랜덤하게 하나 선택 (현재 정답이 아닌 것)
        while True:
            random_a = random.choice(all_answers)
            if random_a != p['a']:
                break
        
        processed_data.append({
            'text_a': p['q'], 
            'text_b': random_a, 
            'label': 0
        })
        
    print(f"최종 학습 데이터 크기: {len(processed_data)} (Positive:Negative = 1:1)")
    return pd.DataFrame(processed_data)

# --- 3. BERT 모델 정의 ---
class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        # 이진 분류 (0:부적절, 1:적절)
        self.out = nn.Linear(self.bert.config.hidden_size, 2) 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output: [CLS] 토큰의 임베딩 (문장 전체 의미)
        pooled_output = outputs.pooler_output 
        output = self.drop(pooled_output)
        return self.out(output)

class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # BERT 입력 포맷: [CLS] 질문 [SEP] 답변 [SEP]
        text = str(row['text_a']) + " [SEP] " + str(row['text_b'])
        label = row['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False, # BERT 기본 모델 사용 시 보통 불필요 (필요시 True)
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. 메인 실행 ---
def main():
    # 데이터 경로 (사용자 환경)
    data_dir = "/local_datasets/AACommu/Training/02.라벨링데이터"
    save_path = "./aac_bert_model.pt"

    if not os.path.exists(data_dir):
        print(f"❌ 경로 오류: {data_dir} 를 찾을 수 없습니다.")
        return

    # 1. 데이터 로드
    df = create_bert_dataset(data_dir)
    if df.empty:
        print("❌ 학습할 데이터가 없습니다.")
        return

    # 2. 토크나이저 & 모델 준비
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertClassifier(MODEL_NAME).to(device)
    
    dataset = BertDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 3. 학습 루프
    print("\n🚀 BERT 학습 시작 (적절성 평가 모델)...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(loader)
        acc = correct_predictions.double() / len(df)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 4. 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 모델 저장 완료: {save_path}")

    # 5. 추론 테스트
    def predict_appropriateness(q, a):
        model.eval()
        text = q + " [SEP] " + a
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            score = probs[0][1].item() # 1(적절)일 확률
            
        return score

    print("\n--- [TEST] ---")
    q_test = "아이스 아메리카노 나오셨습니다."
    a_good = "네, 감사합니다. 빨대는 어디있나요?"
    a_bad = "어서오세요 몇분이세요?"
    
    print(f"Q: {q_test}")
    print(f"A(적절): {a_good} -> 점수: {predict_appropriateness(q_test, a_good):.4f}")
    print(f"A(부적절): {a_bad} -> 점수: {predict_appropriateness(q_test, a_bad):.4f}")

if __name__ == "__main__":
    main()