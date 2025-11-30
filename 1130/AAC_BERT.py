import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
import json
import random
from pathlib import Path
from tqdm import tqdm
import os

# --- 1. 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# --- 2. 데이터 처리 (Negative Sampling 포함) ---
# KoGPT2와 동일한 파싱 로직이지만, 오답 데이터(Negative Sample)를 생성합니다.
def create_bert_dataset(data_dir):
    data_dir = Path(data_dir)
    pairs = []
    
    # 1. 정답 쌍 추출
    if not os.path.exists(data_dir):
        # 더미 데이터
        raw_pairs = [{'q': '어서오세요', 'a': '안녕하세요'}, {'q': '주문하시겠어요?', 'a': '잠시만요'}]
    else:
        raw_pairs = []
        for json_path in list(data_dir.rglob('*.json'))[:100]: # 테스트용으로 100개만 (전체하려면 [:100] 제거)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                interactions = data.get('video', {}).get('interactions', [])
                for interaction in interactions:
                    h_txt = interaction.get('human_event', {}).get('utterances', [{}])[0].get('utterance_cap', '').strip()
                    r_txt = interaction.get('robot_response', [{}])[0].get('answer', '').strip()
                    if h_txt and r_txt:
                        # AAC 상황: Input=Robot, Output=Human
                        raw_pairs.append({'q': r_txt, 'a': h_txt})
            except: continue

    if not raw_pairs: return pd.DataFrame()

    # 2. Positive(정답) & Negative(오답) 데이터 생성
    processed_data = []
    all_answers = [p['a'] for p in raw_pairs]
    
    print("데이터셋 생성 및 Negative Sampling 진행 중...")
    for p in raw_pairs:
        # (1) Positive: Label 1
        processed_data.append({'text_a': p['q'], 'text_b': p['a'], 'label': 1})
        
        # (2) Negative: Label 0 (다른 답변 랜덤 매칭)
        neg_a = random.choice(all_answers)
        while neg_a == p['a']: # 정답과 같으면 다시 뽑기
            neg_a = random.choice(all_answers)
        processed_data.append({'text_a': p['q'], 'text_b': neg_a, 'label': 0})
        
    return pd.DataFrame(processed_data)

# --- 3. BERT 모델 클래스 ---
class BertClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2) # 0(부적절), 1(적절)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(output.pooler_output))

class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # BERT 입력: [CLS] 질문 [SEP] 답변 [SEP]
        text = str(row['text_a']) + " [SEP] " + str(row['text_b'])
        
        inputs = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True
        )
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

# --- 4. 메인 실행 ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertClassifier(MODEL_NAME).to(device)
    
    # 데이터 로드
    df = create_bert_dataset("./Training/02.라벨링데이터") # 경로 수정 필수
    if df.empty:
        print("데이터가 없습니다.")
        return

    dataset = BertDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("\n🚀 BERT 학습 시작...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, Acc {correct/len(df):.4f}")

    # 저장
    torch.save(model.state_dict(), "./aac_bert_model.pt")
    print("💾 모델 저장 완료")

    # 추론 테스트
    def predict_score(q, a):
        model.eval()
        text = q + " [SEP] " + a
        inputs = tokenizer(text, return_tensors='pt', max_length=MAX_LEN, padding='max_length', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs[0][1].item() # 1(적절)일 확률

    print("\n--- [TEST] ---")
    q_test = "드시고 가시나요?"
    a_good = "네 먹고 갈게요"
    a_bad = "어서오세요"
    print(f"Q: {q_test}")
    print(f"A1: {a_good} -> 점수: {predict_score(q_test, a_good):.4f}")
    print(f"A2: {a_bad} -> 점수: {predict_score(q_test, a_bad):.4f}")

if __name__ == "__main__":
    main()