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
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5

# --- 2. 데이터 처리 함수 ---
def create_bert_dataset(data_dir):
    data_path = Path(data_dir)
    
    # 1. 경로 탐색: 하위의 모든 TL_01 폴더 찾기
    print(f"🔍 '{data_path}' 경로 하위에서 'TL_01' 폴더를 찾는 중...")
    target_dirs = [
        p for p in data_path.rglob("*") 
        if p.is_dir() and p.name.startswith("TL_01")
    ]
    
    if not target_dirs:
        print(f"❌ '{data_path}' 안에서 'TL_01'로 시작하는 폴더를 찾을 수 없습니다.")
        return pd.DataFrame()

    print(f"🎯 발견된 폴더 ({len(target_dirs)}개)")

    # 2. JSON 파일 수집
    raw_pairs = []
    json_files = []
    for d in target_dirs:
        json_files.extend(list(d.glob('*.json')))
        
    print(f"📂 총 {len(json_files)}개의 JSON 파일을 분석합니다.")

    # 3. 데이터 추출 (인코딩 utf-8-sig 적용)
    for json_path in tqdm(json_files, desc="데이터 추출"):
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            if 'video' not in data: continue
            interactions = data['video'].get('interactions', [])
            for interaction in interactions:
                h_txt = ""
                if 'human_event' in interaction:
                    utts = interaction['human_event'].get('utterances', [])
                    if utts: h_txt = utts[0].get('utterance_cap', '').strip()

                r_txt = ""
                if 'robot_response' in interaction:
                    resps = interaction['robot_response']
                    if resps: r_txt = resps[0].get('answer', '').strip()
                
                if h_txt and r_txt:
                    raw_pairs.append({'q': r_txt, 'a': h_txt})
        except: continue

    if not raw_pairs: return pd.DataFrame()

    # 4. Negative Sampling (정답 1개 + 오답 1개)
    processed_data = []
    all_answers = [p['a'] for p in raw_pairs]
    
    print("부정 샘플(Negative Sample) 생성 중...")
    for p in raw_pairs:
        # Positive (정답) -> Label 1
        processed_data.append({'text_a': p['q'], 'text_b': p['a'], 'label': 1})
        
        # Negative (오답) -> Label 0
        while True:
            random_a = random.choice(all_answers)
            if random_a != p['a']: break
        processed_data.append({'text_a': p['q'], 'text_b': random_a, 'label': 0})
        
    print(f"✅ 최종 학습 데이터: {len(processed_data)}개")
    return pd.DataFrame(processed_data)

# --- 3. BERT 모델 클래스 ---
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

class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text_a']) + " [SEP] " + str(row['text_b'])
        label = row['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. 메인 실행 함수 ---
def main():
    # 데이터 경로
    data_dir = "/local_datasets/AACommu/Training/02.라벨링데이터"
    save_path = "./aac_bert_model.pt"

    if not os.path.exists(data_dir):
        print(f"❌ 경로 오류: {data_dir} 를 찾을 수 없습니다.")
        return

    # 1. 데이터 로드
    df = create_bert_dataset(data_dir)
    if df.empty:
        print("❌ 학습 데이터가 없습니다.")
        return

    # 2. 모델 준비
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertClassifier(MODEL_NAME).to(device)
    
    dataset = BertDataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 3. 학습 루프
    print("\n🚀 BERT 학습 시작...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
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
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/len(df):.4f}")

    # 4. 저장
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 모델 저장 완료: {save_path}")

    # 간단 테스트
    def predict_score(q, a):
        model.eval()
        text = q + " [SEP] " + a
        inputs = tokenizer(text, return_tensors='pt', max_length=MAX_LEN, padding='max_length', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs[0][1].item()

    print("\n--- [TEST] ---")
    print(f"Q: 드시고 가시나요? / A: 네 먹고 갈게요 -> 점수: {predict_score('드시고 가시나요?', '네 먹고 갈게요'):.4f}")

if __name__ == "__main__":
    main()