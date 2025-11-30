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

# --- [수정됨] 데이터 로딩 관련 설정 (AAC_model_BERT.py 스타일) ---
INPUT_DIR_NAMES = [
    "TL_01.식당카페_01.입장_및_이용안내",
    "TL_01.식당카페_02.자리안내",
    "TL_01.식당카페_03.메뉴추천",
    "TL_01.식당카페_04.메뉴주문",
    "TL_01.식당카페_05.식음료서빙",
    "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
]

def extract_dialogue_pairs_from_json(json_data):
    """AAC_model_BERT.py의 로직을 차용"""
    pairs = []
    video_data = json_data.get('video')
    if not video_data:
        return pairs

    interactions = video_data.get('interactions', [])
    
    for interaction in interactions:
        human_utterances = interaction.get('human_event', {}).get('utterances', [])
        robot_responses = interaction.get('robot_response', [])
        
        input_text = ""
        if human_utterances:
            input_text = human_utterances[0].get('utterance_cap', '').strip()

        output_text = ""
        if robot_responses:
            output_text = robot_responses[0].get('answer', '').strip()
        
        if input_text and output_text:
            pairs.append({
                "q": input_text,     # 손님 (BERT 학습 시 text_b로 사용 예정) -> 문맥에 따라 순서 변경 가능
                "a": output_text     # 점원 (BERT 학습 시 text_a로 사용 예정)
            })
            # 원래 AAC_BERT 로직: q=점원(질문), a=손님(답변)
            # AAC_model_BERT 로직: input=손님, output=점원
            # 여기서는 AAC_BERT의 기존 흐름(적절성 판단)에 맞춰 매핑합니다.
            # 보통 문맥(q) -> 반응(a)의 적절성이므로:
            # 점원 말(output_text) -> 손님 말(input_text)의 적절성인지, 
            # 아니면 손님 말 -> 점원 말의 적절성인지 확인 필요.
            # AAC_BERT 원본 코드 주석: "손님 말(정답/Human)", "직원 말(질문/Robot)"
            # 즉, 직원이 물었을 때 손님이 대답하는 상황을 가정.
            
    # AAC_BERT 원본 흐름 유지:
    # q: robot_response (직원)
    # a: human_event (손님)
    result_pairs = []
    for interaction in interactions:
        human_utterances = interaction.get('human_event', {}).get('utterances', [])
        robot_responses = interaction.get('robot_response', [])
        
        h_txt = "" # 손님
        if human_utterances: h_txt = human_utterances[0].get('utterance_cap', '').strip()
        
        r_txt = "" # 직원
        if robot_responses: r_txt = robot_responses[0].get('answer', '').strip()
        
        if h_txt and r_txt:
            result_pairs.append({'q': r_txt, 'a': h_txt})
            
    return result_pairs

# --- 2. 데이터 처리 및 Negative Sampling ---
def create_bert_dataset(data_dir):
    data_path = Path(data_dir)
    raw_pairs = []
    
    print(f"🎯 학습 대상 디렉토리 ({len(INPUT_DIR_NAMES)}개) 순회 시작...")

    for dir_name in INPUT_DIR_NAMES:
        target_dir = data_path / dir_name
        if not target_dir.exists():
            print(f"  (Skip) {dir_name} 디렉토리가 존재하지 않습니다.")
            continue
            
        json_files = list(target_dir.rglob('*.json'))
        
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # AAC_model_BERT 스타일의 추출 함수 사용
                pairs = extract_dialogue_pairs_from_json(data)
                raw_pairs.extend(pairs)
            except Exception:
                continue

    print(f"✅ 원본 대화 쌍 {len(raw_pairs)}개 추출 완료.")
    if not raw_pairs: return pd.DataFrame()

    # 3. Negative Sampling (정답 1개 + 오답 1개 생성)
    # 이 부분은 AAC_BERT의 고유 로직(분류 모델용)이므로 유지
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
        self.out = nn.Linear(self.bert.config.hidden_size, 2) 

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

# --- 4. 메인 실행 ---
def main():
    # 데이터 경로 (AAC_model_BERT.py와 동일하게 설정)
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

if __name__ == "__main__":
    main()