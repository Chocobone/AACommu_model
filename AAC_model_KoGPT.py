#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel
import json
from pathlib import Path
import random
import numpy as np

# --- 1. 기본 설정 및 장치 확인 ---
# 시드 고정 (재현성을 위해)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 설정
MODEL_NAME = "skt/kogpt2-base-ve"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 2. 데이터 추출 및 전처리 (Negative Sampling 적용) ---

INPUT_DIR_NAMES = [
    "TL_01.식당카페_01.입장_및_이용안내",
    "TL_01.식당카페_02.자리안내",
    "TL_01.식당카페_03.메뉴추천",
    "TL_01.식당카페_04.메뉴주문",
    "TL_01.식당카페_05.식음료서빙",
    "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
]
INPUT_DIR = Path("/local_datasets/AACommu/Training/02.라벨링데이터") 
DEFAULT_CONTEXT = "카페"

def extract_dialogue_pairs_from_json(json_data, context):
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
                "context": context,
                "input_text": input_text,
                "output_text": output_text 
            })
    return pairs

def create_training_data(raw_pairs):
    """
    핵심 수정: 부정 샘플링 (Negative Sampling)
    - Positive Sample (Label 1): 실제 질문 + 실제 답변
    - Negative Sample (Label 0): 실제 질문 + (랜덤하게 뽑은 다른 답변)
    """
    if not raw_pairs:
        return pd.DataFrame()

    processed_data = []
    all_answers = [p['output_text'] for p in raw_pairs] # 랜덤 추출용 전체 답변 리스트

    print("데이터셋 생성 중 (Negative Sampling 적용)...")
    
    for pair in raw_pairs:
        question = pair['input_text']
        answer = pair['output_text']
        
        # 1. 정답 데이터 (Positive) -> Label 1
        # BERT 입력 형식: [CLS] 질문 [SEP] 답변 [SEP]
        # 여기서는 text column에 [SEP]를 넣어두고 토크나이징 때 처리
        processed_data.append({
            'text': f"{question} [SEP] {answer}",
            'label': 1
        })
        
        # 2. 오답 데이터 (Negative) -> Label 0
        # 전체 답변 중 랜덤하게 하나 선택 (현재 정답이 아닌 것)
        while True:
            random_answer = random.choice(all_answers)
            if random_answer != answer:
                break
        
        processed_data.append({
            'text': f"{question} [SEP] {random_answer}",
            'label': 0
        })
    
    return pd.DataFrame(processed_data)

def load_and_process_data():
    all_raw_pairs = []
    
    print(f"총 {len(INPUT_DIR_NAMES)}개의 디렉토리를 순회합니다.")
    
    for dir_name in INPUT_DIR_NAMES:
        dir_path = INPUT_DIR / dir_name
        if not dir_path.is_dir():
            print(f"  (Skip) {dir_name} 디렉토리 없음")
            continue

        json_paths = list(dir_path.rglob('*.json'))
        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                pairs = extract_dialogue_pairs_from_json(data, DEFAULT_CONTEXT)
                all_raw_pairs.extend(pairs)
            except:
                continue

    print(f"✅ 원본 Q&A 쌍 {len(all_raw_pairs)}개 추출 완료.")
    
    # 학습용 데이터셋 생성 (Positive + Negative)
    df = create_training_data(all_raw_pairs)
    
    # 데이터가 없을 경우 더미 데이터 생성 (코드 테스트용)
    if len(df) == 0:
        print("⚠️ 데이터가 없어 더미 데이터를 생성합니다.")
        df = pd.DataFrame({
            'text': ["안녕하세요 [SEP] 어서오세요", "안녕하세요 [SEP] 3000원입니다"],
            'label': [1, 0]
        })
    
    print(f"최종 학습 데이터 크기 (Positive+Negative): {len(df)}")
    return df

# 데이터 로드 실행
df_total = load_and_process_data()

# --- 3. Train / Validation 분할 및 DataLoader ---

# 전체 데이터를 8:2로 분할 (stratify 옵션으로 0과 1 비율 유지)
if len(df_total) > 10:
    df_train, df_val = train_test_split(df_total, test_size=0.2, random_state=42, stratify=df_total['label'])
else:
    df_train = df_total
    df_val = df_total # 데이터가 너무 적으면 복사해서 사용

print(f"Train Set: {len(df_train)}, Validation Set: {len(df_val)}")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # [SEP] 토큰이 이미 텍스트 안에 포함되어 있음
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # [CLS], [SEP] 자동 추가 (맨 앞, 맨 뒤)
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(df_train.text.to_list(), df_train.label.to_list(), tokenizer, MAX_LEN)
val_dataset = TextDataset(df_val.text.to_list(), df_val.label.to_list(), tokenizer, MAX_LEN)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 4. 모델 정의 ---

class AACommuModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super(AACommuModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = AACommuModel(2, MODEL_NAME) # 클래스는 0(부적절)과 1(적절) 두 개
model = model.to(device)

# --- 5. 학습 및 평가 함수 ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

# --- 6. 학습 실행 ---

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss().to(device)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_accuracy = 0

print("\n=== Start Model Training ===")

for epoch in range(EPOCHS):
    print(f'\n[Epoch {epoch + 1}/{EPOCHS}]')
    
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        len(df_train)
    )
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f'Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_acc:.4f}')

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    # 검증 정확도가 높아지면 모델 저장
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'AACommu_model_best.pt')
        best_accuracy = val_acc
        print("Model weights saved (Best Accuracy).")

print("\n=== Training Complete ===")

# --- 7. 모델 테스트 (추론 예시) ---
# 학습된 모델이 실제로 어떻게 작동하는지 확인하는 함수
def predict_appropriateness(question, answer):
    text = f"{question} [SEP] {answer}"
    encoding = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        prob = torch.nn.functional.softmax(outputs, dim=1)
    
    return prediction.item(), prob[0][1].item() # 1일 확률

print("\n--- Inference Test ---")
# 테스트 케이스
test_q = "아이스 아메리카노 한 잔 주세요"
test_a_correct = "네, 알겠습니다. 드시고 가시나요?"
test_a_wrong = "오늘 날씨가 참 좋네요."

pred, prob = predict_appropriateness(test_q, test_a_correct)
print(f"Q: {test_q}\nA: {test_a_correct}\n-> 예측: {'적절함' if pred==1 else '부적절'} (확률: {prob:.4f})")

pred, prob = predict_appropriateness(test_q, test_a_wrong)
print(f"Q: {test_q}\nA: {test_a_wrong}\n-> 예측: {'적절함' if pred==1 else '부적절'} (확률: {prob:.4f})")