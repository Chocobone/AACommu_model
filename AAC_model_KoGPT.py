#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader
# AdamW 경고 해결: transformers 대신 torch.optim 사용
from torch.optim import AdamW 
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
import json
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. 설정 및 장치 확인 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# KoGPT2 모델 사용 (한국어 생성 모델)
MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 64  # 문장이 너무 길면 자름 (메모리 절약)
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# [🚨 중요 수정] KoGPT2는 기본 pad_token이 없으므로 eos_token을 pad_token으로 설정
# 이 코드가 없으면 "ValueError: Asking to pad but the tokenizer..." 에러가 발생합니다.
tokenizer.pad_token = tokenizer.eos_token

# --- 2. 데이터 추출 (GPT 형식으로 변환) ---
# GPT 학습 데이터 형식: "<q>질문</s><a>답변</s>" 
INPUT_DIR_NAMES = [
    "TL_01.식당카페_01.입장_및_이용안내",
    "TL_01.식당카페_02.자리안내",
    "TL_01.식당카페_03.메뉴추천",
    "TL_01.식당카페_04.메뉴주문",
    "TL_01.식당카페_05.식음료서빙",
    "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
]
INPUT_DIR = Path("/local_datasets/AACommu/Training/02.라벨링데이터") 

def load_data():
    pairs = []
    print("데이터 로딩 중...")
    
    for dir_name in INPUT_DIR_NAMES:
        dir_path = INPUT_DIR / dir_name
        if not dir_path.is_dir(): continue
        
        for json_path in list(dir_path.rglob('*.json')):
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    
                video = data.get('video', {})
                for interaction in video.get('interactions', []):
                    human = interaction.get('human_event', {}).get('utterances', [])
                    q_text = human[0].get('utterance_cap', '').strip() if human else ""
                    
                    robot = interaction.get('robot_response', [])
                    a_text = robot[0].get('answer', '').strip() if robot else ""
                    
                    if q_text and a_text:
                        # KoGPT2의 bos_token(</s>)을 문장 끝에 붙임
                        formatted_text = f"<q>{q_text}</s><a>{a_text}</s>"
                        pairs.append(formatted_text)
            except:
                continue
                
    print(f"총 {len(pairs)}개의 대화 쌍 추출 완료.")
    return pairs

data_list = load_data()

# 데이터가 없을 때 테스트용 더미 데이터
if not data_list:
    print("⚠️ 데이터 없음. 더미 데이터 사용.")
    data_list = [
        "<q>주문하시겠어요?</s><a>아이스 아메리카노 주세요</s>",
        "<q>드시고 가시나요?</s><a>아니요 테이크아웃 할게요</s>"
    ] * 100

# 학습/검증 분리
train_texts, val_texts = train_test_split(data_list, test_size=0.1, random_state=42)

# --- 3. Dataset 클래스 (GPT 전용) ---
class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

train_dataset = GPTDataset(train_texts, tokenizer, MAX_LEN)
val_dataset = GPTDataset(val_texts, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 4. 모델 초기화 및 학습 ---
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train_step(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def val_step(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

print("\n=== KoGPT2 학습 시작 ===")
for epoch in range(EPOCHS):
    train_loss = train_step(model, train_loader)
    val_loss = val_step(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "AAC_KoGPT2_best.pt")
print("✅ 모델 저장 완료: AAC_KoGPT2_best.pt")

# --- 5. [핵심] Stepwise Chunk Picker 추론 로직 ---
# 사용자가 원했던 "확률적 Top 3 추천" 기능입니다.

def recommend_next_chunks(question, current_answer, top_k=3):
    model.eval()
    
    # 문맥 생성: <q>질문</s><a>현재답변
    input_text = f"<q>{question}</s><a>{current_answer}"
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        results = []
        for i in range(top_k):
            token_id = top_k_indices[i].item()
            probability = top_k_probs[i].item()
            word = tokenizer.decode([token_id])
            results.append((word, probability))
            
    return results

# --- 6. 시연 (Simulation) ---
print("\n--- 🛒 AAC 키오스크 시연 ---")
q = "주문 도와드릴까요?"
curr_a = "" 

print(f"🤖 점원: {q}")

# Step 1: 첫 단어 추천
suggestions = recommend_next_chunks(q, curr_a)
print(f"User 현재 상태: (공란)")
print(f"추천 청크: {[s[0] for s in suggestions]}")

# 가정: 사용자가 추천된 것 중 하나를 선택하거나 직접 입력함
selected_word = "아이스" 
curr_a += selected_word

# Step 2: 다음 단어 추천 (조건부 확률)
suggestions = recommend_next_chunks(q, curr_a)
print(f"\nUser 현재 상태: {curr_a}")
print(f"추천 청크: {[s[0] for s in suggestions]}")