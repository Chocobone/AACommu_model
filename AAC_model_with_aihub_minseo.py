#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel
import zipfile
import json
from pathlib import Path
import random
import numpy as np
from typing import List, Dict
import torch.nn.functional as F


# In[ ]:


# --- 1. 기본 설정 및 장치 확인 (AAC_model.ipynb) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 설정
MODEL_NAME = "klue/bert-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[ ]:


# --- 2. 데이터 추출 함수 ---

# 파일 경로 설정 (실제 환경에 맞게 수정 필요)
INPUT_DIR_NAMES = [
    "TL_01.식당카페_01.입장_및_이용안내",
    "TL_01.식당카페_02.자리안내",
    "TL_01.식당카페_03.메뉴추천",
    "TL_01.식당카페_04.메뉴주문",
    "TL_01.식당카페_05.식음료서빙",
    "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
]
# ZIP 파일이 위치한 디렉토리 (AAC_model.ipynb의 더미 경로 대신 실제 ZIP 경로 사용)
INPUT_DIR = Path("/local_datasets/AACommu/Training/02.라벨링데이터") 
DEFAULT_CONTEXT = "카페"
all_dialogue_pairs = []


# In[ ]:


def extract_dialogue_pairs_from_json(json_data, context):
    """
    JSON 파일 구조를 참고하여 'human_event.utterances.utterance_cap'을
    input_text로, 'robot_response.answer'를 output_text로 추출합니다.
    """
    pairs = []
    video_data = json_data.get('video')
    if not video_data:
        return pairs

    interactions = video_data.get('interactions', [])
    
    for interaction in interactions:
        human_utterances = interaction.get('human_event', {}).get('utterances', [])
        robot_responses = interaction.get('robot_response', [])
        
        # Human Input 추출: 첫 번째 발화의 'utterance_cap'을 input으로 사용
        input_text = ""
        if human_utterances:
            input_text = human_utterances[0].get('utterance_cap', '').strip()

        # Robot Output 추출: 첫 번째 로봇 응답의 'answer'를 output으로 사용
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
"""
def load_and_extract_data():
    print(f"총 {len(INPUT_ZIP_FILENAMES)}개의 압축 파일을 처리합니다.")
    
    for zip_filename in INPUT_ZIP_FILENAMES:
        zip_path = INPUT_DIR / zip_filename
        # print(f"\n--- {zip_filename} 처리 시작 ---")
        
        if not zip_path.exists():
            # print(f"🚨 경고: {zip_filename} 파일을 찾을 수 없습니다. (스킵)")
            continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                json_files = [f for f in zf.namelist() if f.endswith('.json')]
                
                for json_file in json_files:
                    try:
                        with zf.open(json_file, 'r') as f:
                            raw_data = f.read().decode('utf-8')
                            data = json.loads(raw_data)
                        
                        # 수정된 추출 함수 사용
                        pairs = extract_dialogue_pairs_from_json(data, DEFAULT_CONTEXT)
                        global all_dialogue_pairs
                        all_dialogue_pairs.extend(pairs)
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            # print(f"🚨 기타 오류 발생: {e}")
            continue

    print(f"\n✅ 데이터 추출 완료. 총 {len(all_dialogue_pairs)}개의 Q&A 쌍을 로드했습니다.")
    
    # 추출된 Q&A 쌍을 BERT 분류 모델에 맞게 변환 (AAC_model.ipynb의 load_and_preprocess_data 로직을 대체)
    if not all_dialogue_pairs:
        # 추출된 데이터가 없을 경우 더미 데이터 생성
        print("⚠️ Warning: No data extracted. Using dummy data for structure.")
        df_combined = pd.DataFrame({
            'text': ["JSON 파일 하나만 사용합니다.", "오늘의 학습 주제는 BERT입니다.", "데이터셋을 단일 파일로 구성했습니다."], 
            'label': [0, 1, 0]
        })
    else:
        # 데이터프레임 생성
        df_combined = pd.DataFrame(all_dialogue_pairs)
        
        # AAC_model.ipynb 로직에 따라 text 컬럼과 임의의 이진 label 컬럼 생성
        df_combined['text'] = df_combined['input_text'] # input_text를 텍스트 분류 입력으로 사용
        df_combined['label'] = 0
        df_combined.loc[df_combined.index % 2 == 0, 'label'] = 1 # 짝수 인덱스에 임의로 레이블 1 부여

        # 필수 컬럼만 남김
        df_combined = df_combined.loc[:, ['text', 'label']]
        
    print(f"Total data size (BERT Classification): {len(df_combined)}")
    return df_combined """


def load_and_extract_data():
    """
    INPUT_DIR 하위에서 INPUT_DIR_NAMES 리스트에 해당하는 폴더를 찾아, 
    그 폴더 내부의 모든 .json 파일을 로드합니다.
    """
    global all_dialogue_pairs
    all_dialogue_pairs = [] # 데이터 중복 방지를 위해 초기화
    
    print(f"총 {len(INPUT_DIR_NAMES)}개의 디렉토리를 순회하며 JSON 파일을 처리합니다.")
    
    # 훈련 데이터셋 디렉토리 순회
    for dir_name in INPUT_DIR_NAMES:
        dir_path = INPUT_DIR / dir_name
        print(f"\n--- {dir_name} 디렉토리 처리 시작 ---")

        if not dir_path.is_dir():
            print(f"🚨 경고: {dir_name} 디렉토리를 찾을 수 없습니다. (스킵)")
            continue

        json_paths = list(dir_path.rglob('*.json'))
        print(f"  총 {len(json_paths)}개의 JSON 파일 발견.")
        
        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                # 추출 함수 사용
                pairs = extract_dialogue_pairs_from_json(data, DEFAULT_CONTEXT)
                all_dialogue_pairs.extend(pairs)
                
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    print(f"\n✅ 데이터 추출 완료. 총 {len(all_dialogue_pairs)}개의 Q&A 쌍을 로드했습니다.")
    
    # 추출된 Q&A 쌍을 BERT 분류 모델에 맞게 변환
    if not all_dialogue_pairs:
        # 추출된 데이터가 없을 경우 더미 데이터 생성
        print("⚠️ Warning: No data extracted. Using dummy data for structure.")
        df_combined = pd.DataFrame({
            'text': ["JSON 파일 하나만 사용합니다.", "오늘의 학습 주제는 BERT입니다.", "데이터셋을 단일 파일로 구성했습니다."], 
            'label': [0, 1, 0]
        })
    else:
        # 데이터프레임 생성
        df_combined = pd.DataFrame(all_dialogue_pairs)
        
        # input_text를 텍스트 분류 입력으로 사용하고 임의의 이진 label 컬럼 생성
        df_combined['text'] = df_combined['input_text'] 
        df_combined['label'] = 0
        df_combined.loc[df_combined.index % 2 == 0, 'label'] = 1 

        # 필수 컬럼만 남김
        df_combined = df_combined.loc[:, ['text', 'label']]
        
    print(f"Total data size (BERT Classification): {len(df_combined)}")
    return df_combined


# 데이터 추출 및 전처리 실행
df = load_and_extract_data()

# --- 3. Dataset 및 DataLoader 생성 (AAC_model.ipynb) ---

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

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
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


df_train = df
df_val = df.iloc[:1] if len(df) > 0 else pd.DataFrame({'text': [], 'label': []})
# Dataset 및 DataLoader 생성
train_dataset = TextDataset(
    texts=df_train.text.to_list(),
    labels=df_train.label.to_list(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_dataset = TextDataset(
    texts=df_val.text.to_list(),
    labels=df_val.label.to_list(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# In[ ]:


# --- 4. 모델 정의 및 초기화 ---

class AACommuModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super(AACommuModel, self).__init__()
        # 사전 학습된 BERT 모델 로드
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        # 분류를 위한 출력 레이어
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # BERT 모델에 입력 (pooled_output만 사용)
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

# 모델 초기화 (레이블 수에 따라 클래스 개수 설정)
N_CLASSES = df['label'].nunique() if len(df) > 0 else 2
model = AACommuModel(N_CLASSES, MODEL_NAME)
model = model.to(device)


# In[ ]:

# ================================
# 7. 청크 추천용 유틸 함수
# ================================

def encode_sentence_with_bert(text: str) -> torch.Tensor:
    """
    주어진 문장을 BERT의 pooled_output 벡터로 인코딩합니다.
    AACommuModel 안의 BERT를 그대로 사용합니다.
    """
    model.eval()

    encoding = tokenizer(
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
        # model.bert는 (sequence_output, pooled_output)을 반환
        _, pooled_output = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # (1, hidden_size) -> (hidden_size,)
    return pooled_output.squeeze(0)


'''def recommend_chunks(
    category: str,
    stt_text: str,
    prev_chunk: str = "",
    top_k: int = 3
):
    """
    카테고리(예: '카페'), STT 텍스트, 이전 청크를 받아
    현재 상황에서 사용할 수 있는 Top-K 청크(= 로봇 응답 후보)를 추천합니다.

    - category: '카페' 등 context에 해당
    - stt_text: 사용자의 음성 인식(STT) 결과 문장
    - prev_chunk: 직전 선택된 청크(시작이면 빈 문자열 "")
    - top_k: 상위 몇 개의 청크를 반환할지 (기본 3개)

    반환값: [(score, chunk_text, raw_pair_dict), ...] 리스트
    """
    if not all_dialogue_pairs:
        print("⚠️ all_dialogue_pairs가 비어 있습니다. 먼저 load_and_extract_data()가 호출되었는지 확인하세요.")
        return []

    # 1) 카테고리로 필터링 (예: context == '카페')
    candidates = [
        pair for pair in all_dialogue_pairs
        if pair.get("context") == category
    ]

    if not candidates:
        print(f"⚠️ 카테고리 '{category}' 에 해당하는 후보 청크가 없습니다.")
        return []

    # 2) (선택) 이전 청크 기반 추가 필터링
    #    현재 JSON 구조에는 이전 청크 정보가 없으므로, prev_chunk는 아직 사용하지 않고
    #    나중에 대화 흐름 정보가 생기면 여기서 조건을 추가할 수 있습니다.
    # if prev_chunk:
    #     candidates = [c for c in candidates if c.get('prev_chunk') == prev_chunk]

    # 3) 입력 문장(STT) 임베딩
    query_emb = encode_sentence_with_bert(stt_text)

    scores = []
    for pair in candidates:
        chunk_text = pair.get("output_text", "").strip()
        if not chunk_text:
            continue

        chunk_emb = encode_sentence_with_bert(chunk_text)
        # 코사인 유사도 계산
        sim = F.cosine_similarity(query_emb, chunk_emb, dim=0).item()
        scores.append((sim, chunk_text, pair))

    if not scores:
        print("⚠️ 유효한 청크 텍스트가 없어 추천을 만들 수 없습니다.")
        return []

    # 4) 유사도 기준으로 내림차순 정렬 후 Top-K 반환
    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:top_k]

    return top_scores'''

def recommend_chunks(
    category: str,
    stt_text: str,
    prev_chunk: str = "",
    top_k: int = 3
):
    """
    카테고리, STT 텍스트, 이전 청크를 받아
    Top-K 청크 문자열 리스트를 반환합니다.

    반환 예:
    ["아이스 아메리카노 추천드릴까요?", "아메리카노 어떠세요?", "차가운 음료를 추천드릴게요."]
    """
    if not all_dialogue_pairs:
        print("⚠️ all_dialogue_pairs가 비어 있습니다. 먼저 load_and_extract_data()가 호출되었는지 확인하세요.")
        return []

    # 1) 카테고리 필터링
    candidates = [
        pair for pair in all_dialogue_pairs
        if pair.get("context") == category
    ]

    if not candidates:
        print(f"⚠️ 카테고리 '{category}' 에 해당하는 청크가 없습니다.")
        return []

    # 2) 입력(STT) 문장 임베딩
    query_emb = encode_sentence_with_bert(stt_text)

    scored_chunks = []

    for pair in candidates:
        chunk_text = pair.get("output_text", "").strip()
        if not chunk_text:
            continue

        chunk_emb = encode_sentence_with_bert(chunk_text)

        # 코사인 유사도 계산
        sim = F.cosine_similarity(query_emb, chunk_emb, dim=0).item()

        scored_chunks.append((sim, chunk_text))

    if not scored_chunks:
        return []

    # 3) 유사도 내림차순 정렬 → Top-K
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_scores = scored_chunks[:top_k]

    # ✅ 오직 "청크 문자열만" 리스트로 반환
    return [chunk_text for _, chunk_text in top_scores]



# --- 5. 학습 및 평가 함수 ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for step, d in enumerate(data_loader):
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
        
        if step % 100 == 0 and step > 0:
            print(f"  Step {step}: Loss {loss.item():.4f}")

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


# In[ ]:


# --- 6. 학습 루프 실행 ---

if len(df_train) > 0 and len(df_val) > 0:
    # 옵티마이저와 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 학습 시작
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_accuracy = 0

    print("\n=== Start Model Training ===")

    for epoch in range(EPOCHS):
        print(f'\n[Epoch {epoch + 1}/{EPOCHS}]')
        
        # 훈련 (Train)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(df_train)
        )

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        
        # 검증 (Validation)
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # 결과 저장 및 최적 모델 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'AACommu_model_best_weights.pt')
            best_accuracy = val_acc
            print("Model weights saved: AACommu_model_best_weights.pt")

    print("\n=== Training Complete ===")
else:
    print("\n❌ 데이터 부족으로 학습을 시작할 수 없습니다. ZIP 파일 경로를 확인해 주세요.")


# In[ ]:


# Validation datasets을 사용한 모델 검증 ---

# 검증 ZIP 파일명 목록 (압축 해제된 폴더명으로 사용)
VALIDATION_DIR_NAMES = [
    "VL_01.식당카페_01.입장_및_이용안내_01.json",
    "VL_01.식당카페_02.자리안내_01.json",
    "VL_01.식당카페_03.메뉴추천_01.json",
    "VL_01.식당카페_04.메뉴주문_01.json",
    "VL_01.식당카페_05.식음료서빙_01.json",
    "VL_01.식당카페_06.결제_및_할인_포인트적립_안내_01.json",
]

# ZIP 파일이 압축 해제된 폴더들을 담고 있는 최상위 디렉토리
VALIDATION_DIR = Path("/local_datasets/AACommu/Validation/02.라벨링데이터") 
DEFAULT_CONTEXT = "카페"

# 모든 검증 쌍을 저장할 리스트
all_validation_pairs = []


# In[ ]:


# 검증 데이터 로드 및 전처리
def load_and_extract_validation_data():
    """
    VALIDATION_DIR 하위에서 VALIDATION_DIR_NAMES 리스트에 해당하는 폴더를 찾아, 
    그 폴더 내부의 모든 .json 파일을 로드합니다.
    """
    global all_validation_pairs
    all_validation_pairs = [] # 초기화
    
    print(f"\n총 {len(VALIDATION_DIR_NAMES)}개의 검증 디렉토리를 순회하며 JSON 파일을 처리합니다.")
    
    # 검증 데이터셋 디렉토리 순회
    for dir_name in VALIDATION_DIR_NAMES:
        dir_path = VALIDATION_DIR / dir_name
        
        if not dir_path.is_dir():
            # print(f"🚨 경고: {dir_name} 디렉토리를 찾을 수 없습니다. (스킵)")
            continue

        json_paths = list(dir_path.rglob('*.json'))
        # print(f"  총 {len(json_paths)}개의 JSON 파일 발견.")
        
        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                pairs = extract_dialogue_pairs_from_json(data, DEFAULT_CONTEXT)
                all_validation_pairs.extend(pairs)
                
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    print(f"✅ 검증 데이터 추출 완료. 총 {len(all_validation_pairs)}개의 Q&A 쌍을 로드했습니다.")
    
    # BERT 분류 모델에 맞게 변환
    if not all_validation_pairs:
        df_val_combined = pd.DataFrame({'text': ["더미 검증 데이터"], 'label': [0]})
    else:
        df_val_combined = pd.DataFrame(all_validation_pairs)
        df_val_combined['text'] = df_val_combined['input_text']
        df_val_combined['label'] = 0 
        df_val_combined.loc[df_val_combined.index % 2 != 0, 'label'] = 1 
        df_val_combined = df_val_combined.loc[:, ['text', 'label']]
        
    print(f"Total validation data size (BERT Classification): {len(df_val_combined)}")
    return df_val_combined

# 검증 데이터 로드 실행
df_validation = load_and_extract_validation_data()

# 검증 데이터셋 및 DataLoader 생성
validation_dataset = TextDataset(
    texts=df_validation.text.to_list(),
    labels=df_validation.label.to_list(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

# 최종 검증 실행
if len(df_validation) > 0:
    print("\n=== Start Final Validation ===")
    try:
        model.load_state_dict(torch.load('AACommu_model_best_weights.pt'))
        model.to(device)
        print("Best model weights loaded.")
        
        final_val_acc, final_val_loss = eval_model(
            model,
            validation_data_loader,
            loss_fn,
            device,
            len(df_validation)
        )
        
        print(f'\nFinal Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_acc:.4f}')
    except FileNotFoundError:
        print("⚠️ Best model weights file not found. Skipping final validation with saved weights.")
    except NameError:
        print("⚠️ Loss function or model not defined. Skipping final validation. Check training execution.")
    except Exception as e:
        print(f"⚠️ Error during final validation: {e}")
    print("=== Final Validation Complete ===")
else:
    print("\n❌ 검증 데이터 부족으로 최종 검증을 건너뜁니다.")

if __name__ == "__main__":
    # 예시: 카테고리 = "카페", STT 문장 = "추천하시겠어요", 이전 청크는 시작이므로 빈 문자열
    test_category = "카페"
    test_stt = "추천하시겠어요"
    prev_chunk = ""

    results = recommend_chunks(test_category, test_stt, prev_chunk, top_k=3)

    print("\n=== 추천 청크 Top 3 ===")
    for rank, (score, chunk_text, pair) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} | chunk='{chunk_text}'")
