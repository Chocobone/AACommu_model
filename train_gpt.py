# train_gpt.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW 
from transformers import AutoTokenizer, GPT2LMHeadModel
import json
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split

# 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 5e-5

# 토크나이저 설정 (중요: 패딩 토큰 추가)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token 

# 데이터 로드
INPUT_DIR = Path("/local_datasets/AACommu/Training/02.라벨링데이터") 
INPUT_DIR_NAMES = [
    "TL_01.식당카페_01.입장_및_이용안내", "TL_01.식당카페_02.자리안내",
    "TL_01.식당카페_03.메뉴추천", "TL_01.식당카페_04.메뉴주문",
    "TL_01.식당카페_05.식음료서빙", "TL_01.식당카페_06.결제_및_할인_포인트적립_안내",
]

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
                    robot = interaction.get('robot_response', [])
                    if human and robot:
                        q = human[0].get('utterance_cap', '').strip()
                        a = robot[0].get('answer', '').strip()
                        if q and a:
                            # GPT 학습 포맷: <q>질문</s><a>답변</s>
                            pairs.append(f"<q>{q}</s><a>{a}</s>")
            except: continue
    return pairs

data_list = load_data()
train_texts, val_texts = train_test_split(data_list, test_size=0.1, random_state=42)

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
            text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(GPTDataset(train_texts, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)

# 모델 학습
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("=== GPT 학습 시작 ===")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "AAC_KoGPT2_best.pt")
print("✅ GPT 학습 완료: AAC_KoGPT2_best.pt 생성됨")