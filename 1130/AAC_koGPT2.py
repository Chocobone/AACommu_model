import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import os
import random
import numpy as np

# --- 1. 설정 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-5

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
    """AAC_model_BERT.py의 파싱 로직 적용"""
    pairs = []
    video_data = json_data.get('video')
    if not video_data:
        return pairs

    interactions = video_data.get('interactions', [])
    for interaction in interactions:
        # 손님 말 (Target)
        human_text = ""
        if 'human_event' in interaction and 'utterances' in interaction['human_event']:
            utts = interaction['human_event']['utterances']
            if utts: human_text = utts[0].get('utterance_cap', '').strip()
        
        # 직원 말 (Input)
        robot_text = ""
        if 'robot_response' in interaction:
            resps = interaction['robot_response']
            if resps: robot_text = resps[0].get('answer', '').strip()
        
        if human_text and robot_text:
            pairs.append({
                "q": robot_text, 
                "a": human_text
            })
    return pairs

# --- 2. 데이터셋 로드 및 파싱 클래스 ---
class AACDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    # 기존 AACDataProcessor 클래스 내부의 load_data 함수를 이걸로 통째로 바꾸세요
    def load_data(self):
        pairs = []
        
        # [핵심 수정] iterdir() 대신 rglob('*')을 사용하여 하위 폴더를 깊숙이 탐색
        print(f"🔍 '{self.data_dir}' 경로 하위에서 'TL_01' 폴더를 찾는 중...")
        
        # 1. 모든 하위 경로 중 디렉토리이면서 이름이 'TL_01'로 시작하는 것 찾기
        target_dirs = [
            p for p in self.data_dir.rglob("*") 
            if p.is_dir() and p.name.startswith("TL_01")
        ]
        
        if not target_dirs:
            print(f"❌ '{self.data_dir}' 안에서 'TL_01'로 시작하는 폴더를 하나도 못 찾았습니다.")
            print("   -> 경로 설정이 상위 폴더(예: AACommu)로 되어있는지, 혹은 절대 경로가 맞는지 확인해주세요.")
            return pd.DataFrame()
        
        print(f"🎯 발견된 폴더 ({len(target_dirs)}개):")
        for d in target_dirs[:3]: # 너무 많으면 3개만 출력
            print(f"  - {d}")
        if len(target_dirs) > 3: print("  ... (생략)")

        # 2. JSON 파일 수집
        json_files = []
        for d in target_dirs:
            json_files.extend(list(d.glob('*.json')))
            
        print(f"📂 총 {len(json_files)}개의 JSON 파일을 분석합니다.")

        # 3. 파싱 (기존과 동일)
        for json_path in tqdm(json_files, desc="JSON 파싱"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'video' not in data: continue
                interactions = data['video'].get('interactions', [])
                for interaction in interactions:
                    human_text = ""
                    if 'human_event' in interaction and 'utterances' in interaction['human_event']:
                        utts = interaction['human_event']['utterances']
                        if utts: human_text = utts[0].get('utterance_cap', '').strip()
                    
                    robot_text = ""
                    if 'robot_response' in interaction:
                        resps = interaction['robot_response']
                        if resps: robot_text = resps[0].get('answer', '').strip()
                    
                    if human_text and robot_text:
                        pairs.append({"q": robot_text, "a": human_text})
            except: continue
                
        print(f"✅ 총 {len(pairs)}개의 대화 쌍 추출 완료.")
        return pd.DataFrame(pairs)
                
        print(f"✅ 총 {len(pairs)}개의 대화 쌍 추출 완료.")
        return pd.DataFrame(pairs)

# --- 3. 데이터셋 클래스 (KoGPT2 포맷) ---
class KoGPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.q_token = "<usr>"
        self.a_token = "<sys>"
        self.eos = "</s>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q_text = row['q']
        a_text = row['a']
        
        # 포맷: <usr>직원말<sys>손님말</s>
        text = self.q_token + q_text + self.a_token + a_text + self.eos
        
        tokenized = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True
        )
        
        input_ids = tokenized['input_ids'][0]
        mask = tokenized['attention_mask'][0]
        
        return {'input_ids': input_ids, 'attention_mask': mask}

# --- 4. 메인 실행 ---
def main():
    save_path = "./aac_kogpt2_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')
    
    tokenizer.add_special_tokens({'additional_special_tokens': ['<usr>', '<sys>']})
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 경로 설정 (AAC_model_BERT.py와 동일)
    data_dir = "/local_datasets/AACommu/Training/02.라벨링데이터" 
    
    if not os.path.exists(data_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {data_dir}")
        return

    processor = AACDataProcessor(data_dir)
    df = processor.load_data()

    if len(df) == 0:
        print("❌ 학습할 데이터가 없습니다.")
        return

    dataset = KoGPT2Dataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("\n🚀 KoGPT2 학습 시작 (식당/카페 전용)...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=inputs, attention_mask=mask, labels=inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n💾 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    main()