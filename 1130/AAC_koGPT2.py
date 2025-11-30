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

MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-5

# --- 2. 데이터 처리 클래스 ---
class AACDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def load_data(self):
        pairs = []
        
        # 1. 경로 탐색: 하위 모든 폴더 중 'TL_01'로 시작하는 폴더 찾기
        print(f"🔍 '{self.data_dir}' 경로 하위에서 'TL_01' 폴더를 찾는 중...")
        
        target_dirs = [
            p for p in self.data_dir.rglob("*") 
            if p.is_dir() and p.name.startswith("TL_01")
        ]
        
        if not target_dirs:
            print(f"❌ '{self.data_dir}' 안에서 'TL_01'로 시작하는 폴더를 하나도 못 찾았습니다.")
            return pd.DataFrame()
        
        print(f"🎯 발견된 폴더 ({len(target_dirs)}개)")

        # 2. JSON 파일 수집
        json_files = []
        for d in target_dirs:
            json_files.extend(list(d.glob('*.json')))
            
        print(f"📂 총 {len(json_files)}개의 JSON 파일을 분석합니다.")

        # 3. 데이터 파싱 (인코딩 utf-8-sig 적용)
        for json_path in tqdm(json_files, desc="JSON 파싱"):
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
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
            except Exception as e:
                continue
                
        print(f"✅ 총 {len(pairs)}개의 대화 쌍 추출 완료.")
        return pd.DataFrame(pairs)

# --- 3. 데이터셋 클래스 ---
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
        
        # GPT 포맷: <usr>질문<sys>답변</s>
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

# --- 4. 메인 실행 함수 ---
def main():
    # 저장 파일명 설정 (.pt 파일)
    # BERT와 동일하게 현재 디렉토리에 저장됩니다.
    save_model_path = "./aac_kogpt2_model.pt"
    save_tokenizer_path = "./aac_tokenizer" # 토크나이저는 폴더로 저장해야 안전함

    # 1. 모델 & 토크나이저 설정
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')
    
    tokenizer.add_special_tokens({'additional_special_tokens': ['<usr>', '<sys>']})
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 2. 데이터 로드
    data_dir = "/local_datasets/AACommu/Training/02.라벨링데이터" 
    
    if not os.path.exists(data_dir):
        print(f"❌ 경로 오류: {data_dir} 를 찾을 수 없습니다.")
        return

    processor = AACDataProcessor(data_dir)
    df = processor.load_data()

    if len(df) == 0:
        print("❌ 학습 데이터가 없습니다.")
        return

    dataset = KoGPT2Dataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 3. 학습 루프
    print("\n🚀 KoGPT2 학습 시작...")
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

    # 4. 저장 (수정됨)
    print("\n💾 저장 진행 중...")
    
    # 모델 가중치만 .pt 파일로 저장 (BERT와 동일한 방식)
    torch.save(model.state_dict(), save_model_path)
    print(f"   - 모델 가중치: {save_model_path}")
    
    # 토크나이저는 폴더에 저장 (특수 토큰 정보 유지용)
    if not os.path.exists(save_tokenizer_path):
        os.makedirs(save_tokenizer_path)
    tokenizer.save_pretrained(save_tokenizer_path)
    print(f"   - 토크나이저: {save_tokenizer_path}")

    # 5. 간단 테스트
    def generate_response(text):
        model.eval()
        input_text = f"<usr>{text}<sys>"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=50, 
                repetition_penalty=2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\n--- [TEST] ---")
    test_q = "어서오세요, 주문하시겠어요?"
    print(f"Q: {test_q}")
    print(f"A: {generate_response(test_q)}")

if __name__ == "__main__":
    main()