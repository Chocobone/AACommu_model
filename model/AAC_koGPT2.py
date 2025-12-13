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

# --- [사용자 설정] 디렉토리명 접두사 vs 장소 태그 매핑 ---
# "폴더명이 이걸로 시작하면" : "이 장소 태그를 붙여라"
DIR_CATEGORY_MAP = {
    "TL_01": "카페",   # TL_01... 폴더 안에 있는 건 무조건 <LOC_카페>
    # "TL_02": "식당", # (예시) 나중에 추가 가능
    # "TL_03": "편의점"
}

# --- 1. 기본 설정 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-5

# --- 2. 데이터 처리 클래스 ---
class AACDataProcessor:
    def __init__(self, data_dir, DIR_CATEGORY_MAP):
        self.data_dir = Path(data_dir)
        self.dir_map = DIR_CATEGORY_MAP
        
    def load_data(self):
        pairs = []
        
        # 설정된 매핑(폴더 규칙)마다 반복
        for dir_prefix, tag_name in self.dir_map.items():
            print(f"🔍 '{dir_prefix}'로 시작하는 폴더를 찾는 중... (태그: {tag_name})")
            
            # 해당 접두사로 시작하는 폴더 찾기
            target_dirs = [
                p for p in self.data_dir.rglob("*") 
                if p.is_dir() and p.name.startswith(dir_prefix)
            ]
            
            if not target_dirs:
                print(f"   ⚠️ '{dir_prefix}'로 시작하는 폴더를 찾지 못했습니다. 건너뜁니다.")
                continue
                
            # 파일 수집
            json_files = []
            for d in target_dirs:
                json_files.extend(list(d.glob('*.json')))
            
            print(f"   📂 발견: {len(target_dirs)}개 폴더, {len(json_files)}개 파일 -> 모두 '{tag_name}' 태그 적용")

            # 데이터 파싱
            for json_path in tqdm(json_files, desc=f"{tag_name} 데이터 파싱"):
                try:
                    with open(json_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                    
                    if 'video' not in data: continue
                    
                    # [핵심] JSON 내부 장소 정보는 무시하고, 폴더 규칙에 따른 태그 강제 할당
                    current_tag = f"<LOC_{tag_name}>"

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
                            pairs.append({
                                "place_tag": current_tag,
                                "q": robot_text, 
                                "a": human_text
                            })
                except Exception:
                    continue
                
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
        place_tag = row['place_tag'] # 이미 <LOC_카페> 
        q_text = row['q']
        a_text = row['a']
        
        # 포맷: <LOC_카페><usr>질문<sys>답변</s>
        text = place_tag + self.q_token + q_text + self.a_token + a_text + self.eos
        
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
    save_model_path = "./aac_kogpt2_dir_tag_model.pt"
    save_tokenizer_path = "./aac_dir_tag_tokenizer"

    # 1. 모델 & 토크나이저 설정
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')
    
    # [핵심] 맵핑에 있는 장소들을 스페셜 토큰으로 등록
    # DIR_CATEGORY_MAP의 값(Value)들만 뽑아서 태그로 만듦
    mapped_places = list(DIR_CATEGORY_MAP.values()) # ['카페', ...]
    loc_tokens = [f"<LOC_{p}>" for p in mapped_places]
    
    # 혹시 모를 미분류를 위해 기타 추가 (필요 없으면 제거 가능)
    if "<LOC_기타>" not in loc_tokens:
        loc_tokens.append("<LOC_기타>")

    special_tokens = loc_tokens + ['<usr>', '<sys>']
    
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"✅ 등록된 장소 태그: {loc_tokens}")

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 2. 데이터 로드
    data_dir = "/local_datasets/AACommu/Training/02.라벨링데이터"
    if not os.path.exists(data_dir):
        print("❌ 경로 오류")
        return

    processor = AACDataProcessor(data_dir, DIR_CATEGORY_MAP)
    df = processor.load_data()

    if len(df) == 0:
        print("❌ 학습 데이터가 없습니다. 폴더명을 확인해주세요.")
        return

    print(f"📊 총 데이터: {len(df)}개")
    print(f"   - 데이터 예시: {df.iloc[0]['place_tag']} | Q: {df.iloc[0]['q']}")

    dataset = KoGPT2Dataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 3. 학습
    print("\n🚀 학습 시작...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}")
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

    # 4. 저장
    torch.save(model.state_dict(), save_model_path)
    if not os.path.exists(save_tokenizer_path):
        os.makedirs(save_tokenizer_path)
    tokenizer.save_pretrained(save_tokenizer_path)
    print("💾 저장 완료")

    # 5. 테스트
    def generate_response(place_name, text):
        model.eval()
        tag = f"<LOC_{place_name}>"
        
        # 학습에 안 쓴 장소를 넣으면 경고
        if tag not in tokenizer.get_added_vocab():
             print(f"⚠️ 학습되지 않은 장소 태그입니다: {tag}")

        input_text = f"{tag}<usr>{text}<sys>"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=64, 
                repetition_penalty=2.0, 
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\n--- [TEST] ---")
    # 설정한 장소 중 하나 테스트
    test_place = list(DIR_CATEGORY_MAP.values())[0] 
    print(f"장소: {test_place}")
    print(f"Q: 주문하시겠어요?")
    print(f"A: {generate_response(test_place, '주문하시겠어요?')}")

if __name__ == "__main__":
    main()