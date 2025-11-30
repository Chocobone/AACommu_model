import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import os

# --- 1. 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "skt/kogpt2-base-v2"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5

# --- 2. 데이터셋 로드 및 파싱 (JSON 구조 반영) ---
class AACDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def load_data(self):
        """
        JSON 구조:
        video -> interactions -> list -> 
          human_event -> utterances[0] -> utterance_cap (손님 말)
          robot_response[0] -> answer (직원/로봇 말)
        """
        pairs = []
        # 폴더 내 모든 json 파일 검색 (재귀)
        json_files = list(self.data_dir.rglob('*.json'))
        print(f"📂 {len(json_files)}개의 JSON 파일을 찾았습니다.")

        for json_path in tqdm(json_files, desc="JSON 파싱 중"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # video 키 확인
                if 'video' not in data: continue
                
                interactions = data['video'].get('interactions', [])
                for interaction in interactions:
                    # 1. 손님 말 추출 (Target)
                    human_text = ""
                    if 'human_event' in interaction and 'utterances' in interaction['human_event']:
                        utts = interaction['human_event']['utterances']
                        if utts: human_text = utts[0].get('utterance_cap', '').strip()
                    
                    # 2. 직원 말 추출 (Input)
                    robot_text = ""
                    if 'robot_response' in interaction:
                        resps = interaction['robot_response']
                        if resps: robot_text = resps[0].get('answer', '').strip()
                    
                    # AAC 목적: 상대방(Robot)의 말을 듣고 -> 내(Human)가 대답
                    if human_text and robot_text:
                        pairs.append({
                            "q": robot_text,  # 상대방의 말 (STT 입력)
                            "a": human_text   # 추천해줄 나의 대답
                        })
            except Exception as e:
                continue
                
        print(f"✅ 총 {len(pairs)}개의 대화 쌍을 추출했습니다.")
        return pd.DataFrame(pairs)

# --- 3. 데이터셋 클래스 ---
class KoGPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 화자 토큰 정의 (Q:상대방, A:나)
        self.q_token = "<usr>"
        self.a_token = "<sys>"
        self.bos = "</s>"
        self.eos = "</s>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q_text = row['q']
        a_text = row['a']
        
        # 학습 포맷: <usr>상대방말<sys>나의말</s>
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

# --- 4. 학습 및 메인 실행 ---
def main():
    # 1. 토크나이저 & 모델 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>')
    
    # 중요: 화자 구분을 위한 스페셜 토큰 추가
    tokenizer.add_special_tokens({'additional_special_tokens': ['<usr>', '<sys>']})
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer)) # 토큰 개수 변경 반영
    model.to(device)

    # 2. 데이터 로드 (경로를 실제 데이터셋 경로로 수정하세요)
    data_dir = "./Training/02.라벨링데이터"  # ⚠️ 실제 경로로 변경 필수
    
    # 경로가 없으면 테스트용 더미 데이터 생성
    if not os.path.exists(data_dir):
        print("⚠️ 경로를 찾지 못해 더미 데이터로 테스트합니다.")
        df = pd.DataFrame([
            {'q': '어서오세요 주문하시겠어요?', 'a': '아이스 아메리카노 주세요'},
            {'q': '드시고 가시나요?', 'a': '아니요 테이크아웃 할게요'}
        ])
    else:
        processor = AACDataProcessor(data_dir)
        df = processor.load_data()

    dataset = KoGPT2Dataset(df, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 3. 학습 루프
    print("\n🚀 KoGPT2 학습 시작...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # GPT는 입력을 그대로 정답(labels)으로 사용
            outputs = model(input_ids=inputs, attention_mask=mask, labels=inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    # 4. 저장
    model.save_pretrained("./aac_kogpt2_model")
    tokenizer.save_pretrained("./aac_kogpt2_model")
    print("💾 모델 저장 완료: ./aac_kogpt2_model")

    # 5. 추론 테스트 (문장 완성 기능)
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
                top_k=50 # Top-K 샘플링
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\n--- [TEST] ---")
    stt_input = "포인트 적립 하시겠어요?"
    print(f"직원(STT): {stt_input}")
    print(f"AAC추천: {generate_response(stt_input)}")

if __name__ == "__main__":
    main()