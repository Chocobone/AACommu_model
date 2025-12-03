import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pandas as pd
import numpy as np

# 1. 모델 설정 (SKT KoGPT2 사용)
MODEL_NAME = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME, 
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>') 
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# 2. 데이터셋 정의 (Chatbot 스타일)
class AACDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.q_token = "<usr>" # 질문 시작 토큰
        self.a_token = "<sys>" # 답변 시작 토큰
        self.sent_token = "</s>" # 문장 끝 토큰

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        q = record['input_text']
        a = record['output_text']
        
        # GPT 학습 포맷: <usr>질문<sys>답변</s>
        text = self.q_token + q + self.a_token + a + self.sent_token
        
        tokenized = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128,
            add_special_tokens=True
        )
        
        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0]
        }

# (데이터 로드 부분은 기존 코드 활용하되, create_training_data 함수만 수정하여 raw pair를 그대로 사용)
# df_train = ... (기존 load_and_process_data에서 Negative Sampling 제거하고 원본 쌍만 유지)

# 3. 추론 로직 (핵심: 다음 단어 추천)
def suggest_next_tokens(model, tokenizer, question, current_answer_context, top_k=3):
    """
    question: 상대방의 말 (STT)
    current_answer_context: 사용자가 지금까지 선택해서 만든 문장
    """
    model.eval()
    
    # 입력 구성: <usr>질문<sys>현재까지_답변
    input_text = f"<usr>{question}<sys>{current_answer_context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits[0, -1, :] # 마지막 토큰의 예측값
        
        # 확률이 가장 높은 Top-K 토큰 추출
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        suggestions = []
        for idx in top_indices:
            token = tokenizer.decode([idx])
            suggestions.append(token)
            
    return suggestions