# KoGPT-2 기반 AAC 청크 추천 MVP(Minimum Viable Product) 프로젝트
# 이 스크립트는 KoGPT-2 모델을 특정 상황에 맞는 AAC 청크 추천을 위해 파인튜닝하고,
# 학습된 모델을 사용하여 실제 추론까지 실행하는 전체 과정을 담고 있습니다.

# --- 1. 환경 설정 ---
# 필요한 라이브러리를 설치합니다. (최초 실행 시 한 번만 필요)
# !pip install torch transformers datasets pandas scikit-learn sentencepiece protobuf

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

print("라이브러리 로드 완료!")

# --- 2. 샘플 데이터셋 생성 ---
# MVP 학습을 위해 데이터를 20개로 구성합니다.
train_data_json = [
    {"context": "카페", "input_text": "안녕하세요, 주문 도와드릴까요?", "output_text": "아이스/따뜻한 <sep> 아메리카노/카페라떼 <sep> 주세요"},
    {"context": "카페", "input_text": "더 필요한 거 있으세요?", "output_text": "아니요 <sep> 괜찮아요"},
    {"context": "카페", "input_text": "포인트 적립이나 할인카드 있으신가요?", "output_text": "아니요 <sep> 없어요"},
    {"context": "카페", "input_text": "드시고 가세요? 아니면 포장이세요?", "output_text": "먹고 갈게요/포장해주세요"},
    {"context": "병원", "input_text": "어디가 불편해서 오셨어요?", "output_text": "어제부터/오늘 아침부터 <sep> 머리가/배가 <sep> 아파요"},
    {"context": "병원", "input_text": "성함이 어떻게 되세요?", "output_text": "제 이름은 <sep> 김민준/이서연 <sep> 입니다"},
    {"context": "병원", "input_text": "주민등록번호 앞자리를 말씀해주세요.", "output_text": "900101/001231 <sep> 입니다"},
    {"context": "병원", "input_text": "이전에 저희 병원 오신 적 있으세요?", "output_text": "네, 있어요/아니요, 처음이에요"},
    {"context": "식당", "input_text": "몇 분이세요?", "output_text": "한 명/두 명/세 명 <sep> 이요"},
    {"context": "식당", "input_text": "메뉴 뭐로 하시겠어요?", "output_text": "김치찌개/된장찌개 <sep> 하나 <sep> 주세요"},
    {"context": "식당", "input_text": "주문하신 메뉴 나왔습니다.", "output_text": "네 <sep> 감사합니다"},
    {"context": "식당", "input_text": "더 필요한 반찬 있으시면 말씀해주세요.", "output_text": "김치/단무지 <sep> 좀 더 주세요"},
    {"context": "편의점", "input_text": "봉투 필요하세요?", "output_text": "네/아니요 <sep> 괜찮아요"},
    {"context": "편의점", "input_text": "결제 도와드리겠습니다.", "output_text": "카드로/현금으로 <sep> 할게요"},
    {"context": "편의점", "input_text": "담배 찾으시는 거 있으세요?", "output_text": "아니요/에쎄 체인지 <sep> 주세요"},
    {"context": "편의점", "input_text": "따로 데워드릴까요?", "output_text": "네, 데워주세요/아니요, 그냥 주세요"},
    {"context": "관공서", "input_text": "어떤 업무 때문에 오셨나요?", "output_text": "등본/인감 <sep> 떼러 왔어요"},
    {"context": "관공서", "input_text": "신분증 좀 보여주시겠어요?", "output_text": "네 <sep> 여기요"},
    {"context": "관공서", "input_text": "이 서류에 서명 부탁드립니다.", "output_text": "어디에 하면 되나요?"},
    {"context": "관공서", "input_text": "수수료는 500원입니다.", "output_text": "네 <sep> 알겠습니다"}
]

print("샘플 데이터 생성 완료!")

# --- 3. 모델 및 토크나이저 로드 ---
MODEL_NAME = "skt/kogpt2-base-v2"

# 3.1. 토크나이저 로드 및 특수 토큰 추가
# [수정] GPT2Tokenizer -> AutoTokenizer로 변경하여 호환성 문제를 해결합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(special_tokens)

# 3.2. GPT-2 모델 로드
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer)) # 새로운 토큰에 맞게 모델 임베딩 크기 조정

print("모델 및 토크나이저 준비 완료!")

# --- 4. 데이터셋 전처리 ---
df = pd.DataFrame(train_data_json)
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    # GPT-2는 하나의 연속된 텍스트로 학습합니다.
    # 형식: <bos> 상황 <sep> 입력 발화 <sep> 출력 청크 <eos>
    full_texts = [f"{tokenizer.bos_token}{ctx} {tokenizer.sep_token} {inp} {tokenizer.sep_token} {out}{tokenizer.eos_token}"
                  for ctx, inp, out in zip(examples['context'], examples['input_text'], examples['output_text'])]
    
    return tokenizer(full_texts, padding="max_length", truncation=True, max_length=128)

processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print("데이터셋 전처리 완료!")

# --- 5. 모델 파인튜닝 ---
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 데이터 콜레이터 설정 (Causal LM용)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # Masked Language Modeling이 아닌 Causal Language Modeling
)

# 학습 인자(Hyperparameters) 설정
training_args = TrainingArguments(
    output_dir="./kogpt2_aac_mvp_model",
    num_train_epochs=30, # 데이터가 적으므로 충분히 학습하도록 epoch 증가
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=3,
)

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("모델 파인튜닝을 시작합니다...")
trainer.train()
print("모델 파인튜닝 완료!")

# 최종 모델 저장
FINAL_MODEL_PATH = "./final_kogpt2_aac_mvp_model"
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"최종 모델이 '{FINAL_MODEL_PATH}'에 저장되었습니다.")

# --- 6. 학습된 모델로 추론(Inference) 실행 ---
def recommend_chunks(context, input_text, model_path=FINAL_MODEL_PATH):
    """
    파인튜닝된 KoGPT-2 모델을 로드하여 새로운 입력에 대한 AAC 청크를 추천하는 함수
    """
    # 1. 저장된 모델과 토크나이저 로드
    # [수정] GPT2Tokenizer -> AutoTokenizer로 변경
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 2. 추론을 위한 입력 프롬프트 생성
    prompt = f"{tokenizer.bos_token}{context} {tokenizer.sep_token} {input_text} {tokenizer.sep_token}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. 모델을 통해 결과 생성
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=5,
        repetition_penalty=2.0,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 4. 생성된 토큰 ID를 텍스트로 변환
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 5. 후처리: 프롬프트 부분을 제거하고 청크로 분리
    generated_text = generated_text.replace(f"{context} {tokenizer.sep_token} {input_text} {tokenizer.sep_token}", "").strip()
    chunks = generated_text.split(tokenizer.sep_token)
    result = [chunk.strip().split('/') for chunk in chunks if chunk.strip()]

    return result

# --- 7. MVP 결과 확인 ---
print("\n--- MVP 추론 테스트 ---")

# 테스트할 새로운 입력값
test_context = "카페"
test_input_text = "어떤 음료 주문 하시겠어요요?"

print(f"입력 상황: {test_context}")
print(f"입력 발화: {test_input_text}")

# 추천 함수 실행
recommended = recommend_chunks(test_context, test_input_text)

print("\n[추천된 AAC 청크]")
print(recommended)
print("--------------------")

# 또 다른 테스트
test_context_2 = "식당"
test_input_text_2 = "반찬 더 드릴까요?"

print(f"\n입력 상황: {test_context_2}")
print(f"입력 발화: {test_input_text_2}")
recommended_2 = recommend_chunks(test_context_2, test_input_text_2)
print("\n[추천된 AAC 청크]")
print(recommended_2)
print("--------------------")