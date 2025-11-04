# !pip install torch transformers datasets pandas scikit-learn


import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

print("라이브러리 로드 완료!")

# --- 2. 샘플 데이터셋 생성 (데이터 보강) ---
# MVP 학습을 위해 데이터를 20개로 늘려 모델이 더 많은 패턴을 학습하도록 합니다.
train_data_json = [
    {"context": "카페", "input_text": "안녕하세요, 주문 도와드릴까요?", "output_text": "<bos> 아이스/따뜻한 <sep> 아메리카노/카페라떼 <sep> 주세요 <eos>"},
    {"context": "카페", "input_text": "더 필요한 거 있으세요?", "output_text": "<bos> 아니요 <sep> 괜찮아요 <eos>"},
    {"context": "카페", "input_text": "포인트 적립이나 할인카드 있으신가요?", "output_text": "<bos> 아니요 <sep> 없어요 <eos>"},
    {"context": "카페", "input_text": "드시고 가세요? 아니면 포장이세요?", "output_text": "<bos> 먹고 갈게요/포장해주세요 <eos>"},
    {"context": "병원", "input_text": "어디가 불편해서 오셨어요?", "output_text": "<bos> 어제부터/오늘 아침부터 <sep> 머리가/배가 <sep> 아파요 <eos>"},
    {"context": "병원", "input_text": "성함이 어떻게 되세요?", "output_text": "<bos> 제 이름은 <sep> 김민준/이서연 <sep> 입니다 <eos>"},
    {"context": "병원", "input_text": "주민등록번호 앞자리를 말씀해주세요.", "output_text": "<bos> 900101/001231 <sep> 입니다 <eos>"},
    {"context": "병원", "input_text": "이전에 저희 병원 오신 적 있으세요?", "output_text": "<bos> 네, 있어요/아니요, 처음이에요 <eos>"},
    {"context": "식당", "input_text": "몇 분이세요?", "output_text": "<bos> 한 명/두 명/세 명 <sep> 이요 <eos>"},
    {"context": "식당", "input_text": "메뉴 뭐로 하시겠어요?", "output_text": "<bos> 김치찌개/된장찌개 <sep> 하나 <sep> 주세요 <eos>"},
    {"context": "식당", "input_text": "주문하신 메뉴 나왔습니다.", "output_text": "<bos> 네 <sep> 감사합니다 <eos>"},
    {"context": "식당", "input_text": "더 필요한 반찬 있으시면 말씀해주세요.", "output_text": "<bos> 김치/단무지 <sep> 좀 더 주세요 <eos>"},
    {"context": "편의점", "input_text": "봉투 필요하세요?", "output_text": "<bos> 네/아니요 <sep> 괜찮아요 <eos>"},
    {"context": "편의점", "input_text": "결제 도와드리겠습니다.", "output_text": "<bos> 카드로/현금으로 <sep> 할게요 <eos>"},
    {"context": "편의점", "input_text": "담배 찾으시는 거 있으세요?", "output_text": "<bos> 아니요/에쎄 체인지 <sep> 주세요 <eos>"},
    {"context": "편의점", "input_text": "따로 데워드릴까요?", "output_text": "<bos> 네, 데워주세요/아니요, 그냥 주세요 <eos>"},
    {"context": "관공서", "input_text": "어떤 업무 때문에 오셨나요?", "output_text": "<bos> 등본/인감 <sep> 떼러 왔어요 <eos>"},
    {"context": "관공서", "input_text": "신분증 좀 보여주시겠어요?", "output_text": "<bos> 네 <sep> 여기요 <eos>"},
    {"context": "관공서", "input_text": "이 서류에 서명 부탁드립니다.", "output_text": "<bos> 어디에 하면 되나요? <eos>"},
    {"context": "관공서", "input_text": "수수료는 500원입니다.", "output_text": "<bos> 네 <sep> 알겠습니다 <eos>"}
]


print("샘플 데이터 생성 완료!")

# --- 3. 모델 및 토크나이저 로드 ---
MODEL_NAME = "klue/bert-base"

# 3.1. 토크나이저 로드 및 특수 토큰 추가
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(special_tokens)

# 3.2. Encoder-Decoder 모델 로드
model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME)

# 3.3. 모델의 인코더와 디코더 각각의 임베딩 크기를 조정합니다. (오류 수정)
model.encoder.resize_token_embeddings(len(tokenizer))
model.decoder.resize_token_embeddings(len(tokenizer))

# 3.4. 모델 설정 (시작/끝 토큰 ID 설정)
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

print("모델 및 토크나이저 준비 완료!")

# --- 4. 데이터셋 전처리 ---
df = pd.DataFrame(train_data_json)
dataset = Dataset.from_pandas(df)

MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = [f"[CLS] {ctx} [SEP] {inp} [SEP]" for ctx, inp in zip(examples['context'], examples['input_text'])]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    
    labels = tokenizer(examples['output_text'], max_length=MAX_LENGTH, truncation=True, padding="max_length").input_ids
    
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != tokenizer.pad_token_id else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print("데이터셋 전처리 완료!")

# --- 5. 모델 파인튜닝 ---
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 학습 인자(Hyperparameters) 설정 (학습 시간 증가)
training_args = Seq2SeqTrainingArguments(
    output_dir="./aac_mvp_model",
    num_train_epochs=20,  # 데이터가 적으므로 학습 횟수를 20회로 늘립니다.
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=3,
)

# 트레이너 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
)

print("모델 파인튜닝을 시작합니다...")
trainer.train()
print("모델 파인튜닝 완료!")

# 최종 모델 저장
FINAL_MODEL_PATH = "./final_aac_mvp_model"
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"최종 모델이 '{FINAL_MODEL_PATH}'에 저장되었습니다.")


# --- 6. 학습된 모델로 추론(Inference) 실행 ---
def recommend_chunks(context, input_text, model_path=FINAL_MODEL_PATH):
    """
    파인튜닝된 모델을 로드하여 새로운 입력에 대한 AAC 청크를 추천하는 함수
    """
    # 1. 저장된 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval() # 추론 모드로 설정

    # 2. 입력 데이터 전처리
    input_sequence = f"[CLS] {context} [SEP] {input_text} [SEP]"
    inputs = tokenizer(input_sequence, return_tensors="pt").to(device)

    # 3. 모델을 통해 결과 생성 (반복 페널티 추가)
    outputs = model.generate(
        inputs.input_ids,
        max_length=MAX_LENGTH,
        num_beams=5,
        repetition_penalty=2.0, # 동일한 단어 반복을 억제하는 페널티를 추가합니다.
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 4. 생성된 토큰 ID를 텍스트로 변환 (버그 수정: special token 유지)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 5. 후처리: 텍스트를 2차원 배열(청크)로 변환
    # <bos>, <eos>, <pad> 등 불필요한 토큰 제거 후 <sep>으로 분리
    cleaned_text = generated_text.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
    chunks = cleaned_text.split(tokenizer.sep_token)
    # 각 청크의 앞뒤 공백을 제거하고, 빈 청크가 생성되는 것을 방지합니다.
    result = [chunk.strip().split('/') for chunk in chunks if chunk.strip()]

    return result

# --- 7. MVP 결과 확인 ---
print("\n--- MVP 추론 테스트 ---")

# 테스트할 새로운 입력값
test_context = "카페"
test_input_text = "네, 메뉴 나오셨습니다. 포장해드릴까요?"

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

