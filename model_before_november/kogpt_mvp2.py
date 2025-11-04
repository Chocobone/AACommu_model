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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

print("모델 및 토크나이저 준비 완료!")

# --- 4. 데이터셋 전처리 ---
df = pd.DataFrame(train_data_json)
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    full_texts = [f"{tokenizer.bos_token}{ctx} {tokenizer.sep_token} {inp} {tokenizer.sep_token} {out}{tokenizer.eos_token}"
                  for ctx, inp, out in zip(examples['context'], examples['input_text'], examples['output_text'])]
    return tokenizer(full_texts, padding="max_length", truncation=True, max_length=128)

processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print("데이터셋 전처리 완료!")

# --- 5. 모델 파인튜닝 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./kogpt2_aac_mvp_model",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=3,
)

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

FINAL_MODEL_PATH = "./final_kogpt2_aac_mvp_model"
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"최종 모델이 '{FINAL_MODEL_PATH}'에 저장되었습니다.")

# --- 6. 학습된 모델로 대화형(Interactive) 추론 실행 ---
def generate_next_chunks(model, tokenizer, context, input_text, current_sentence):
    """
    현재까지 완성된 문장을 바탕으로 다음 청크 선택지를 생성하는 함수
    """
    # 1. 추론을 위한 입력 프롬프트 생성
    prompt = f"{tokenizer.bos_token}{context} {tokenizer.sep_token} {input_text} {tokenizer.sep_token}{current_sentence}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_len = len(inputs.input_ids[0])

    # 2. 모델을 통해 다음 텍스트 생성
    outputs = model.generate(
        inputs.input_ids,
        max_length=input_ids_len + 30, # 현재 길이에 30토큰 정도만 더 생성하도록 제한
        num_beams=5,
        repetition_penalty=2.0,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 3. [수정] 입력 프롬프트 부분을 제외하고 순수하게 생성된 부분만 디코딩
    generated_ids = outputs[0][input_ids_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 4. 생성된 텍스트에서 첫 번째 청크 그룹만 분리하여 반환
    chunks = generated_text.split(tokenizer.sep_token)
    if chunks and chunks[0].strip():
        next_chunk_options = chunks[0].strip().split('/')
        return next_chunk_options
    else:
        return []

# --- 7. MVP 결과 확인 (대화형으로 수정) ---
print("\n--- MVP 대화형 추론 테스트 ---")

# 7.1. 추론을 위해 파인튜닝된 모델과 토크나이저를 다시 로드합니다.
# (루프 안에서 반복적으로 로드하는 것을 방지하기 위해 한 번만 실행)
tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(FINAL_MODEL_PATH)
model.to(device)
model.eval()

# 7.2. 테스트 상황 설정
test_context = "카페"
test_input_text = "안녕하세요, 주문 도와드릴까요?"

print(f"입력 상황: {test_context}")
print(f"입력 발화: {test_input_text}\n")

# 7.3. 대화 루프 시작
selected_chunks = []
while True:
    current_sentence = " ".join(selected_chunks)
    
    # 다음 추천 청크 생성
    next_options = generate_next_chunks(model, tokenizer, test_context, test_input_text, current_sentence)

    # 추천할 내용이 없거나, 문장이 완성되었다고 판단되면 루프 종료
    if not next_options or len(next_options) == 1 and next_options[0] in ["주세요", "입니다", "갈게요", "괜찮아요", "없어요"]:
        selected_chunks.extend(next_options)
        break
        
    print("다음 중 선택하세요:")
    for i, option in enumerate(next_options):
        print(f"{i+1}: {option}")
    
    # 사용자 선택
    try:
        choice_idx = int(input("선택 (번호): ")) - 1
        if 0 <= choice_idx < len(next_options):
            chosen_chunk = next_options[choice_idx]
            selected_chunks.append(chosen_chunk)
            print(f"\n>> 현재 문장: {' '.join(selected_chunks)}\n")
        else:
            print("잘못된 번호입니다. 다시 시도하세요.\n")
    except ValueError:
        print("숫자를 입력해주세요.\n")

# 7.4. 최종 결과 출력
final_sentence = " ".join(selected_chunks)
print("\n--- 최종 완성된 문장 ---")
print(final_sentence)
print("--------------------")