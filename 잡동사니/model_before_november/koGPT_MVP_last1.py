import torch
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

print("라이브러리 로드 완료!")

# --- Chapter 1. 사전 학습된 모델 및 토크나이저 선정 ---
# Transformer의 Decoder 아키텍처를 기반으로 한 skt/kogpt2-base-v2는
# '다음 단어 예측' (Causal Language Modeling)에 특화되어 있어 본 생성 과업에 가장 적합합니다.
MODEL_NAME = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

print("Chapter 1: 모델 및 토크나이저 준비 완료!")

# --- Chapter 2. 학습 데이터셋 구축 ---
# 실제 프로젝트에서는 수백~수천 개의 데이터를 사용해야 합니다.
# MVP에서는 데이터 분리 및 평가 과정을 보여주기 위해 샘플 데이터 수를 늘립니다.
train_data_json = [
    {"context": "카페", "input_text": "안녕하세요, 주문 도와드릴까요?", "output_text": "아이스/따뜻한 <sep> 아메리카노/카페라떼 <sep> 주세요"},
    {"context": "카페", "input_text": "더 필요한 거 있으세요?", "output_text": "아니요 <sep> 괜찮아요"},
    {"context": "카페", "input_text": "포인트 적립이나 할인카드 있으신가요?", "output_text": "아니요 <sep> 없어요"},
    {"context": "카페", "input_text": "드시고 가세요? 아니면 포장이세요?", "output_text": "먹고 갈게요/포장해주세요"},
    {"context": "카페", "input_text": "저기요, 주문할게요.", "output_text": "네 <sep> 무엇으로/어떤걸로 <sep> 도와드릴까요?"},
    {"context": "병원", "input_text": "어디가 불편해서 오셨어요?", "output_text": "어제부터/오늘 아침부터 <sep> 머리가/배가 <sep> 아파요"},
    {"context": "병원", "input_text": "성함이 어떻게 되세요?", "output_text": "제 이름은 <sep> 김민준/이서연 <sep> 입니다"},
    {"context": "병원", "input_text": "주민등록번호 앞자리를 말씀해주세요.", "output_text": "900101/001231 <sep> 입니다"},
    {"context": "병원", "input_text": "이전에 저희 병원 오신 적 있으세요?", "output_text": "네, 있어요/아니요, 처음이에요"},
    {"context": "병원", "input_text": "진료 다 끝나셨습니다. 처방전 받아가세요.", "output_text": "네 <sep> 감사합니다"},
    {"context": "식당", "input_text": "몇 분이세요?", "output_text": "한 명/두 명/세 명 <sep> 이요"},
    {"context": "식당", "input_text": "메뉴 뭐로 하시겠어요?", "output_text": "김치찌개/된장찌개 <sep> 하나 <sep> 주세요"},
    {"context": "식당", "input_text": "주문하신 메뉴 나왔습니다.", "output_text": "네 <sep> 감사합니다"},
    {"context": "식당", "input_text": "더 필요한 반찬 있으시면 말씀해주세요.", "output_text": "김치/단무지 <sep> 좀 더 주세요"},
    {"context": "식당", "input_text": "계산 도와드리겠습니다.", "output_text": "네 <sep> 카드로/현금으로 <sep> 할게요"},
]
print("Chapter 2: 학습 데이터셋 샘플 준비 완료!")


# --- Chapter 3. 데이터셋 전처리 파이프라인 ---
df = pd.DataFrame(train_data_json)

# 3.1. 텍스트 정제 및 정규화 (Normalization)
def normalize_text(text):
    # 불필요한 공백 및 특수문자 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['input_text'] = df['input_text'].apply(normalize_text)
# 여기서 더 나아가 konlpy 등을 사용한 어간/표제어 추출, 불용어 처리 등을 추가할 수 있으나,
# Transformer 모델은 문맥 자체를 학습하므로 과도한 변형은 오히려 성능 저하를 유발할 수 있습니다.

# 3.2. 데이터 분리 (Train / Validation Split)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

raw_datasets = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'eval': Dataset.from_pandas(eval_df)
})

# 3.3. 토큰화 함수 정의
def preprocess_function(examples):
    full_texts = [f"{tokenizer.bos_token}{ctx} {tokenizer.sep_token} {inp} {tokenizer.sep_token} {out}{tokenizer.eos_token}"
                  for ctx, inp, out in zip(examples['context'], examples['input_text'], examples['output_text'])]
    return tokenizer(full_texts, padding="max_length", truncation=True, max_length=128)

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names,
)
print("Chapter 3: 데이터셋 전처리 및 분리 완료!")


# --- Chapter 4. 모델 파인튜닝 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4.1. 평가 기반 학습을 위한 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./professional_aac_model",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    learning_rate=5e-5,          # Adam Optimizer의 학습률
    weight_decay=0.01,
    # --- 고도화된 부분 (오류 수정) ---
    # 구버전 transformers 라이브러리와의 호환성을 위해 'evaluation_strategy'를 'eval_strategy'로 변경합니다.
    eval_strategy="epoch",
    save_strategy="epoch",       # 매 에폭마다 모델 저장
    load_best_model_at_end=True, # 훈련 종료 후 최고 성능 모델 로드
    metric_for_best_model="eval_loss", # 최고 모델 선정 기준
    greater_is_better=False,     # loss는 낮을수록 좋음
    # --------------------
    logging_dir='./logs',
    logging_steps=10,
)

# 4.2. 훈련 및 검증 데이터셋을 모두 포함한 Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"], # 검증 데이터셋 추가
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Chapter 4: 모델 파인튜닝을 시작합니다...")
trainer.train()
print("모델 파인튜닝 완료!")

FINAL_MODEL_PATH = "./final_professional_aac_model"
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"최고 성능의 모델이 '{FINAL_MODEL_PATH}'에 저장되었습니다.")


# --- Chapter 5. 대화형 추론 시스템 실행 ---
inference_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
inference_model = GPT2LMHeadModel.from_pretrained(FINAL_MODEL_PATH)
inference_model.to(device)
inference_model.eval()

def generate_next_chunks(context, input_text, current_sentence, strategy='beam'):
    prompt = f"{inference_tokenizer.bos_token}{context} {inference_tokenizer.sep_token} {input_text} {inference_tokenizer.sep_token}{current_sentence}"
    inputs = inference_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_len = len(inputs.input_ids[0])

    # 5.1. 다양한 생성 전략 적용
    generation_params = {
        "max_length": input_ids_len + 30,
        "early_stopping": True,
        "eos_token_id": inference_tokenizer.eos_token_id,
        "pad_token_id": inference_tokenizer.pad_token_id,
        "repetition_penalty": 2.0,
    }

    if strategy == 'beam':
        # Beam Search: 문법적으로 안정적이고 완성도 높은 결과물 생성
        generation_params["num_beams"] = 5
    elif strategy == 'sampling':
        # Top-k / Top-p Sampling: 더 다양하고 창의적인 결과물 생성
        generation_params["do_sample"] = True
        generation_params["top_k"] = 50
        generation_params["top_p"] = 0.95

    outputs = inference_model.generate(inputs.input_ids, **generation_params)

    generated_ids = outputs[0][input_ids_len:]
    generated_text = inference_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    chunks = generated_text.split(inference_tokenizer.sep_token)
    if chunks and chunks[0].strip():
        return chunks[0].strip().split('/')
    else:
        return []

def run_interactive_inference():
    print("\n--- Chapter 5: 프로페셔널 MVP 대화형 추론 테스트 ---")
    
    test_context = "병원"
    test_input_text = "어디가 아파서 오셨어요?"
    
    print(f"입력 상황: {test_context}")
    print(f"입력 발화: {test_input_text}\n")
    
    selected_chunks = []
    while True:
        current_sentence = " ".join(selected_chunks)
        
        # 'beam' 또는 'sampling' 전략 선택 가능
        next_options = generate_next_chunks(test_context, test_input_text, current_sentence, strategy='beam')

        if not next_options or len(selected_chunks) > 5: # 무한루프 방지
            if next_options: selected_chunks.extend(next_options)
            break
        
        # 문장 종결 토큰으로 루프 종료
        if len(next_options) == 1 and next_options[0] in ["주세요", "입니다", "갈게요", "괜찮아요", "없어요", "감사합니다", "아파요", "이요"]:
            selected_chunks.extend(next_options)
            break
            
        print("다음 중 선택하세요:")
        for i, option in enumerate(next_options):
            print(f"{i+1}: {option}")
        
        try:
            choice_idx = int(input("선택 (번호): ")) - 1
            if 0 <= choice_idx < len(next_options):
                chosen_chunk = next_options[choice_idx]
                selected_chunks.append(chosen_chunk)
                print(f"\n>> 현재 문장: {' '.join(selected_chunks)}\n")
            else:
                print("잘못된 번호입니다.\n")
        except (ValueError, IndexError):
            print("유효한 숫자를 입력해주세요.\n")

    final_sentence = " ".join(selected_chunks)
    print("\n--- 최종 완성된 문장 ---")
    print(f"'{final_sentence}'")
    print("--------------------")

if __name__ == "__main__":
    run_interactive_inference()