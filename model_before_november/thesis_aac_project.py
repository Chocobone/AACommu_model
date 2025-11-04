import torch
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import warnings
import json
import zipfile # <-- zipfile 모듈 추가
import os

warnings.filterwarnings("ignore")
print("라이브러리 로드 완료!")

# --- 1. 사전 학습된 모델 및 토크나이저 선정 ---
MODEL_NAME = "skt/kogpt2-base-v2"
# 🚨 수정: ZIP 파일들이 있는 디렉토리 경로로 변경
TRAINING_DIR = "/local_datasets/AACommu/Training/02.라벨링데이터/"
VALIDATION_DIR = "/local_datasets/AACommu/Validation/02.라벨링데이터/" # 검증 데이터 경로도 유사할 것으로 가정
FINAL_MODEL_PATH = "/data/yho7374/repos/AACommu_model/thesis_aac_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

print("Chapter 1: 모델 및 토크나이저 준비 완료!")

# --- 2. 데이터 로딩 함수 (ZIP 파일 처리로 변경) ---
def load_data_from_zip_dir(data_dir):
    """
    지정된 디렉토리 내의 모든 ZIP 파일을 열어, 그 안에 있는 모든 JSON 파일을 읽어 통합합니다.
    """
    all_data = []
    
    try:
        # 디렉토리 내 모든 파일 순회
        for filename in os.listdir(data_dir):
            if filename.endswith('.zip'):
                zip_path = os.path.join(data_dir, filename)
                print(f"   ZIP 파일 '{filename}'을 엽니다...")
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # ZIP 파일 내부의 모든 멤버를 순회
                    for member_name in zf.namelist():
                        # .json 파일만 처리
                        if member_name.endswith('.json'):
                            # zf.open()을 사용하여 파일 객체를 얻고 바로 읽습니다.
                            with zf.open(member_name) as f:
                                data = f.read().decode('utf-8-sig')
                                
                                # JSON 로딩 및 통합
                                json_data = json.loads(data)
                                
                                if isinstance(json_data, list):
                                    all_data.extend(json_data)
                                else:
                                    all_data.append(json_data)

                                print(f"      [LOAD] {member_name} 로드 완료.")
        
        return all_data
    
    except FileNotFoundError:
        print(f"Error: '{data_dir}' 디렉토리를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
    except Exception as e:
        print(f"Error processing data in {data_dir}: {e}")
        exit()

# --- 2. 학습 데이터셋 로드 및 통합 ---
print(f"Chapter 2: '{TRAINING_DIR}' 및 '{VALIDATION_DIR}'에서 학습 데이터 로드 시작!")

# 학습 데이터 로드
training_data = load_data_from_zip_dir(TRAINING_DIR)
print(f"총 {len(training_data)}개의 학습 데이터 레코드를 로드했습니다.")
train_df = pd.DataFrame(training_data)

# 검증 데이터 로드
validation_data = load_data_from_zip_dir(VALIDATION_DIR)
print(f"총 {len(validation_data)}개의 검증 데이터 레코드를 로드했습니다.")
eval_df = pd.DataFrame(validation_data)

# 🚨 중요: 라벨링 데이터의 필드 확인 및 맞춤 (이하 동일)
# ... (이하 코드는 원본과 동일)

if 'input' not in train_df.columns or 'output' not in train_df.columns:
    print("\n⚠️ 경고: 로드된 데이터프레임에 'input' 또는 'output' 필드가 없습니다.")
    print("      원본 JSON 파일 구조를 확인하여 'preprocess_function' 및 'df' 생성 부분을 수정해야 합니다.")
    print(f"      현재 데이터프레임의 컬럼: {train_df.columns.tolist()}")
    
raw_datasets = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'eval': Dataset.from_pandas(eval_df)
})

# --- 3. 데이터셋 전처리 및 분리 ---

def preprocess_function(examples):
    # 'input'과 'output' 필드를 연결하여 모델의 입력 시퀀스를 생성
    full_texts = [inp + out for inp, out in zip(examples['input'], examples['output'])]
    return tokenizer(full_texts, padding="max_length", truncation=True, max_length=128)

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names,
)
print("Chapter 3: 데이터셋 전처리 및 분리 완료!")

# --- 4. 모델 파인튜닝 (이하 동일) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./thesis_aac_model_checkpoints",
    num_train_epochs=10, 
    per_device_train_batch_size=8, 
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir='./logs',
    logging_steps=100, 
    save_total_limit=2, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Chapter 4: 모델 파인튜닝을 시작합니다...")
trainer.train()
print("모델 파인튜닝 완료!")

trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"최고 성능의 모델이 '{FINAL_MODEL_PATH}'에 저장되었습니다.")

# --- 5. 대화형 추론 시스템 실행 (핵심 기능) ---
print("\n[추론 시스템 로딩 중...]")
inference_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
inference_model = GPT2LMHeadModel.from_pretrained(FINAL_MODEL_PATH)
inference_model.to(device)
inference_model.eval()

# '무형문화재' 같은 문맥과 전혀 상관없는 단어 생성을 억제하기 위한 ID 리스트
bad_words = ["문화재", "유형", "무형", "국보", "보물", "아파요", "병원"] 
bad_words_ids = [inference_tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]
bad_words_ids = [item for sublist in bad_words_ids for item in sublist]


def generate_next_chunk(context, input_text, current_sentence):
    prompt = f"{inference_tokenizer.bos_token}{context}{inference_tokenizer.sep_token}{input_text}{inference_tokenizer.sep_token}{current_sentence}"
    inputs = inference_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_len = len(inputs.input_ids[0])

    generation_params = {
        "max_length": input_ids_len + 30,
        "num_beams": 5,
        "repetition_penalty": 2.5, 
        "early_stopping": True,
        "eos_token_id": inference_tokenizer.eos_token_id,
        "pad_token_id": inference_tokenizer.pad_token_id,
        "no_repeat_ngram_size": 2, 
    }
    
    if context == "카페":
        generation_params["bad_words_ids"] = bad_words_ids

    
    outputs = inference_model.generate(inputs.input_ids, **generation_params)

    generated_ids = outputs[0][input_ids_len:]
    generated_text = inference_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    chunks = generated_text.split(inference_tokenizer.sep_token)
    if chunks and chunks[0].strip():
        next_chunk_options = chunks[0].strip().split('/')
        return next_chunk_options
    else:
        return []

def run_interactive_inference():
    print("\n--- Chapter 5: AAC 논문 수준 대화형 추론 테스트 (안정화) ---")
    
    test_context = "카페"
    test_input_text = "주문 도와드릴까요?"
    
    print(f"입력 상황: {test_context}")
    print(f"입력 발화: {test_input_text}\n")
    
    selected_chunks = []
    
    while True:
        current_sentence = " ".join(selected_chunks)
        if current_sentence:
            current_sentence += " "
            
        next_options = generate_next_chunk(test_context, test_input_text, current_sentence)

        if not next_options or len(selected_chunks) > 7:
            if not next_options:
                print("[모델] 추천할 내용이 더 이상 없습니다.")
            break
        
        if len(next_options) == 1 and next_options[0] in ["주세요", "입니다", "갈게요", "괜찮아요", "없어요", "감사합니다", "아파요", "이요", "할게요", "알겠습니다", "내릴게요", "세워주세요"]:
            print(f"[모델] 문장 종결 토큰 '{next_options[0]}'이(가) 감지되었습니다.")
            selected_chunks.append(next_options[0])
            break
            
        print("다음 중 선택하세요:")
        for i, option in enumerate(next_options):
            print(f"{i+1}: {option}")
        
        direct_input_option_num = len(next_options) + 1
        print(f"{direct_input_option_num}: (직접 입력)")
        
        try:
            choice_idx = int(input(f"선택 (1-{direct_input_option_num}): ")) - 1
            
            if 0 <= choice_idx < len(next_options):
                chosen_chunk = next_options[choice_idx]
                selected_chunks.append(chosen_chunk)
                print(f"\n>> 현재 문장: {' '.join(selected_chunks)}\n")
            
            elif choice_idx == len(next_options):
                custom_chunk = input("직접 입력할 텍스트: ")
                selected_chunks.append(custom_chunk)
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