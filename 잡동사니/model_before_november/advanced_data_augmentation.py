import json
import itertools
from tqdm import tqdm

# --- 1. 설정 (Constants) ---
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
PAD = "<pad>"
SPECIAL_TOKENS = [BOS, EOS, PAD, SEP]

# 1단계에서 수동으로 구축한 청사진 데이터 파일
INPUT_FILE = "source_blueprints.json" 
# 2단계에서 생성될, 실제 훈련에 사용될 거대 데이터 파일
OUTPUT_FILE = "finetuning_dataset.json"

# --- 2. 핵심 로직 함수 ---

def parse_output_text(output_text):
    """
    "아이스/따뜻한 <sep> 아메리카노 <sep> 주세요" 문자열을
    [['아이스', '따뜻한'], ['아메리카노'], ['주세요']] 리스트로 변환
    """
    try:
        chunk_groups = []
        chunks = output_text.split(SEP)
        for chunk in chunks:
            options = [opt.strip() for opt in chunk.strip().split('/')]
            if options:
                chunk_groups.append(options)
        return chunk_groups
    except Exception as e:
        print(f"Error parsing output_text: {output_text} | Error: {e}")
        return None

def generate_exploded_data(blueprint):
    """
    하나의 청사진(blueprint)으로부터 모든 중간 단계 데이터를 생성
    """
    context = blueprint['context']
    input_text = blueprint['input_text']
    output_text = blueprint['output_text']
    
    chunk_groups = parse_output_text(output_text)
    if not chunk_groups:
        return []

    # 1. 청크 그룹에서 모든 가능한 '경로(path)'를 생성
    # 예: [('아이스', '아메리카노', '주세요'), ('따뜻한', '아메리카노', '주세요')]
    all_paths = list(itertools.product(*chunk_groups))

    generated_data = []

    # 2. 각 경로(path)에 대해 '중간 단계'를 모두 생성
    for path in all_paths:
        # 2-1. 첫 번째 단계 (아무것도 선택하지 않은 상태)
        # 이 단계는 모델이 '첫 번째 청크 그룹'을 예측하도록 학습
        initial_input = f"{BOS}{context}{SEP}{input_text}{SEP}"
        # (주의) 첫 번째 단계의 정답은 '단일 경로'가 아닌, '전체 선택지'가 포함된
        # 원본 output_text 여야 모델이 선택지를 학습할 수 있습니다.
        # (이 부분은 설계에 따라 달라질 수 있으나, 여기서는 원본을 따름)
        # initial_output = output_text + EOS
        
        # [논문 수준 수정] 
        # 첫 번째 단계부터 특정 경로를 따르도록 학습시킬 수도 있습니다.
        # 여기서는 '순차적 추론'을 위해 각 경로를 명시적으로 학습시킵니다.
        initial_output = f"{SEP.join(path)}{EOS}"
        generated_data.append({"input": initial_input, "output": initial_output})

        # 2-2. 두 번째 단계부터 마지막 단계까지 (중간 단계)
        for i in range(1, len(path) + 1):
            # i=1: current_path = ('아이스',)
            # i=2: current_path = ('아이스', '아메리카노')
            current_path_chunks = path[:i]
            # i=1: next_path_chunks = ('아메리카노', '주세요')
            # i=2: next_path_chunks = ('주세요',)
            next_path_chunks = path[i:]

            # 입력 프롬프트: <bos>C<sep>I<sep> (누적된 선택)
            current_sentence = " ".join(current_path_chunks)
            input_prompt = f"{BOS}{context}{SEP}{input_text}{SEP}{current_sentence} "

            # 출력 라벨: (남은 선택지)
            if next_path_chunks:
                output_label = f"{SEP.join(next_path_chunks)}{EOS}"
            else:
                # 경로의 끝에 도달하면 <eos>만 출력하도록 학습
                output_label = EOS

            generated_data.append({"input": input_prompt, "output": output_label})

    return generated_data

# --- 3. 메인 실행 로직 ---

def main():
    print(f"'{INPUT_FILE}'에서 청사진 데이터를 로드합니다...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            blueprints = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILE}'을 찾을 수 없습니다.")
        print("1단계: 'source_blueprints.json' 파일을 먼저 생성해야 합니다.")
        # MVP를 위해 임시 데이터 생성
        print("임시 MVP 데이터로 대신 실행합니다...")
        blueprints = [
            {"context": "카페", "input_text": "주문 도와드릴까요?", "output_text": "아이스/따뜻한 <sep> 아메리카노/라떼 <sep> 주세요"},
            {"context": "병원", "input_text": "어디가 불편하세요?", "output_text": "머리가/배가 <sep> 어제부터 <sep> 아파요"}
        ]
    except json.JSONDecodeError:
        print(f"Error: '{INPUT_FILE}'의 JSON 형식이 올바르지 않습니다.")
        return

    print(f"총 {len(blueprints)}개의 청사진을 로드했습니다.")
    
    final_training_data = []
    
    print("데이터 폭발(Explosion)을 시작합니다...")
    for bp in tqdm(blueprints, desc="Processing blueprints"):
        exploded = generate_exploded_data(bp)
        final_training_data.extend(exploded)

    print("\n데이터 폭발 완료!")
    print(f"총 {len(blueGPrints)}개의 청사진으로부터 {len(final_training_data)}개의 학습 데이터를 생성했습니다.")

    print(f"'{OUTPUT_FILE}' 파일로 저장합니다...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_training_data, f, ensure_ascii=False, indent=2)

    print("--- 완료 ---")
    print(f"이제 'professional_aac_mvp.py'의 학습 데이터를 '{OUTPUT_FILE}'로 변경하여 훈련을 진행하세요.")
    
    # 생성된 데이터 예시 출력
    if final_training_data:
        print("\n[생성된 데이터 예시 5개]")
        for item in final_training_data[:5]:
            print(item)

if __name__ == "__main__":
    main()
