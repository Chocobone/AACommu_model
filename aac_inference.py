import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel
import numpy as np

# ==========================================
# 1. 모델 클래스 정의 (학습 때와 똑같아야 함)
# ==========================================

class AACommuModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super(AACommuModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

# ==========================================
# 2. 통합 추론 시스템 클래스 (GPT + BERT)
# ==========================================

class AACSystem:
    def __init__(self, bert_weight_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading AAC System on {self.device}...")

        # --- A. 생성 모델 (Generator): KoGPT2 ---
        # (만약 직접 학습시킨 GPT 가중치가 있다면 여기서 로드하세요)
        self.gpt_model_name = "skt/kogpt2-base-v2"
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model_name)
        self.gpt_model.to(self.device)
        self.gpt_model.eval()

        # --- B. 판별 모델 (Validator): BERT ---
        self.bert_model_name = "klue/bert-base"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        # 모델 구조 초기화 및 가중치 로드
        self.bert_model = AACommuModel(n_classes=2, model_name=self.bert_model_name)
        try:
            self.bert_model.load_state_dict(torch.load(bert_weight_path, map_location=self.device))
            print("✅ BERT Validator loaded successfully.")
        except FileNotFoundError:
            print(f"⚠️ Warning: {bert_weight_path} not found. Running with random weights.")
        
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # 현재 처리 중인 후보 문장들 저장소
        self.current_candidates = []

    def _score_sentence(self, question, answer):
        """BERT를 사용하여 (질문+답변) 쌍의 적절성 점수(0~1)를 계산"""
        text = f"{question} [SEP] {answer}"
        encoding = self.bert_tokenizer.encode_plus(
            text,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
            prob = torch.nn.functional.softmax(outputs, dim=1)
        
        return prob[0][1].item() # '적절함(1)'일 확률 반환

    def generate_and_filter(self, question, threshold=0.85):
        """
        1. GPT로 후보 문장 20개 생성
        2. BERT로 검수하여 점수 미달 탈락시킴
        3. 살아남은 문장들을 '어절 리스트'로 변환하여 저장
        """
        print(f"\n[Process] 질문 수신: '{question}'")
        
        # 1. GPT 생성 (Beam Search)
        input_text = f"<q>{question}</s><a>"
        input_ids = self.gpt_tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt_model.generate(
                input_ids,
                max_length=32,
                num_beams=20,             # 후보를 많이 뽑기 위해 빔 사이즈 키움
                num_return_sequences=20,  # 20개 문장 생성
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=self.gpt_tokenizer.eos_token_id
            )

        raw_candidates = []
        for out in outputs:
            decoded = self.gpt_tokenizer.decode(out, skip_special_tokens=False)
            try:
                # <q>...<a>... 파싱
                ans = decoded.split("<a>")[1].split("</s>")[0].strip()
                if ans: raw_candidates.append(ans)
            except:
                pass
        
        raw_candidates = list(set(raw_candidates)) # 중복 제거

        # 2. BERT 검수 (Re-ranking & Filtering)
        valid_candidates = []
        print(f" -> GPT 생성 후보 {len(raw_candidates)}개 검수 시작...")
        
        for cand in raw_candidates:
            score = self._score_sentence(question, cand)
            if score >= threshold:
                valid_candidates.append((cand, score))
                # print(f"    (O) {cand} [점수: {score:.4f}]")
            else:
                pass
                # print(f"    (X) {cand} [점수: {score:.4f}] - 탈락")
        
        # 점수 높은 순으로 정렬
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 최종 후보 저장 (어절 단위로 분리)
        self.current_candidates = [item[0].split() for item in valid_candidates]
        print(f" -> 최종 통과 문장: {len(self.current_candidates)}개")
        
        return [c[0] for c in valid_candidates] # 디버깅용 문장 리스트 반환

    def get_next_chunks(self, step_idx, current_selection_path):
        """
        사용자의 현재 선택(step)에 맞춰 다음 추천 단어(청크)들을 반환
        - step_idx: 현재 몇 번째 어절을 고를 차례인지
        - current_selection_path: 지금까지 고른 단어 리스트 (예: ['아이스', '아메리카노'])
        """
        recommendations = set()
        
        for sent_chunks in self.current_candidates:
            # 1. 문장 길이가 현재 단계보다 길어야 함
            if len(sent_chunks) <= step_idx:
                continue
                
            # 2. 지금까지 사용자가 고른 단어들과 문장의 앞부분이 일치해야 함
            match = True
            for i, selected_word in enumerate(current_selection_path):
                if sent_chunks[i] != selected_word:
                    match = False
                    break
            
            # 3. 조건이 맞으면 다음 단어를 추천 리스트에 추가
            if match:
                recommendations.add(sent_chunks[step_idx])
                
        return sorted(list(recommendations))

# ==========================================
# 3. 실행 예시 (메인 함수)
# ==========================================

if __name__ == "__main__":
    # 시스템 로드
    aac = AACSystem(bert_weight_path="AACommu_model_best.pt")

    # --- 시나리오 시작 ---
    user_q = "주문 도와드릴까요?"
    
    # 1. 질문이 들어오면 후보군 생성 및 검수
    passed_sentences = aac.generate_and_filter(user_q, threshold=0.90) # 0.9점 이상만 통과
    
    if not passed_sentences:
        print("⚠️ 적절한 답변을 생성하지 못했습니다. 기준을 낮추거나 다시 시도하세요.")
    else:
        # 2. 단계별 청크 선택 시뮬레이션
        current_path = [] # 사용자가 선택한 단어들
        
        for step in range(5): # 최대 5어절까지 해봄
            # 다음 추천 단어 가져오기
            next_chunks = aac.get_next_chunks(step, current_path)
            
            if not next_chunks:
                print("✅ 문장 완성!")
                break
                
            print(f"\n[Step {step+1}] 추천 단어: {next_chunks}")
            
            # (사용자 입력 가정) 첫 번째 추천 단어를 선택한다고 가정
            user_choice = next_chunks[0]
            print(f"👉 사용자 선택: '{user_choice}'")
            current_path.append(user_choice)
            
        print(f"\n🎉 최종 문장: {' '.join(current_path)}")
