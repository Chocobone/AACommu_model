import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel
import numpy as np

# 1. 모델 클래스 정의 (BERT) - 학습 코드와 동일
class AACommuModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super(AACommuModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

# 2. 통합 추론 시스템
class AACSystem:
    def __init__(self, bert_weight_path, gpt_weight_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading AAC System on {self.device}...")

        # --- A. GPT 로드 (수정된 부분) ---
        self.gpt_model_name = "skt/kogpt2-base-v2"
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token # 패딩 설정 필수
        
        self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model_name)
        
        # [핵심] 학습시킨 GPT 가중치 로드!
        try:
            self.gpt_model.load_state_dict(torch.load(gpt_weight_path, map_location=self.device))
            print("✅ KoGPT2 (생성 모델) 로드 완료.")
        except FileNotFoundError:
            print(f"🚨 오류: {gpt_weight_path} 파일이 없습니다! GPT 학습을 먼저 진행하세요.")
            
        self.gpt_model.to(self.device)
        self.gpt_model.eval()

        # --- B. BERT 로드 ---
        self.bert_model_name = "klue/bert-base"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        self.bert_model = AACommuModel(n_classes=2, model_name=self.bert_model_name)
        try:
            self.bert_model.load_state_dict(torch.load(bert_weight_path, map_location=self.device))
            print("✅ BERT (검수 모델) 로드 완료.")
        except FileNotFoundError:
            print(f"🚨 오류: {bert_weight_path} 파일이 없습니다!")
        
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        self.current_candidates = []

    def _score_sentence(self, question, answer):
        text = f"{question} [SEP] {answer}"
        encoding = self.bert_tokenizer.encode_plus(
            text, max_length=128, add_special_tokens=True, return_token_type_ids=False,
            padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
            prob = torch.nn.functional.softmax(outputs, dim=1)
        return prob[0][1].item()

    def generate_and_filter(self, question, threshold=0.85):
        print(f"\n[Process] 질문: '{question}'")
        
        # GPT 입력 생성 (<q>질문</s><a>)
        input_text = f"<q>{question}</s><a>"
        input_ids = self.gpt_tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt_model.generate(
                input_ids, max_length=32, num_beams=10, num_return_sequences=10,
                no_repeat_ngram_size=2, early_stopping=True,
                eos_token_id=self.gpt_tokenizer.eos_token_id,
                pad_token_id=self.gpt_tokenizer.eos_token_id
            )

        raw_candidates = []
        for out in outputs:
            decoded = self.gpt_tokenizer.decode(out, skip_special_tokens=False)
            if "<a>" in decoded:
                ans = decoded.split("<a>")[1].split("</s>")[0].strip()
                if ans: raw_candidates.append(ans)
        
        raw_candidates = list(set(raw_candidates))

        # BERT 검수
        valid_candidates = []
        print(f" -> 후보 {len(raw_candidates)}개 검수 중...")
        for cand in raw_candidates:
            score = self._score_sentence(question, cand)
            if score >= threshold:
                valid_candidates.append((cand, score))
        
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        self.current_candidates = [item[0].split() for item in valid_candidates]
        print(f" -> 최종 통과: {len(self.current_candidates)}개")
        
        for vc in valid_candidates[:3]: # 상위 3개 출력 확인
            print(f"    (통과) {vc[0]} [{vc[1]:.2f}]")

        return [c[0] for c in valid_candidates]

    def get_next_chunks(self, step_idx, current_selection_path):
        recommendations = set()
        for sent_chunks in self.current_candidates:
            if len(sent_chunks) <= step_idx: continue
            
            match = True
            for i, selected_word in enumerate(current_selection_path):
                if sent_chunks[i] != selected_word:
                    match = False
                    break
            
            if match:
                recommendations.add(sent_chunks[step_idx])
        return sorted(list(recommendations))

if __name__ == "__main__":
    # 두 개의 학습된 파일 경로를 모두 넣어줍니다.
    aac = AACSystem(
        bert_weight_path="AACommu_model_best.pt", 
        gpt_weight_path="AAC_KoGPT2_best.pt"
    )

    user_q = "주문 도와드릴까요?"
    passed = aac.generate_and_filter(user_q, threshold=0.8) # 문턱값 조절 가능

    if passed:
        curr = []
        for step in range(5):
            chunks = aac.get_next_chunks(step, curr)
            if not chunks: 
                print("✅ 완성!")
                break
            print(f"\n[Step {step+1}] 추천: {chunks}")
            choice = chunks[0] # 자동 선택 예시
            print(f"👉 선택: {choice}")
            curr.append(choice)
        print(f"🎉 결과: {' '.join(curr)}")
    else:
        print("⚠️ 적절한 답변을 생성하지 못했습니다.")