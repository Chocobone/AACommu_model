import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class AACDataProcessor:
    def __init__(self, data_dir, DIR_CATEGORY_MAP):
        self.data_dir = Path(data_dir)
        self.dir_map = DIR_CATEGORY_MAP
        
    def load_data(self):
        pairs = []
        
        # 설정된 매핑(폴더 규칙)마다 반복
        for dir_prefix, tag_name in self.dir_map.items():
            print(f"🔍 '{dir_prefix}'로 시작하는 폴더를 찾는 중... (태그: {tag_name})")
            
            # 해당 접두사로 시작하는 폴더 찾기
            target_dirs = [
                p for p in self.data_dir.rglob("*") 
                if p.is_dir() and p.name.startswith(dir_prefix)
            ]
            
            if not target_dirs:
                print(f"   ⚠️ '{dir_prefix}'로 시작하는 폴더를 찾지 못했습니다. 건너뜁니다.")
                continue
                
            # 파일 수집
            json_files = []
            for d in target_dirs:
                json_files.extend(list(d.glob('*.json')))
            
            print(f"   📂 발견: {len(target_dirs)}개 폴더, {len(json_files)}개 파일 -> 모두 '{tag_name}' 태그 적용")

            # 데이터 파싱
            for json_path in tqdm(json_files, desc=f"{tag_name} 데이터 파싱"):
                try:
                    with open(json_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                    
                    if 'video' not in data: continue
                    
                    # [핵심] JSON 내부 장소 정보는 무시하고, 폴더 규칙에 따른 태그 강제 할당
                    current_tag = f"<LOC_{tag_name}>"

                    interactions = data['video'].get('interactions', [])
                    for interaction in interactions:
                        human_text = ""
                        if 'human_event' in interaction and 'utterances' in interaction['human_event']:
                            utts = interaction['human_event']['utterances']
                            if utts: human_text = utts[0].get('utterance_cap', '').strip()
                        
                        robot_text = ""
                        if 'robot_response' in interaction:
                            resps = interaction['robot_response']
                            if resps: robot_text = resps[0].get('answer', '').strip()
                        
                        if human_text and robot_text:
                            pairs.append({
                                "place_tag": current_tag,
                                "q": robot_text, 
                                "a": human_text
                            })
                except Exception:
                    continue
                
        return pd.DataFrame(pairs)