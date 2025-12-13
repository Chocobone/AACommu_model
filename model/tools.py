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
        
        # [수정] 경로가 존재하는지 미리 확인
        if not self.data_dir.exists():
            print(f"❌ [에러] 지정된 데이터 경로를 찾을 수 없습니다: {self.data_dir}")
            return pd.DataFrame(pairs)
            
        # 설정된 매핑(폴더 규칙)마다 반복
        for dir_prefix, tag_name in self.dir_map.items():
            # [수정] 탐색 경로를 명시하여 로그를 개선
            print(f"🔍 '{dir_prefix}'로 시작하는 폴더를 '{self.data_dir}' 경로에서 찾는 중... (태그: {tag_name})")
            
            # 해당 접두사로 시작하는 폴더 찾기
            # rglob("*")으로 data_dir 하위의 모든 디렉토리를 재귀적으로 탐색합니다.
            target_dirs = [
                p for p in self.data_dir.rglob("*") 
                if p.is_dir() and p.name.startswith(dir_prefix)
            ]
            
            if not target_dirs:
                # [수정] 경로를 다시 안내하여 사용자에게 경로 문제 확인을 유도
                print(f"   ⚠️ '{dir_prefix}'로 시작하는 폴더를 찾지 못했습니다. 현재 탐색 경로: {self.data_dir}. 건너뜹니다.")
                continue
                
            # 파일 수집
            json_files = []
            for d in target_dirs:
                json_files.extend(list(d.glob('*.json')))
            
            print(f"   📂 발견: {len(target_dirs)}개 폴더, {len(json_files)}개 파일 -> 모두 '{tag_name}' 태그 적용")

            # 데이터 파싱
            for json_path in tqdm(json_files, desc=f"{tag_name} 데이터 파싱"):
                try:
                    # 'utf-8-sig'로 파일을 열 때 발생하는 에러 처리
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
                
                # [수정] 구체적인 예외 처리로 디버깅 용이성 향상
                except FileNotFoundError:
                    print(f"\n   ❌ 파일 로드 실패 (FileNotFoundError): {json_path}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"\n   ❌ JSON 파싱 오류 (JSONDecodeError): {json_path} - 에러 메시지: {e}")
                    continue
                except Exception as e:
                    print(f"\n   ❌ 기타 데이터 처리 오류: {json_path} - 에러 메시지: {e}")
                    continue
                
        return pd.DataFrame(pairs)