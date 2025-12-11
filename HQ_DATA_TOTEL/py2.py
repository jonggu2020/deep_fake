import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk # (pip install Pillow)
import os
import pandas as pd

# --- 1. 사용자 설정 ---

# ⚠️ [수정필요 1]
# 1차 필터링 스크립트가 생성한 '의심' 목록 텍스트 파일
SUSPECT_LIST_FILE = "./suspect_list.txt" 

# ⚠️ [수정필요 2]
# 원본 CSV 파일 (v9 스크립트로 생성한 최신 CSV)
# 예: "./master_summary_v11_cleaned_final.csv"
ORIGINAL_CSV_FILE = "./master_summary_v11_cleaned_final.csv" 

# ⚠️ [수정필요 3]
# NPY 파일이 있는 폴더 경로
NPY_DIR = "./2_npy_timeseries"

# ⚠️ [수정필요 4]
# PNG 파일이 있는 폴더 경로
PNG_DIR = "./3_audio_spectrograms"

# ⚠️ [출력]
# 최종적으로 노이즈가 제거된 CSV가 저장될 *새* 경로
FINAL_OUTPUT_CSV = "./master_summary_v12_audio_cleaned.csv"

# ---

class ImageReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("오디오(PNG) 노이즈 수동 검수")
        
        # 데이터 로드
        try:
            with open(SUSPECT_LIST_FILE, 'r') as f:
                self.suspect_ids = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            messagebox.showerror("오류", f"'{SUSPECT_LIST_FILE}'을(를) 찾을 수 없습니다.\n1차 필터링 스크립트를 먼저 실행하세요.")
            self.root.destroy()
            return

        if not self.suspect_ids:
            messagebox.showinfo("완료", "검토할 '의심' 이미지가 없습니다.")
            self.root.destroy()
            return
            
        self.total_count = len(self.suspect_ids)
        self.current_index = 0
        self.delete_list = [] # 삭제하기로 결정한 ID 목록

        # --- GUI 위젯 설정 ---
        
        # 1. 진행 상황 레이블
        self.progress_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.progress_label.pack(pady=10)
        
        # 2. 이미지 이름 레이블
        self.filename_label = tk.Label(root, text="", font=("Helvetica", 10, "bold"))
        self.filename_label.pack(pady=5)

        # 3. 이미지 표시용 캔버스 (크기 조절)
        self.canvas = tk.Canvas(root, width=600, height=300, bg="black")
        self.canvas.pack(padx=20, pady=10)

        # 4. 버튼 프레임
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=20)

        # 5. '보류' 버튼 (초록색)
        self.keep_button = tk.Button(
            self.button_frame, 
            text="➡️ 보류 (Keep)", 
            font=("Helvetica", 14, "bold"),
            width=15, 
            command=self.keep_image,
            bg="#4CAF50", 
            fg="white"
        )
        self.keep_button.pack(side=tk.LEFT, padx=15)

        # 6. '삭제' 버튼 (빨간색)
        self.delete_button = tk.Button(
            self.button_frame, 
            text="🗑️ 삭제 (Delete)", 
            font=("Helvetica", 14, "bold"),
            width=15, 
            command=self.delete_image,
            bg="#F44336", 
            fg="white"
        )
        self.delete_button.pack(side=tk.RIGHT, padx=15)
        
        # 키보드 바인딩 (왼쪽 화살표 = 보류, 오른쪽/스페이스 = 삭제)
        self.root.bind('<Left>', lambda e: self.keep_image())
        self.root.bind('<Right>', lambda e: self.delete_image())
        self.root.bind('<space>', lambda e: self.delete_image())
        
        # 창 닫기 이벤트(프로토콜) 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 첫 번째 이미지 로드
        self.load_image()

    def load_image(self):
        if self.current_index >= self.total_count:
            self.finish_review()
            return

        # 1. 진행 상황 업데이트
        progress_text = f"검수 진행: {self.current_index + 1} / {self.total_count}"
        self.progress_label.config(text=progress_text)
        
        # 2. 파일명 업데이트
        video_id = self.suspect_ids[self.current_index]
        self.filename_label.config(text=video_id)
        
        # 3. 이미지 로드 및 리사이징 (Pillow 사용)
        png_path = os.path.join(PNG_DIR, f"{video_id}.png")
        
        try:
            img = Image.open(png_path)
            
            # 캔버스 크기(600x300)에 맞게 이미지 리사이징
            img.thumbnail((600, 300), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all") # 이전 이미지 삭제
            self.canvas.create_image(300, 150, image=self.photo, anchor=tk.CENTER)
            
        except Exception as e:
            print(f"오류: {video_id}.png 로드 실패: {e}")
            # 이미지 로드 실패 시 자동으로 '보류' 처리하고 다음으로 넘어감
            self.keep_image() 

    def keep_image(self):
        # '보류' 목록에는 추가할 필요 없음. 그냥 다음으로 넘어감.
        self.current_index += 1
        self.load_image()

    def delete_image(self):
        # '삭제' 목록에 현재 ID 추가
        video_id = self.suspect_ids[self.current_index]
        self.delete_list.append(video_id)
        
        print(f"삭제 목록 추가: {video_id}") # 터미널에 로그 출력
        
        self.current_index += 1
        self.load_image()

    def finish_review(self):
        # 모든 검토가 끝났을 때
        messagebox.showinfo("검토 완료", f"총 {self.total_count}개의 검토가 완료되었습니다.\n\n"
                                      f"삭제하기로 결정한 파일: {len(self.delete_list)} 개\n\n"
                                      "이제 최종 파일 정리를 시작합니다.")
        self.root.destroy() # GUI 창 닫기
        self.execute_final_cleanup() # 실제 파일 처리 함수 호출

    def on_closing(self):
        # 윈도우의 'X' 버튼을 눌렀을 때
        if messagebox.askokcancel("종료 확인", "아직 검토가 끝나지 않았습니다.\n"
                                           "지금 종료하면 현재까지의 진행 상황이 저장되지 않습니다.\n"
                                           "(삭제 목록이 처리되지 않습니다)\n\n정말로 종료하시겠습니까?"):
            self.root.destroy() # 저장 없이 강제 종료

    def execute_final_cleanup(self):
        # --- (최종 정리 작업) ---
        
        print("\n" + "="*70)
        print(f"PART 3: 최종 파일 정리 작업 시작")
        print(f"         (총 {len(self.delete_list)}개 ID 삭제)")
        print("="*70)

        ids_to_delete_set = set(self.delete_list)

        if not ids_to_delete_set:
            print("삭제할 항목이 없습니다. 작업을 종료합니다.")
            return

        # 1. CSV 파일 처리
        try:
            df = pd.read_csv(ORIGINAL_CSV_FILE)
            rows_before = len(df)
            
            # 'video_id'가 삭제 목록(Set)에 *없는* 행만 남김
            df_cleaned = df[~df['video_id'].isin(ids_to_delete_set)]
            rows_after = len(df_cleaned)
            
            df_cleaned.to_csv(FINAL_OUTPUT_CSV, index=False, encoding='utf-8-sig')
            
            print(f"✓ 1. CSV 파일 처리 완료.")
            print(f"  - 원본 CSV 행: {rows_before}")
            print(f"  - 삭제된 행: {rows_before - rows_after}")
            print(f"  - 최종 CSV 행: {rows_after}")
            print(f"  - 새 파일 저장: '{FINAL_OUTPUT_CSV}'")
            
        except Exception as e:
            print(f"\n❌ 1. CSV 파일 처리 실패: {e}")
            print("   파일 삭제를 중단합니다. (NPY/PNG는 삭제되지 않았습니다)")
            return

        # 2. NPY / PNG 파일 실제 삭제
        print("\n✓ 2. NPY 및 PNG 파일 삭제 시작...")
        deleted_png_count = 0
        deleted_npy_count = 0
        
        for base_name in ids_to_delete_set:
            # PNG
            png_to_del = os.path.join(PNG_DIR, f"{base_name}.png")
            if os.path.exists(png_to_del):
                try:
                    os.remove(png_to_del)
                    deleted_png_count += 1
                except Exception as e:
                    print(f"  ⚠️ PNG 삭제 실패: {png_to_del} ({e})")
            
            # NPY
            npy_to_del = os.path.join(NPY_DIR, f"{base_name}.npy")
            if os.path.exists(npy_to_del):
                try:
                    os.remove(npy_to_del)
                    deleted_npy_count += 1
                except Exception as e:
                    print(f"  ⚠️ NPY 삭제 실패: {npy_to_del} ({e})")

        print(f"  - PNG 삭제 완료: {deleted_png_count} 개")
        print(f"  - NPY 삭제 완료: {deleted_npy_count} 개")
        
        print("\n🎉 모든 작업이 완료되었습니다.")
        messagebox.showinfo("작업 완료", "모든 파일 정리가 완료되었습니다.")

if __name__ == "__main__":
    if not os.path.exists(SUSPECT_LIST_FILE):
         messagebox.showerror("오류", f"'{SUSPECT_LIST_FILE}'을(를) 찾을 수 없습니다.\n스크립트 1 (find_suspects.py)을 먼저 실행하세요.")
    elif not os.path.exists(ORIGINAL_CSV_FILE):
         messagebox.showerror("오류", f"'{ORIGINAL_CSV_FILE}'을(를) 찾을 수 없습니다.\n경로를 확인하세요.")
    else:
        root = tk.Tk()
        app = ImageReviewer(root)
        root.mainloop()