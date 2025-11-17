import cv2
import dlib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import moviepy.editor as mp # 사용하지 않음
import os
import time
from scipy.spatial import distance as dist

# ============================================================
# 1. 사용자 설정 (필수)
# ============================================================
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
VIDEO_SOURCE_DIR = "../test" # ⚠️ [수정필요] 원본 비디오 폴더ㄴ
OUTPUT_DIR = "../output"       # ⚠️ [수정필요] 결과물 저장 폴더

VAD_TARGET_DURATION = 3.0 # 우리가 원하는 조각의 길이 (초)
VAD_SR = 22050 # VAD 분석을 위한 샘플링 속도 (빠름)A
VAD_TOP_DB = 40 # 침묵을 판단하는 기준 (dB)

# ============================================================
# 2. 출력 폴더 및 파일 경로 설정
# ============================================================
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "1_statistics_all_summary.csv")  
NPY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "2_npy_timeseries")
AUDIO_IMG_DIR = os.path.join(OUTPUT_DIR, "3_audio_spectrograms")

# ============================================================
# 3. dlib 모델 로드 (CPU용 HOG)
# ============================================================
try:
    print("dlib CUDA 사용 가능 여부: False (CPU HOG 탐지기 모드)")
    print("CPU용 dlib HOG 얼굴 탐지기 로드 중...")
    detector = dlib.get_frontal_face_detector()
    print("✓ HOG 탐지기 로드 완료.")
    
    print(f"얼굴 랜드마크 예측기 로드 중 ({DLIB_PREDICTOR_PATH})...")
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    print("✓ 랜드마크 예측기 로드 완료.")
    
except Exception as e:
    print(f"❌ dlib 모델 로드 실패: {e}")
    print(f"'{DLIB_PREDICTOR_PATH}' 파일이 있는지 확인하세요.")
    exit()

# ============================================================
# 4. 얼굴 영역별 랜드마크 인덱스 정의
# ============================================================
FACIAL_LANDMARKS = {
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "nose": list(range(27, 36)),
    "mouth": list(range(48, 68)),
    "jawline": list(range(0, 17)),
    "full_face": list(range(0, 68))
}

# ============================================================
# 5. 헬퍼 함수 정의
# ============================================================

def get_region_bounding_box(shape, landmark_indices):
    points = [(shape.part(i).x, shape.part(i).y) for i in landmark_indices]
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def calculate_region_features(gray_frame, shape, region_name, landmark_indices, 
                               prev_region_mean=None):
    try:
        x_min, y_min, x_max, y_max = get_region_bounding_box(shape, landmark_indices)
        h, w = gray_frame.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        region_crop = gray_frame[y_min:y_max, x_min:x_max]
        
        if region_crop.size == 0:
            return None
        
        laplacian = cv2.Laplacian(region_crop, cv2.CV_64F)
        laplacian_mean = np.abs(laplacian).mean()
        laplacian_var = laplacian.var()
        light_intensity_mean = region_crop.mean()
        
        light_intensity_change = 0.0
        if prev_region_mean is not None:
            light_intensity_change = light_intensity_mean - prev_region_mean
        
        region_area = (x_max - x_min) * (y_max - y_min)
        
        return {
            'laplacian_mean': laplacian_mean,
            'laplacian_var': laplacian_var,
            'light_intensity_mean': light_intensity_mean,
            'light_intensity_change': light_intensity_change,
            'region_area': region_area
        }
    except Exception as e:
        print(f"      ⚠️ {region_name} 영역 특징 추출 실패: {e}")
        return None


def save_mel_spectrogram(audio_segment, sr, file_path):
    try:
        S = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"    ⚠️ 스펙트로그램 저장 실패: {e}")


# ============================================================
# 6. 비디오 세그먼트 처리 함수
# ============================================================
def process_video_segment(video_path, segment_name, start_sec, end_sec, base_output_name):
    print(f"  ⏳ '{segment_name}' 세그먼트 ({start_sec:.2f}s - {end_sec:.2f}s) 처리 시작...")
    
    # --- PART A: 비디오(영상) 처리 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ❌ 비디오 열기 실패: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"    ❌ FPS가 0입니다. 파일 손상 가능성")
        cap.release()
        return None
        
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # [수정] OpenCV가 긴 영상의 먼 곳으로 seek 하는 데 실패할 수 있으므로,
    # set() 함수가 제대로 작동했는지 확인하는 로직 추가
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # 현재 위치를 다시 읽어와서, 설정한 start_frame과 1프레임 이상 차이나면 실패로 간주
    actual_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if abs(actual_start_frame - start_frame) > 1:
        print(f"    ❌ OpenCV 비디오 탐색(seek) 실패. (요청: {start_frame}, 실제: {actual_start_frame})")
        cap.release()
        # moviepy를 사용한 비디오 클립 잘라내기를 시도해 볼 수 있으나, 일단 오류로 처리
        return None
        
    frame_data = { 'frame_number': [], 'timestamp_sec': [] }
    for region_name in FACIAL_LANDMARKS.keys():
        frame_data[f'{region_name}_laplacian_mean'] = []
        frame_data[f'{region_name}_laplacian_var'] = []
        frame_data[f'{region_name}_light_intensity_mean'] = []
        frame_data[f'{region_name}_light_intensity_change'] = []
        frame_data[f'{region_name}_area'] = []
    
    prev_light_means = {region: None for region in FACIAL_LANDMARKS.keys()}
    processed_frames = 0
    
    # [수정] end_frame 대신 (end_frame - start_frame) 만큼 읽도록 변경
    frames_to_read = end_frame - start_frame
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_num = start_frame + i
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray) 
            if len(faces) == 0: continue
            
            face = faces[0] 
            shape = predictor(gray, face)
            
            frame_data['frame_number'].append(current_frame_num)
            frame_data['timestamp_sec'].append(current_frame_num / fps)
            
            for region_name, landmark_indices in FACIAL_LANDMARKS.items():
                features = calculate_region_features(
                    gray, shape, region_name, landmark_indices,
                    prev_region_mean=prev_light_means[region_name]
                )
                if features:
                    frame_data[f'{region_name}_laplacian_mean'].append(features['laplacian_mean'])
                    frame_data[f'{region_name}_laplacian_var'].append(features['laplacian_var'])
                    frame_data[f'{region_name}_light_intensity_mean'].append(features['light_intensity_mean'])
                    frame_data[f'{region_name}_light_intensity_change'].append(features['light_intensity_change'])
                    frame_data[f'{region_name}_area'].append(features['region_area'])
                    prev_light_means[region_name] = features['light_intensity_mean']
                else:
                    [frame_data[f'{region_name}_{ftype}'].append(np.nan) for ftype in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 'light_intensity_change', 'area']]
            
            processed_frames += 1
            
        except Exception as e:
            print(f"      ⚠️ 프레임 {current_frame_num} 처리 중 오류: {e}")
            continue
    
    cap.release()
    if processed_frames == 0:
        print(f"    ❌ 얼굴 감지 실패 (처리된 프레임: 0)")
        return None
    print(f"    ✓ 비디오 처리 완료: {processed_frames} 프레임")
    
    # --- PART B: 통계 데이터 생성 ---
    df_frames = pd.DataFrame(frame_data)
    stats_data = {'video_id': base_output_name, 'segment': segment_name}
    for region_name in FACIAL_LANDMARKS.keys():
        for feature_type in ['laplacian_mean', 'laplacian_var', 'light_intensity_mean', 
                             'light_intensity_change', 'area']:
            col_name = f'{region_name}_{feature_type}'
            if col_name in df_frames.columns:
                series = df_frames[col_name].dropna()
                if not series.empty:
                    stats_data[f'{col_name}_avg'] = series.mean()
                    stats_data[f'{col_name}_std'] = series.std()
                    stats_data[f'{col_name}_max'] = series.max()
                    stats_data[f'{col_name}_min'] = series.min()
                else:
                    stats_data.update({f'{col_name}_avg': 0, f'{col_name}_std': 0, f'{col_name}_max': 0, f'{col_name}_min': 0})
    
    # --- PART C: NPY 파일 생성 (상세한 시계열 데이터) ---
    npy_data = {
        'frame_numbers': np.array(frame_data['frame_number']),
        'timestamps': np.array(frame_data['timestamp_sec']),
        'video_id': base_output_name,
        'segment': segment_name,
        'fps': fps,
        'total_frames': processed_frames
    }
    for region_name in FACIAL_LANDMARKS.keys():
        npy_data[region_name] = {
            'laplacian_mean': np.array(frame_data[f'{region_name}_laplacian_mean']),
            'laplacian_var': np.array(frame_data[f'{region_name}_laplacian_var']),
            'light_intensity_mean': np.array(frame_data[f'{region_name}_light_intensity_mean']),
            'light_intensity_change': np.array(frame_data[f'{region_name}_light_intensity_change']),
            'area': np.array(frame_data[f'{region_name}_area'])
        }
    npy_path = os.path.join(NPY_OUTPUT_DIR, f"{base_output_name}.npy")
    np.save(npy_path, npy_data, allow_pickle=True)
    print(f"    ✓ NPY 저장 완료: {npy_path}")
    
    # --- PART D: 오디오 스펙트로그램 이미지 생성 (v8과 동일) ---
    try:
        target_sr = 44100 
        duration = end_sec - start_sec
        
        y, sr = librosa.load(
            video_path, 
            sr=target_sr, 
            offset=start_sec, 
            duration=duration
        )
        
        if y.size == 0:
            print(f"    ⚠️ 오디오 트랙이 비어있음 (librosa)")
        else:
            audio_img_path = os.path.join(AUDIO_IMG_DIR, f"{base_output_name}.png")
            save_mel_spectrogram(y, sr, audio_img_path)
            print(f"    ✓ 오디오 이미지 저장 완료: {audio_img_path}")
            
    except Exception as e:
        print(f"    ⚠️ 오디오 처리 실패 (librosa): {e}")
    
    return stats_data

# ============================================================
# 8. 메인 실행 함수 (VAD 버그 수정됨)
# ============================================================

def main():
    start_time = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(NPY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_IMG_DIR, exist_ok=True)
    
    print("="*70)
    print("🎬 딥페이크 감지용 멀티모달 특징 추출 파이프라인 (v10 - CPU HOG / VAD 버그 수정)")
    print("="*70)
    print(f"📂 비디오 소스: {VIDEO_SOURCE_DIR}")
    print(f"📂 출력 폴더: {OUTPUT_DIR}")
    print(f"🎤 VAD (음성 감지) 기준: {VAD_TOP_DB}dB, {VAD_TARGET_DURATION}초")
    print("="*70)
    
    video_files = []
    for root, _, files in os.walk(VIDEO_SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {VIDEO_SOURCE_DIR}")
        return
    
    print(f"\n🔍 발견된 비디오: {len(video_files)}개\n")
    
    # 이어하기 기능 로직
    all_new_stats = [] 
    processed_video_ids = set()
    file_exists = os.path.exists(CSV_OUTPUT_PATH)

    if file_exists:
        print(f"🔄 기존 통계 파일 {CSV_OUTPUT_PATH}을(를) 불러옵니다. (이어하기)")
        try:
            df_existing = pd.read_csv(CSV_OUTPUT_PATH)
            processed_video_ids = set(df_existing['video_id'])
            print(f"   ({len(processed_video_ids)}개 세그먼트가 이미 처리됨)")
        except Exception as e:
            print(f"   ⚠️ 기존 파일 읽기 실패: {e}. 새 파일로 시작합니다.")
            file_exists = False
    
    total_segments_to_process = 0
    success_count = 0 
    fail_count = 0 
    skipped_count = 0 
    
    # 각 비디오 처리
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        print(f"\n{'='*70}")
        print(f"📹 [{i+1}/{len(video_files)}] {video_name}")
        print(f"{'='*70}")
        
        # --- VAD (Voice Activity Detection) 로직 ---
        segments_to_process = {}
        try:
            print("  🔉 오디오 분석 (VAD) 시작...")
            y_full, sr_full = librosa.load(video_path, sr=VAD_SR)
            
            # [버그 수정] librosa.effects.split은 '샘플' 인덱스를 반환
            speech_segment_samples = librosa.effects.split(y_full, top_db=VAD_TOP_DB, hop_length=512)
            
            # [버그 수정] frames_to_time 대신 samples_to_time 사용
            speech_segments_time = librosa.samples_to_time(speech_segment_samples, sr=sr_full)
            
            # 3초 이상인 음성 구간 필터링
            valid_segments = []
            for start_sec, end_sec in speech_segments_time:
                if end_sec - start_sec >= VAD_TARGET_DURATION:
                    valid_segments.append((start_sec, end_sec))
            
            if not valid_segments:
                print("  ⚠️ 3초 이상의 음성 구간을 찾지 못했습니다. 이 비디오를 건너뜁니다.")
                continue
            
            print(f"  ✓ {len(valid_segments)}개의 유효한 음성 구간 발견.")

            # 3개 구간 샘플링 (초반, 중반, 후반)
            if len(valid_segments) == 1:
                start = valid_segments[0][0]
                segments_to_process['speech_1'] = (start, start + VAD_TARGET_DURATION)
            elif len(valid_segments) == 2:
                start1 = valid_segments[0][0]
                start2 = valid_segments[-1][0] # 마지막 구간
                segments_to_process['speech_1'] = (start1, start1 + VAD_TARGET_DURATION)
                segments_to_process['speech_2'] = (start2, start2 + VAD_TARGET_DURATION)
            else: # 3개 이상
                early_start = valid_segments[0][0]
                mid_start = valid_segments[len(valid_segments) // 2][0]
                late_start = valid_segments[-1][0]
                segments_to_process['speech_early'] = (early_start, early_start + VAD_TARGET_DURATION)
                segments_to_process['speech_mid'] = (mid_start, mid_start + VAD_TARGET_DURATION)
                segments_to_process['speech_late'] = (late_start, late_start + VAD_TARGET_DURATION)

        except Exception as e:
            print(f"  ❌ VAD 처리 실패 (오디오 로드 오류 등): {e}")
            continue # 다음 비디오로
        
        # --- VAD 로직 끝 ---

        total_segments_to_process += len(segments_to_process)
        for segment_name, (start_sec, end_sec) in segments_to_process.items():
            segment_id = f"{base_name}_{segment_name}"
            
            if segment_id in processed_video_ids:
                print(f"  ➡️ '{segment_name}' (ID: {segment_id})는 이미 처리되어 건너뜁니다.")
                skipped_count += 1
                continue
            
            stats_data = process_video_segment(
                video_path=video_path,
                segment_name=segment_name,
                start_sec=start_sec,
                end_sec=end_sec,
                base_output_name=segment_id
            )
            
            if stats_data:
                all_new_stats.append(stats_data) 
                success_count += 1
            else:
                fail_count += 1
        
        # 10개 비디오마다 중간 저장
        if (i + 1) % 10 == 0 and all_new_stats:
            print(f"\n... 📊 {len(all_new_stats)}개 신규 데이터 중간 저장 중 ...")
            try:
                df_intermediate = pd.DataFrame(all_new_stats)
                df_intermediate.to_csv(CSV_OUTPUT_PATH, mode='a', header=(not file_exists), index=False)
                all_new_stats = []
                file_exists = True
                print("    ✓ 중간 저장 완료.")
            except Exception as e:
                print(f"    ❌ 중간 저장 실패: {e}")

    
    # 최종 저장
    if all_new_stats:
        print(f"\n... 📊 {len(all_new_stats)}개 최종 데이터 저장 중 ...")
        try:
            df_final = pd.DataFrame(all_new_stats)
            df_final.to_csv(CSV_OUTPUT_PATH, mode='a', header=(not file_exists), index=False)
            print("    ✓ 최종 저장 완료.")
        except Exception as e:
            print(f"    ❌ 최종 저장 실패: {e}")

    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 모든 처리 완료!")
    print("="*70)
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
    print(f"✅ 새로 성공: {success_count} 세그먼트")
    print(f"❌ 새로 실패: {fail_count} 세그먼트")
    print(f"➡️ 건너뛰기: {skipped_count} 세그먼트")
    try:
        total_in_csv = len(pd.read_csv(CSV_OUTPUT_PATH))
    except:
        total_in_csv = 0
    print(f"💾 CSV 총계: {total_in_csv} 개 (VAD가 찾은 총 유효 세그먼트 수)")
    print(f"\n📁 출력 파일 위치:")
    print(f"   1. CSV (통합 통계): {CSV_OUTPUT_PATH}")
    print(f"   2. NPY (시계열 numpy 배열): {NPY_OUTPUT_DIR}")
    print(f"   3. PNG (오디오 스펙트로그램): {AUDIO_IMG_DIR}")
    print("="*70)


if __name__ == "__main__":

    main()

