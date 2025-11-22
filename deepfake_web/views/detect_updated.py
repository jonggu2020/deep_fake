# views/detect.py
import json
import streamlit as st

from services import backend_api, db


def _show_response(resp):
    if resp is None:
        st.error("응답이 없습니다.")
        return None

    st.write(f"**Status code:** {resp.status_code}")
    try:
        data = resp.json()
        st.json(data)
        return data
    except Exception:
        st.text(resp.text)
        return None


def render_detect_page(base_url: str, access_token: str | None, username: str | None):
    st.title("detect - 딥페이크 탐지 API")

    if not access_token:
        st.warning("탐지 기능을 사용하려면 먼저 로그인 해주세요. (Authorization 필요 가정)")
    else:
        st.info(f"현재 사용자: {username or '알 수 없음'}")

    tab_upload, tab_youtube, tab_history = st.tabs(
        ["POST /detect/upload", "POST /detect/youtube", "사용기록 조회"]
    )

    # -------------------------
    # 업로드 탐지
    # -------------------------
    with tab_upload:
        st.subheader("POST /detect/upload  - 업로드한 영상으로 탐지")

        uploaded_file = st.file_uploader(
            "탐지할 영상 파일 업로드",
            type=["mp4", "avi", "mov", "mkv"],
        )

        # 업로드된 영상 미리보기 (크기 조절)
        if uploaded_file is not None:
            st.write("**영상 미리보기:**")
            col1, col2, col3 = st.columns([1, 2, 1])  # 중앙에 배치
            with col2:
                st.video(uploaded_file)

        if st.button("업로드하여 탐지 요청 보내기"):
            if uploaded_file is None:
                st.error("먼저 영상 파일을 업로드해주세요.")
            else:
                with st.spinner("영상 분석 중..."):
                    try:
                        resp = backend_api.post_detect_upload(
                            base_url=base_url,
                            access_token=access_token,
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            mime_type=uploaded_file.type or "application/octet-stream",
                        )
                        data = _show_response(resp)

                        # 결과 시각화
                        if data and resp.status_code == 200:
                            st.success("✅ 분석 완료!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="딥페이크 여부",
                                    value="Deepfake" if data.get("is_deepfake") == 1 else "Real"
                                )
                            with col2:
                                confidence = data.get("confidence", 0)
                                st.metric(
                                    label="신뢰도",
                                    value=f"{confidence * 100:.1f}%"
                                )
                            
                            # 랜드마크 영상 표시
                            landmark_path = data.get("landmark_video_path")
                            if landmark_path:
                                st.markdown("---")
                                st.subheader("🎯 얼굴 랜드마크 추출 영상")
                                
                                # 랜드마크 처리 정보 표시
                                landmark_info = data.get("landmark_info")
                                if landmark_info:
                                    info_col1, info_col2, info_col3 = st.columns(3)
                                    with info_col1:
                                        st.metric("처리 시간", f"{landmark_info.get('processing_time', 0):.2f}초")
                                    with info_col2:
                                        st.metric("처리 프레임", f"{landmark_info.get('processed_frames', 0)}")
                                    with info_col3:
                                        st.metric("얼굴 감지", f"{landmark_info.get('faces_detected', 0)}프레임")
                                
                                # 랜드마크 영상 다운로드 및 표시
                                video_id = data.get("video_id")
                                landmark_url = f"{base_url}/detect/landmark/{video_id}"
                                
                                st.write("**랜드마크 영상:**")
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    try:
                                        # 영상 표시
                                        st.video(landmark_url)
                                    except Exception as e:
                                        st.warning(f"영상을 표시할 수 없습니다: {e}")
                                
                                # 다운로드 링크
                                st.markdown(f"[📥 랜드마크 영상 다운로드]({landmark_url})")
                            else:
                                st.info("ℹ️  랜드마크 영상이 생성되지 않았습니다.")

                        # DB에 로그 저장
                        db.save_detection_history(
                            username=username,
                            source_type="upload",
                            source_value=uploaded_file.name,
                            result=data,
                        )
                        st.success("사용기록 저장 완료.")
                    except Exception as e:
                        st.error(f"요청 실패: {e}")

    # -------------------------
    # 유튜브 탐지
    # -------------------------
    with tab_youtube:
        st.subheader("POST /detect/youtube  - 유튜브 링크로 탐지")

        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/...",
        )

        # 유튜브 영상 미리보기 (크기 조절)
        if youtube_url:
            st.write("**영상 미리보기:**")
            col1, col2, col3 = st.columns([1, 2, 1])  # 중앙에 배치
            with col2:
                try:
                    st.video(youtube_url)
                except Exception:
                    st.warning("영상을 표시할 수 없습니다.")

        if st.button("유튜브 링크로 탐지 요청 보내기"):
            if not youtube_url:
                st.error("YouTube URL을 입력해주세요.")
            else:
                with st.spinner("영상 다운로드 및 분석 중..."):
                    try:
                        resp = backend_api.post_detect_youtube(
                            base_url=base_url,
                            access_token=access_token,
                            youtube_url=youtube_url,
                        )
                        data = _show_response(resp)

                        # 결과 시각화
                        if data and resp.status_code == 200:
                            st.success("✅ 분석 완료!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="딥페이크 여부",
                                    value="Deepfake" if data.get("is_deepfake") == 1 else "Real"
                                )
                            with col2:
                                confidence = data.get("confidence", 0)
                                st.metric(
                                    label="신뢰도",
                                    value=f"{confidence * 100:.1f}%"
                                )
                            
                            # 랜드마크 영상 표시
                            landmark_path = data.get("landmark_video_path")
                            if landmark_path:
                                st.markdown("---")
                                st.subheader("🎯 얼굴 랜드마크 추출 영상")
                                
                                # 랜드마크 처리 정보 표시
                                landmark_info = data.get("landmark_info")
                                if landmark_info:
                                    info_col1, info_col2, info_col3 = st.columns(3)
                                    with info_col1:
                                        st.metric("처리 시간", f"{landmark_info.get('processing_time', 0):.2f}초")
                                    with info_col2:
                                        st.metric("처리 프레임", f"{landmark_info.get('processed_frames', 0)}")
                                    with info_col3:
                                        st.metric("얼굴 감지", f"{landmark_info.get('faces_detected', 0)}프레임")
                                
                                # 랜드마크 영상 다운로드 및 표시
                                video_id = data.get("video_id")
                                landmark_url = f"{base_url}/detect/landmark/{video_id}"
                                
                                st.write("**랜드마크 영상:**")
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    try:
                                        # 영상 표시
                                        st.video(landmark_url)
                                    except Exception as e:
                                        st.warning(f"영상을 표시할 수 없습니다: {e}")
                                
                                # 다운로드 링크
                                st.markdown(f"[📥 랜드마크 영상 다운로드]({landmark_url})")
                            else:
                                st.info("ℹ️  랜드마크 영상이 생성되지 않았습니다.")

                        db.save_detection_history(
                            username=username,
                            source_type="youtube",
                            source_value=youtube_url,
                            result=data,
                        )
                        st.success("사용기록 저장 완료.")
                    except Exception as e:
                        st.error(f"요청 실패: {e}")

    # -------------------------
    # 사용기록 조회
    # -------------------------
    with tab_history:
        st.subheader("사용기록 조회 (로컬 DB)")

        rows = db.get_detection_history(username=username)
        if not rows:
            st.info("저장된 사용기록이 없습니다.")
        else:
            for row in rows:
                st.markdown("---")
                st.write(f"**ID:** {row['id']} / **시간:** {row['created_at']}")
                st.write(f"**유저:** {row['username']}")
                st.write(f"**타입:** {row['source_type']}")
                st.write(f"**값:** {row['source_value']}")

                try:
                    result = json.loads(row["result_json"]) if row["result_json"] else None
                    if result:
                        with st.expander("결과 JSON 보기", expanded=False):
                            st.json(result)
                        
                        # 랜드마크 영상 링크 표시
                        landmark_path = result.get("landmark_video_path")
                        if landmark_path:
                            video_id = result.get("video_id")
                            if video_id:
                                # base_url을 세션에서 가져오거나 기본값 사용
                                hist_base_url = st.session_state.get("base_url", "http://localhost:8000")
                                landmark_url = f"{hist_base_url}/detect/landmark/{video_id}"
                                st.markdown(f"[🎯 랜드마크 영상 보기]({landmark_url})")
                except Exception:
                    st.text(row["result_json"])
