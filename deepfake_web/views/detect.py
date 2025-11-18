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

        if st.button("업로드하여 탐지 요청 보내기"):
            if uploaded_file is None:
                st.error("먼저 영상 파일을 업로드해주세요.")
            else:
                try:
                    resp = backend_api.post_detect_upload(
                        base_url=base_url,
                        access_token=access_token,
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        mime_type=uploaded_file.type or "application/octet-stream",
                    )
                    data = _show_response(resp)

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

        if st.button("유튜브 링크로 탐지 요청 보내기"):
            if not youtube_url:
                st.error("YouTube URL을 입력해주세요.")
            else:
                try:
                    resp = backend_api.post_detect_youtube(
                        base_url=base_url,
                        access_token=access_token,
                        youtube_url=youtube_url,
                    )
                    data = _show_response(resp)

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
                except Exception:
                    st.text(row["result_json"])
