# views/status.py
import streamlit as st
from services import backend_api


def render_status_page(base_url: str):
    st.title("default - API 상태 체크 (GET /)")

    st.subheader("GET /  - Root 엔드포인트")
    st.caption("백엔드가 살아있는지 확인용으로 사용.")

    if st.button("GET / 호출하기"):
        try:
            resp = backend_api.get_root(base_url=base_url)
            st.write(f"**Status code:** {resp.status_code}")
            try:
                st.json(resp.json())
            except Exception:
                st.text(resp.text)
        except Exception as e:
            st.error(f"요청 실패: {e}")

    st.markdown("---")
    st.markdown(
        """
        ### 사용 순서
        1. 사이드바에서 Backend Base URL이 올바른지 확인  
        2. **Auth** 페이지에서 회원가입/로그인  
        3. **Detect** 페이지에서 업로드 또는 유튜브 링크 탐지 + 사용기록 확인
        """
    )
