# main.py
import streamlit as st

from views.auth import render_auth_page
from views.detect import render_detect_page
from views.status import render_status_page
from services import db

DEFAULT_BASE_URL = "https://4eefe46ab453.ngrok-free.app"  # 자동 업데이트됨

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Deepfake Detection Frontend",
    layout="wide",
)

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_BASE_URL

# DB 초기화 (테이블 없으면 생성)
db.init_db()


# -----------------------------
# 사이드바 (공통 UI + 라우팅)
# -----------------------------
st.sidebar.title("Deepfake Detection")

# Backend Base URL 설정
st.sidebar.subheader("Backend 설정")
base_url_input = st.sidebar.text_input(
    "Backend Base URL",
    value=st.session_state.base_url,
    help="예: https://xxxx-xxxx.ngrok-free.app",
)
st.session_state.base_url = base_url_input

st.sidebar.markdown("---")

# 로그인 상태 표시
st.sidebar.subheader("로그인 상태")
if st.session_state.access_token:
    st.sidebar.success(f"로그인됨: {st.session_state.username}")
    if st.sidebar.button("로그아웃"):
        st.session_state.access_token = None
        st.session_state.username = None
        st.sidebar.info("로그아웃 완료")
else:
    st.sidebar.warning("로그인되지 않음")

st.sidebar.markdown("---")

# 라우팅(페이지 선택)
page = st.sidebar.radio(
    "메뉴",
    ["Auth (회원가입/로그인)", "Detect (탐지)", "API 상태 체크"],
)

# -----------------------------
# 라우트에 따라 각 view 호출
# -----------------------------
if page.startswith("Auth"):
    render_auth_page(base_url=st.session_state.base_url)

elif page.startswith("Detect"):
    render_detect_page(
        base_url=st.session_state.base_url,
        access_token=st.session_state.access_token,
        username=st.session_state.username,
    )

else:
    render_status_page(base_url=st.session_state.base_url)

# 10









