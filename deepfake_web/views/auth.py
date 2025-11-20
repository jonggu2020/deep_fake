# views/auth.py
import streamlit as st
from services import backend_api


def _show_response(resp):
    if resp is None:
        st.error("응답이 없습니다.")
        return

    st.write(f"**Status code:** {resp.status_code}")
    try:
        st.json(resp.json())
    except Exception:
        st.text(resp.text)


def render_auth_page(base_url: str):
    st.title("auth - 인증 API")

    tab_signup, tab_login = st.tabs(["POST /auth/signup", "POST /auth/login"])

    # -------------------------
    # 회원가입
    # -------------------------
    with tab_signup:
        st.subheader("POST /auth/signup  - 회원가입")
        st.caption("백엔드 스키마에 맞게 필드명을 수정해서 사용하세요.")

        with st.form("signup_form"):
            username = st.text_input("username")
            email = st.text_input("email (옵션)")
            password = st.text_input("password", type="password")

            submitted = st.form_submit_button("회원가입 요청 보내기")

        if submitted:
            try:
                resp = backend_api.post_signup(
                    base_url=base_url,
                    username=username,
                    password=password,
                    email=email or None,
                )
                _show_response(resp)
            except Exception as e:
                st.error(f"요청 실패: {e}")

    # -------------------------
    # 로그인
    # -------------------------
    with tab_login:
        st.subheader("POST /auth/login  - 로그인")
        st.caption("응답에서 access_token 필드가 있다고 가정합니다.")

        with st.form("login_form"):
            username = st.text_input("username", key="login_username")
            password = st.text_input("password", type="password", key="login_password")
            submitted = st.form_submit_button("로그인 요청 보내기")

        if submitted:
            try:
                resp = backend_api.post_login(
                    base_url=base_url,
                    username=username,
                    password=password,
                )
                _show_response(resp)

                if resp.status_code == 200:
                    data = resp.json()
                    token = data.get("access_token") or data.get("token")
                    if token:
                        st.session_state.access_token = token
                        st.session_state.username = username
                        st.success("로그인 성공, 토큰 세션에 저장됨.")
                    else:
                        st.warning("응답에서 access_token을 찾지 못했습니다. 필드명을 확인하세요.")
            except Exception as e:
                st.error(f"요청 실패: {e}")
