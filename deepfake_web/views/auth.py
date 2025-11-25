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
        st.caption("이메일과 비밀번호로 회원가입합니다.")

        with st.form("signup_form"):
            email = st.text_input("email")
            password = st.text_input("password", type="password")

            submitted = st.form_submit_button("회원가입 요청 보내기")

        if submitted:
            try:
                resp = backend_api.post_signup(
                    base_url=base_url,
                    email=email,
                    password=password,
                )
                _show_response(resp)
            except Exception as e:
                st.error(f"요청 실패: {e}")

    # -------------------------
    # 로그인
    # -------------------------
    with tab_login:
        st.subheader("POST /auth/login  - 로그인")
        st.caption("이메일과 비밀번호로 로그인합니다.")

        with st.form("login_form"):
            email = st.text_input("email", key="login_email")
            password = st.text_input("password", type="password", key="login_password")
            submitted = st.form_submit_button("로그인 요청 보내기")

        if submitted:
            try:
                resp = backend_api.post_login(
                    base_url=base_url,
                    email=email,
                    password=password,
                )
                _show_response(resp)

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.user_email = email
                    st.success("로그인 성공!")
            except Exception as e:
                st.error(f"요청 실패: {e}")
