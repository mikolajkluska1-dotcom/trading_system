import streamlit as st
import time

from core.security import HardwareSecurity
from security.user_manager import UserManager


def render_auth():
    """
    REDLINE AUTH GATE
    Calm, centered, premium security login
    """

    # =====================================================
    # INIT SESSION STATE
    # =====================================================
    if "sys" not in st.session_state:
        st.session_state["sys"] = {
            "auth": False,
            "role": None,
            "user": None,
        }

    # =====================================================
    # AUTH-SPECIFIC CSS (CALM, NO NEON, NO WHITE)
    # =====================================================
    st.markdown(
        """
        <style>
        /* Center auth vertically */
        .block-container {
            padding-top: 4rem;
            padding-bottom: 6rem;
        }

        /* Inputs – dark, calm */
        .stTextInput input {
            background-color: #0b0f19 !important;
            color: #d1d5db !important;
            border: 1px solid #1f2937 !important;
            border-radius: 8px !important;
            padding: 10px 12px !important;
            font-size: 13px !important;
        }

        .stTextInput input:focus {
            border-color: #7aa2f7 !important;
            box-shadow: none !important;
        }

        /* Buttons */
        .stButton button {
            background: #0b0f19;
            border: 1px solid #1f2937;
            color: #d1d5db;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
        }

        .stButton button:hover {
            border-color: #7aa2f7;
            color: #7aa2f7;
        }

        /* Tabs – minimal */
        div[data-baseweb="tab-list"] {
            justify-content: center;
            gap: 18px;
        }

        div[data-baseweb="tab"] {
            font-size: 12px;
            opacity: 0.6;
        }

        div[data-baseweb="tab"][aria-selected="true"] {
            opacity: 1;
            border-bottom: 2px solid #7aa2f7;
        }

        /* Subtle system text */
        .sys-text {
            font-size: 11px;
            color: #9ca3af;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # CENTERED LAYOUT
    # =====================================================
    _, center, _ = st.columns([1, 1.6, 1])

    with center:
        # =================================================
        # HEADER
        # =================================================
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:24px;">
                <div style="font-size:32px; letter-spacing:4px; color:#7aa2f7;">
                    REDLINE
                </div>
                <div style="font-size:11px; color:#9ca3af; letter-spacing:2px;">
                    SECURE TRADING TERMINAL
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # =================================================
        # SYSTEM STATUS (SUBTLE)
        # =================================================
        st.markdown(
            """
            <div class="sys-text" style="margin-bottom:18px;">
                Secure channel initialized<br>
                Cipher handshake verified<br>
                User database online
            </div>
            """,
            unsafe_allow_html=True,
        )

        # =================================================
        # AUTH TABS
        # =================================================
        tab_login, tab_request = st.tabs(["LOGIN", "REQUEST ACCESS"])

        # =================================================
        # LOGIN TAB
        # =================================================
        with tab_login:
            # --- HARDWARE KEY (SUBTLE) ---
            try:
                has_key, drive = HardwareSecurity.scan_for_key()
                if has_key:
                    st.caption(f"Hardware key detected ({drive})")
                    if st.button("Quick Auth via Hardware Key", use_container_width=True):
                        if HardwareSecurity.verify_key_signature(drive):
                            st.session_state["sys"].update(
                                {"auth": True, "role": "ROOT", "user": "hardware"}
                            )
                            time.sleep(0.3)
                            st.rerun()
            except Exception:
                pass

            # --- CREDENTIAL FORM ---
            with st.form("auth_login"):
                user = st.text_input("User")
                passwd = st.text_input("Access Token", type="password")

                if st.form_submit_button("Initiate Uplink", use_container_width=True):
                    role = UserManager.verify_login(user, passwd)
                    if role:
                        st.session_state["sys"].update(
                            {"auth": True, "role": role, "user": user}
                        )
                        st.success("Access granted")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.warning("Invalid credentials")

        # =================================================
        # REQUEST ACCESS TAB
        # =================================================
        with tab_request:
            st.caption("Access requests require administrator approval.")

            with st.form("request_access"):
                new_user = st.text_input("Desired username")
                new_pass = st.text_input("Desired password", type="password")
                contact = st.text_input("Contact (email / telegram)")

                if st.form_submit_button("Submit Request", use_container_width=True):
                    if len(new_user) < 3 or len(new_pass) < 5:
                        st.warning("Username or password too short")
                    else:
                        ok, msg = UserManager.request_account(
                            new_user, new_pass, contact
                        )
                        if ok:
                            st.success("Request submitted")
                            st.caption("Await administrator approval")
                        else:
                            st.warning(msg)

        # =================================================
        # FOOTER (VERY SUBTLE)
        # =================================================
        try:
            hwid = HardwareSecurity.get_hardware_id()
            hwid_text = f"HWID {hwid}"
        except Exception:
            hwid_text = "HWID unavailable"

        st.markdown(
            f"""
            <div style="text-align:center; margin-top:28px;">
                <div class="sys-text">
                    {hwid_text}<br>
                    Unauthorized access attempts are logged
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
