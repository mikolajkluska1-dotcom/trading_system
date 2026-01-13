import streamlit as st
import pandas as pd
from security.user_manager import UserManager

def render_admin():
    if st.session_state['sys']['role'] != "ROOT":
        st.error("ACCESS DENIED")
        return

    st.markdown("##### ADMIN PANEL")
    db = UserManager.load_db()

    st.markdown("###### REQUESTS")
    for u in list(db.get('pending', {}).keys()):
        c1, c2, c3 = st.columns([3,1,1])
        c1.code(u)
        if c2.button("YES", key=f"y_{u}"): UserManager.approve_user(u, "TRADER"); st.rerun()
        if c3.button("NO", key=f"n_{u}"): UserManager.reject_user(u); st.rerun()

    st.markdown("---")
    st.markdown("###### ACTIVE USERS")
    if db['active']:
        st.dataframe(pd.DataFrame(db['active']).T)
