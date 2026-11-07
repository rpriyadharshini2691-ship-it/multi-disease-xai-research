"""Account profile and administrator-only account activity."""

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import require_login

st.set_page_config(page_title="User Management", layout="wide")
client, user = require_login()
st.title("User Management")
st.subheader("Your profile")
st.write(
    {
        "Name": user["full_name"],
        "Username": user.get("username", ""),
        "Email": user["email"],
        "Role": user["role"],
    }
)
if user["role"] == "admin":
    st.subheader("Registered users")
    try:
        users = client.get("/users")
        table = pd.DataFrame(users).drop(columns=["activity"], errors="ignore")
        st.dataframe(table, width="stretch", hide_index=True)
        st.subheader("Manage an account")
        labels = {f"{u['username']} ({u['email']})": u for u in users}
        chosen = labels[st.selectbox("User", list(labels))]
        roles = ["admin", "researcher", "clinician"]
        col_role, col_status = st.columns(2)
        with col_role:
            new_role = st.selectbox("Role", roles, index=roles.index(chosen["role"]))
            if st.button("Apply role change") and new_role != chosen["role"]:
                try:
                    client.patch(f"/users/{chosen['id']}/role", {"role": new_role})
                    st.success("Role updated.")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
        with col_status:
            st.write(f"Status: **{'active' if chosen['is_active'] else 'inactive'}**")
            target_state = not chosen["is_active"]
            if st.button("Activate" if target_state else "Deactivate"):
                try:
                    client.patch(f"/users/{chosen['id']}/status", {"is_active": target_state})
                    st.success("Status updated.")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
    except ApiError as exc:
        st.error(str(exc))
else:
    st.info("Account activity administration is restricted to Admin users.")
