"""Account profile and administrator-only account activity."""

import pandas as pd
import streamlit as st

from xai_dashboard.ui.client import ApiError
from xai_dashboard.ui.components import require_login

st.set_page_config(page_title="User Management", layout="wide")
client, user = require_login()
st.title("User Management")
st.subheader("Your profile")
st.write({"Name": user["full_name"], "Email": user["email"], "Role": user["role"]})
if user["role"] == "admin":
    st.subheader("Registered users and activity")
    try:
        users = client.get("/users")
        st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)
    except ApiError as exc:
        st.error(str(exc))
else:
    st.info("Account activity administration is restricted to Admin users.")
