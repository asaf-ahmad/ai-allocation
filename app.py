import streamlit as st, pandas as pd, numpy as np
from scipy.optimize import linear_sum_assignment

st.set_page_config(page_title="Automatic Agent Allocation", layout="wide")
st.title("Automatic Agent Allocation Demo")

# ----------  File upload  ----------
st.header("1  Upload data")
acc_file   = st.file_uploader("Accounts CSV (loan queue)",  type="csv")
agent_file = st.file_uploader("Agents CSV (capacity & KPI)", type="csv")

if acc_file and agent_file:
    acc   = pd.read_csv(acc_file)
    agent = pd.read_csv(agent_file)

    st.success(f"🗂 Loaded {len(acc)} accounts & {len(agent)} agents")

    # ----------  Build simple cost matrix  ----------
    #  Lower cost = better match.  Customise formula as you wish.
    #
    #  Example heuristic:
    #      cost = (dpd * 2)                       # urgency
    #           + (agent_workload * 5)            # capacity pressure
    #           - (agent_resolution_pct * 10)     # reward high performers
    #

    acc_matrix  = acc["dpd"].values.reshape(-1, 1)            # (A,1)
    agent_load  = agent["open_accounts"].values.reshape(1, -1)  # (1,B)
    agent_perf  = agent["resolution_pct"].values.reshape(1, -1)  # (1,B)

    cost = (acc_matrix * 2) + (agent_load * 5) - (agent_perf * 10)

    # ----------  Solve assignment (Hungarian) ----------
    row_ind, col_ind = linear_sum_assignment(cost)

    allocation = acc.loc[row_ind, ["account_id", "dpd"]].reset_index(drop=True)
    allocation["agent_id"]        = agent.loc[col_ind, "agent_id"].values
    allocation["agent_workload"]  = agent.loc[col_ind, "open_accounts"].values
    allocation["agent_res_pct"]   = agent.loc[col_ind, "resolution_pct"].values

    st.header("2  Optimal allocation")
    st.dataframe(allocation)

    # ----------  Download button  ----------
    csv = allocation.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "allocation.csv", "text/csv")
else:
    st.info("⬆️  Upload both CSV files to start.")
