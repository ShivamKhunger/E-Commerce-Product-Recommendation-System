import streamlit as st
import pandas as pd
import glob

st.set_page_config(page_title="E-Commerce Recommender", layout="wide")

def read_csv_folder(path_glob):
    files = glob.glob(path_glob)
    df_list = [pd.read_csv(f) for f in files]
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

st.sidebar.header("Data Selection")
recs = read_csv_folder("work/out/top_recommendations_pretty/*.csv")
summary = read_csv_folder("work/out/analytics_summary/*.csv")
metrics = pd.read_csv("work/out/model_metrics.csv")

if recs.empty:
    st.error("No recommendation data found — run train_als_implicit.py first.")
    st.stop()

rmse = round(float(metrics.loc[0, "value"]), 4) if not metrics.empty else None

st.title("📊 E-Commerce Product Recommendation Dashboard")
k1, k2, k3 = st.columns(3)
k1.metric("Unique Users", len(recs["user_id"].unique()))
k2.metric("Unique Products", len(recs["product_id"].unique()))
k3.metric("Model RMSE", rmse)

st.markdown("---")

st.subheader("🔎 Personalized Recommendations")
user_ids = sorted(recs["user_id"].unique())
selected_user = st.selectbox("Select a User ID", user_ids, index=0)

user_recs = recs[recs["user_id"] == selected_user].copy()
user_recs = user_recs.sort_values("pred_score", ascending=False)
user_recs["price"] = user_recs["price"].fillna(0).astype(float).round(2)

st.dataframe(
    user_recs[["product_id", "brand", "category_code", "price", "pred_score"]]
        .rename(columns={
            "product_id": "Product ID",
            "brand": "Brand",
            "category_code": "Category",
            "price": "Price ($)",
            "pred_score": "Predicted Score"
        }),
    use_container_width=True,
    hide_index=True
)

st.markdown("### 🏆 Most Popular Products")
if not summary.empty:
    top10 = summary.sort_values("interaction_count", ascending=False).head(10)
    st.bar_chart(
        top10.set_index(top10["product_id"].astype(str))["interaction_count"],
        use_container_width=True
    )
else:
    st.info("Analytics summary not found.")
