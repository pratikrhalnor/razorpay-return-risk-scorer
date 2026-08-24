import streamlit as st
import pandas as pd
import joblib
import json

model = joblib.load("models/return_risk_model.pkl")
with open("models/model_columns.json") as f:
    model_columns = json.load(f)
with open("models/category_options.json") as f:
    cat_options = json.load(f)

st.set_page_config(page_title="Return Risk Scorer", layout="centered")
st.title("🎯 Return Risk Scorer")
st.caption("Built for Razorpay AI Buildathon — Track 02: AI Risk Manager")

tab1, tab2, tab3 = st.tabs(["📋 Problem", "📊 Model Performance", "🔍 Try It Live"])

# ---------------- TAB 1: PROBLEM ----------------
with tab1:
    st.header("The Problem")
    st.write("""
    Merchants lose money to **return abuse** — customers who wear an item once 
    and return it ("wardrobing"), or repeatedly exploit return policies. 
    Most merchants can't tell an honest return from abuse until it's too late.
    """)
    st.header("What This Does")
    st.write("""
    Scores every return request as **Low / Medium / High risk** using the 
    customer's order history and the return's details — so merchants only 
    manually review the risky ones, instead of investigating everything or nothing.
    """)
    st.header("Data & Honesty Note")
    st.info("""
    Trained on a 60,000-row synthetic Kaggle dataset. We found and removed 
    several columns that leaked the answer directly into the data (e.g. a 
    pre-calculated return-rate percentage). We also injected realistic noise 
    to simulate real-world messiness, since even a 3-question decision tree 
    scored 98% on the raw synthetic data — a sign it was too clean to trust.
    Metrics below reflect the noise-adjusted, more realistic version.
    """)

# ---------------- TAB 2: MODEL PERFORMANCE ----------------
with tab2:
    st.header("Held-Out Test Set Results")
    st.caption("12,000 return requests the model never saw during training")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision (Risky)", "94%")
    col2.metric("Recall (Risky)", "87%")
    col3.metric("Overall Accuracy", "94%")

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        [[7972, 210], [479, 3339]],
        index=["Actually Honest", "Actually Risky"],
        columns=["Predicted Honest", "Predicted Risky"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("💰 False-Positive Cost Analysis")
    st.write("""
    | | Count | Est. cost per case | Est. total |
    |---|---|---|---|
    | ❌ False alarms (honest, flagged) | 210 | ~$3 (support time) | ~$630 |
    | ❌ Missed abuse | 479 | ~$150 (avg refund) | ~$71,850 |
    | ✅ Abuse correctly caught | 3,339 | ~$150 (avg refund saved) | ~$500,850 |
    """)
    st.caption("Cost assumptions are illustrative, stated explicitly for transparency — not measured from real merchant data.")

# ---------------- TAB 3: TRY IT LIVE ----------------
with tab3:
    st.header("Score a Return Request")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Customer age", 18, 70, 30)
        account_age_days = st.slider("Account age (days)", 1, 2500, 365)
        customer_segment = st.selectbox("Customer segment", cat_options['customer_segment'])
        country = st.selectbox("Country", cat_options['country'])
        platform = st.selectbox("Platform", cat_options['platform'])
        device_type = st.selectbox("Device type", cat_options['device_type'])
        payment_method = st.selectbox("Payment method", cat_options['payment_method'])
        product_category = st.selectbox("Product category", cat_options['product_category'])

    with col2:
        avg_order_value_usd = st.slider("Avg order value ($)", 15, 800, 150)
        refund_amount_requested_usd = st.slider("Refund amount requested ($)", 12, 837, 100)
        is_high_value_item = st.checkbox("Is high-value item")
        discount_used = st.checkbox("Discount was used")
        days_to_return = st.slider("Days to return", 1, 55, 10)
        return_reason = st.selectbox("Return reason", cat_options['return_reason'])
        shipping_carrier = st.selectbox("Shipping carrier", cat_options['shipping_carrier'])
        total_orders_lifetime = st.slider("Total orders (lifetime)", 1, 120, 20)
        total_returns_lifetime = st.slider("Total returns (lifetime)", 0, 101, 2)
        wishlist_to_cart_time_hrs = st.slider("Wishlist-to-cart time (hrs)", 0.1, 72.0, 24.0)

    if st.button("Score this return", type="primary"):
        raw_input = pd.DataFrame([{
            'age': age, 'account_age_days': account_age_days,
            'customer_segment': customer_segment, 'country': country,
            'platform': platform, 'device_type': device_type,
            'payment_method': payment_method, 'product_category': product_category,
            'avg_order_value_usd': avg_order_value_usd,
            'refund_amount_requested_usd': refund_amount_requested_usd,
            'is_high_value_item': int(is_high_value_item),
            'discount_used': int(discount_used), 'days_to_return': days_to_return,
            'return_reason': return_reason, 'shipping_carrier': shipping_carrier,
            'total_orders_lifetime': total_orders_lifetime,
            'total_returns_lifetime': total_returns_lifetime,
            'wishlist_to_cart_time_hrs': wishlist_to_cart_time_hrs,
        }])

        categorical_cols = ['customer_segment', 'country', 'platform', 'device_type',
                             'payment_method', 'product_category', 'return_reason',
                             'shipping_carrier']
        encoded = pd.get_dummies(raw_input, columns=categorical_cols)
        encoded = encoded.reindex(columns=model_columns, fill_value=0)

        risk_score = model.predict_proba(encoded)[0][1]

        st.divider()
        if risk_score < 0.3:
            st.success(f"🟢 LOW RISK — score: {risk_score:.2f} — Auto-approve refund.")
        elif risk_score < 0.6:
            st.warning(f"🟡 MEDIUM RISK — score: {risk_score:.2f} — Flag for human review.")
        else:
            st.error(f"🔴 HIGH RISK — score: {risk_score:.2f} — Hold refund, request more info.")

        st.subheader("Why this score?")
        importance = pd.DataFrame({
            'feature': model_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        st.dataframe(importance, hide_index=True, use_container_width=True)
with tab4:
    st.header("Model vs. Reality")
    st.caption("Real examples from the held-out test set — including a mistake, not just wins")
    reality_df = pd.read_csv("models/model_vs_reality.csv")
    reality_df['actual'] = reality_df['actual'].map({0: 'Honest', 1: 'Risky'})
    reality_df['predicted'] = reality_df['predicted'].map({0: 'Honest', 1: 'Risky'})
    reality_df['correct'] = reality_df['correct'].map({True: '✅', False: '❌'})
    st.dataframe(reality_df, use_container_width=True, hide_index=True)