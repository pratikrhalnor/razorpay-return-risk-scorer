import streamlit as st
import pandas as pd
import joblib
import json

# --- Load model and metadata ---
model = joblib.load("models/return_risk_model.pkl")
with open("models/model_columns.json") as f:
    model_columns = json.load(f)
with open("models/category_options.json") as f:
    cat_options = json.load(f)

st.set_page_config(page_title="Return Risk Scorer", layout="centered")
st.title("Return Risk Scorer")
st.caption("Built for Razorpay AI Buildathon — Track 02: AI Risk Manager")

st.markdown("Enter a return request's details to get a live risk score.")

# --- Input form ---
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
    # Build a single-row input matching training data structure
    raw_input = pd.DataFrame([{
        'age': age,
        'account_age_days': account_age_days,
        'customer_segment': customer_segment,
        'country': country,
        'platform': platform,
        'device_type': device_type,
        'payment_method': payment_method,
        'product_category': product_category,
        'avg_order_value_usd': avg_order_value_usd,
        'refund_amount_requested_usd': refund_amount_requested_usd,
        'is_high_value_item': int(is_high_value_item),
        'discount_used': int(discount_used),
        'days_to_return': days_to_return,
        'return_reason': return_reason,
        'shipping_carrier': shipping_carrier,
        'total_orders_lifetime': total_orders_lifetime,
        'total_returns_lifetime': total_returns_lifetime,
        'wishlist_to_cart_time_hrs': wishlist_to_cart_time_hrs,
    }])

    # One-hot encode same way as training
    categorical_cols = ['customer_segment', 'country', 'platform', 'device_type',
                         'payment_method', 'product_category', 'return_reason',
                         'shipping_carrier']
    encoded = pd.get_dummies(raw_input, columns=categorical_cols)

    # Align to the exact columns the model was trained on
    encoded = encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    risk_score = model.predict_proba(encoded)[0][1]
    prediction = model.predict(encoded)[0]

    st.divider()
    if risk_score < 0.3:
        st.success(f"🟢 LOW RISK — score: {risk_score:.2f}")
        st.write("Recommendation: Auto-approve refund.")
    elif risk_score < 0.6:
        st.warning(f"🟡 MEDIUM RISK — score: {risk_score:.2f}")
        st.write("Recommendation: Flag for human review.")
    else:
        st.error(f"🔴 HIGH RISK — score: {risk_score:.2f}")
        st.write("Recommendation: Hold refund, request more info before approving.")

    # Show top contributing factors
    st.divider()
    st.subheader("Why this score?")
    importance = pd.DataFrame({
        'feature': model_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    st.dataframe(importance, hide_index=True)