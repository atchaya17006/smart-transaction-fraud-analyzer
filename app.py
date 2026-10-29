import streamlit as st
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

st.set_page_config(
    page_title="Smart Transaction Fraud Risk Analyzer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Smart Transaction Fraud Risk Analyzer")
st.caption("Machine Learning-Based Transaction Risk Review Prototype")

st.info(
    "This prototype uses synthetic demonstration data. "
    "Risk scores are not proof of fraud."
)

# --------------------------------------------------
# 1. Generate synthetic training data
# --------------------------------------------------

random.seed(42)

training_data = []

for _ in range(1000):
    amount = random.randint(100, 100000)
    frequency = random.randint(1, 10)
    new_device = random.randint(0, 1)
    unusual_location = random.randint(0, 1)

    # Synthetic labels for demonstration only.
    # These are not real confirmed fraud labels.
    demo_score = (
        (amount / 100000) * 35
        + (frequency / 10) * 20
        + new_device * 25
        + unusual_location * 20
    )

    label = 1 if demo_score >= 65 else 0

    training_data.append({
        "Amount": amount,
        "Recent Frequency": frequency,
        "New Device": new_device,
        "Unusual Location": unusual_location,
        "Demo Label": label
    })

training_df = pd.DataFrame(training_data)

features = [
    "Amount",
    "Recent Frequency",
    "New Device",
    "Unusual Location"
]

X = training_df[features]
y = training_df["Demo Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Demonstration metrics: agreement with synthetic labels,
# NOT real-world fraud-detection performance.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = model.predict(X_test)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
pr_auc = average_precision_score(y_test, test_probabilities)

# --------------------------------------------------
# 2. Helper function for model risk scores
# --------------------------------------------------

def predict_risk(amount, frequency, new_device, unusual_location):
    input_df = pd.DataFrame([{
        "Amount": amount,
        "Recent Frequency": frequency,
        "New Device": new_device,
        "Unusual Location": unusual_location
    }])

    probability = model.predict_proba(input_df)[0][1]
    risk_score = int(round(probability * 100))

    if risk_score >= 65:
        risk_level = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return risk_score, risk_level


# --------------------------------------------------
# 3. Generate sample transactions
# --------------------------------------------------

if "transactions" not in st.session_state:
    data = []

    for i in range(1, 31):
        amount = random.randint(100, 100000)
        frequency = random.randint(1, 10)
        new_device = random.randint(0, 1)
        unusual_location = random.randint(0, 1)

        risk, level = predict_risk(
            amount,
            frequency,
            new_device,
            unusual_location
        )

        data.append({
            "Transaction ID": f"TXN{i:03}",
            "Amount": amount,
            "Recent Frequency": frequency,
            "New Device": new_device,
            "Unusual Location": unusual_location,
            "Risk Score": risk,
            "Risk Level": level,
            "Review Status": "Pending"
        })

    st.session_state.transactions = pd.DataFrame(data)

df = st.session_state.transactions

# --------------------------------------------------
# 4. Dashboard metrics
# --------------------------------------------------

total = len(df)
high = int((df["Risk Level"] == "High").sum())
medium = int((df["Risk Level"] == "Medium").sum())
low = int((df["Risk Level"] == "Low").sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", total)
c2.metric("High Risk", high)
c3.metric("Medium Risk", medium)
c4.metric("Low Risk", low)

st.divider()

# --------------------------------------------------
# 5. Risk distribution
# --------------------------------------------------

st.subheader("📊 Risk Distribution")
st.bar_chart(df["Risk Level"].value_counts())

# --------------------------------------------------
# 6. Suspicious transaction ranking
# --------------------------------------------------

st.subheader("🚨 Suspicious Transaction Ranking")

ranked = df.sort_values("Risk Score", ascending=False)

st.dataframe(
    ranked,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 7. Model evaluation
# --------------------------------------------------

st.divider()
st.subheader("📈 Model Evaluation")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Precision", f"{precision:.2f}")
m2.metric("Recall", f"{recall:.2f}")
m3.metric("F1-Score", f"{f1:.2f}")
m4.metric("PR-AUC", f"{pr_auc:.2f}")

st.caption(
    "These metrics measure performance against synthetic demonstration "
    "labels only. They do not establish real-world fraud-detection accuracy."
)

# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------

st.subheader("🔎 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance_df.set_index("Feature")
)

# --------------------------------------------------
# 9. What-if risk simulator
# --------------------------------------------------

st.divider()
st.subheader("🧪 What-if Risk Simulator")

amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=1,
    max_value=1000000,
    value=5000
)

frequency = st.slider(
    "Recent Transaction Frequency",
    1,
    10,
    3
)

new_device = st.selectbox(
    "New Device?",
    ["No", "Yes"]
)

unusual_location = st.selectbox(
    "Unusual Location?",
    ["No", "Yes"]
)

if st.button("Analyze Transaction"):
    risk, level = predict_risk(
        amount,
        frequency,
        int(new_device == "Yes"),
        int(unusual_location == "Yes")
    )

    st.metric("Model Risk Score", f"{risk}/100")
    st.write(f"**Risk Category:** {level}")

    reasons = []

    if amount > 20000:
        reasons.append("Transaction amount is high relative to the demo threshold.")
    if frequency >= 7:
        reasons.append("Recent transaction frequency is high.")
    if new_device == "Yes":
        reasons.append("Transaction is associated with a new device.")
    if unusual_location == "Yes":
        reasons.append("Transaction is marked as an unusual location.")

    if reasons:
        st.write("**Indicators to review:**")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("No listed indicators were triggered by these demo rules.")

    st.warning(
        "This is a model prediction based on synthetic training labels. "
        "It is not confirmation of fraud."
    )

# --------------------------------------------------
# 10. Investigation review queue
# --------------------------------------------------

st.divider()
st.subheader("📝 Investigation Review Queue")

selected_id = st.selectbox(
    "Select Transaction ID",
    df["Transaction ID"]
)

new_status = st.selectbox(
    "Update Review Status",
    ["Pending", "Under Review", "Reviewed"]
)

if st.button("Save Review Status"):
    st.session_state.transactions.loc[
        st.session_state.transactions["Transaction ID"] == selected_id,
        "Review Status"
    ] = new_status

    st.success(f"{selected_id} status updated to {new_status}.")
    st.rerun()

st.subheader("📋 Current Review Queue")

st.dataframe(
    st.session_state.transactions[
        ["Transaction ID", "Risk Score", "Risk Level", "Review Status"]
    ].sort_values("Risk Score", ascending=False),
    use_container_width=True,
    hide_index=True
)