import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(APP_DIR)
    
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Pastikan file .joblib ada di direktori yang benar. Error: {e}")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Aplikasi berbasis Machine Learning untuk menganalisis dan memprediksi potensi kehilangan pelanggan (*churn*) secara akurat.")
st.write("")

tab_info, tab_input = st.tabs(["📈 Model Info", "📋 Prediction Form & Analysis"])

with tab_info:
    st.header("✨ Best Model Performance Profile")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""
        ### **Project Overview & Selection**
        Model utama yang dideploy pada sistem ini adalah **Random Forest Classifier (After Tuning)**. Model ini dipilih setelah melalui proses komparasi ketat melawan model baseline, menangani masalah *data imbalance*, serta dioptimalkan otomatis menggunakan **Optuna Hyperparameter Tuning (50 Trials)**.
        
        ### **Hyperparameters (Optuna Optimized):**
        * `n_estimators`: **411**
        * `max_depth`: **10**
        * `min_samples_split`: **7**
        * `min_samples_leaf`: **4**
        * `max_features`: **'sqrt'**
        """)
    
    with col_b:
        st.metric(label="Overall Accuracy", value="84.89%")  
        st.metric(label="F1-Score Macro", value="0.7604", delta="+0.2126 vs Baseline")
        st.metric(label="Recall Kelas Minoritas (Churn)", value="77.00%", delta="+69.00% vs Baseline") 

    st.info("""
    💡 **Key Insight Evaluasi:** Melalui kombinasi Preprocessing dan Hyperparameter Tuning, model berhasil mengatasi kendala *overfitting* dan tidak lagi 'buta' terhadap kelas minoritas. Dengan nilai *Recall* mencapai **77%**, model sangat sensitif dalam menjaring pelanggan yang berpotensi *churn* dengan tingkat kegagalan deteksi (*miss*) yang minim.
    """)

with tab_input:
    st.header("📋 Input Parameter Pelanggan")
    st.caption("Silakan lengkapi data profil dan aktivitas pelanggan di bawah ini untuk memulai analisis.")

    with st.form("churn_input_form"):
        
        with st.container(border=True):
            st.markdown("### 👤 Demografi & Informasi Akun")
            c1_1, c1_2, c1_3 = st.columns(3)
            with c1_1:
                gender = st.selectbox("Gender", options=["Male", "Female"])
                country = st.selectbox("Country", options=["Germany", "India", "Bangladesh", "USA", "UK"])
                city = st.selectbox("City", options=["London", "Mumbai", "Dhaka", "New York", "Delhi", "Berlin", "Hamburg"])
            with c1_2:
                acquisition_channel = st.selectbox("Acquisition Channel", options=["Organic", "Google Ads", "Facebook Ads", "Referral", "Email"])
                device_type = st.selectbox("Device Type", options=["Tablet", "Mobile", "Desktop"])
                subscription_type = st.selectbox("Subscription Type", options=["Monthly", "Annual"])
            with c1_3:
                payment_method = st.selectbox("Payment Method", options=["UPI", "PayPal", "SEPA", "BKash", "Card"])
                is_premium_user = st.radio("Is Premium User?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
                
        st.write("") 

        with st.container(border=True):
            st.markdown("### 📈 Aktivitas & Riwayat Transaksi")
            c2_1, c2_2, c2_3 = st.columns(3)
            with c2_1:
                age = st.number_input("Age", min_value=0.0, max_value=95.0, value=30.0)
                total_visits = st.number_input("Total Visits", min_value=0, max_value=31, value=5)
                avg_session_time = st.number_input("Avg Session Time (minutes)", min_value=0.0, max_value=20.0, value=8.0)
            with c2_2:
                pages_per_session = st.number_input("Pages Per Session", min_value=0.0, max_value=11.0, value=4.0)
                email_open_rate = st.number_input("Email Open Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5)
                email_click_rate = st.number_input("Email Click Rate (0.0 - 0.5)", min_value=0.0, max_value=0.5, value=0.1)
            with c2_3:
                total_spent = st.number_input("Total Spent", min_value=0.0, max_value=16000.0, value=500.0)
                avg_order_value = st.number_input("Avg Order Value", min_value=0.0, max_value=155.0, value=50.0)
                discount_used = st.radio("Discount Used?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
                coupon_code = st.selectbox("Coupon Code", options=["No_Coupon", "REF10", "SALE15", "NEW20"])

        st.write("") 

        with st.container(border=True):
            st.markdown("### ⭐️ Kepuasan Pelanggan & Interaksi Sistem")
            
            c3_1, c3_2, c3_3 = st.columns(3)
            with c3_1:
                support_tickets = st.number_input("Support Tickets", min_value=0, max_value=9, value=0)
                refund_requested = st.radio("Refund Requested?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
            with c3_2:
                delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0, max_value=11, value=0)
                marketing_spend_per_user = st.number_input("Marketing Spend Per User", min_value=0.0, max_value=30.0, value=5.0)
            with c3_3:
                lifetime_value = st.number_input("Lifetime Value", min_value=0.0, max_value=3800.0, value=1000.0)
                last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Freq", min_value=0, max_value=14, value=2)
            
            st.divider() 
            
            c3_b1, c3_b2 = st.columns(2)
            with c3_b1:
                satisfaction_score = st.slider("Satisfaction Score", min_value=1.0, max_value=5.0, value=4.0, step=1.0)
                days_to_last_purchase = st.number_input("Days to Last Purchase", min_value=0, value=15)
            with c3_b2:
                nps_score = st.slider("NPS Score", min_value=0, max_value=10, value=8)
                c_date1, c_date2 = st.columns(2)
                with c_date1:
                    signup_year = st.number_input("Signup Year", min_value=2010, max_value=2026, value=2025)
                with c_date2:
                    signup_month = st.slider("Signup Month", min_value=1, max_value=12, value=6)

        st.write("") 
        
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            submitted = st.form_submit_button("🚀 Run Churn Prediction Analysis", use_container_width=True)

    if submitted:
        st.write("---")
        st.header("🔮 Prediction Analysis Result")
        
        gender_map = {"Female": 0, "Male": 1}
        country_map = {"Bangladesh": 0, "Germany": 1, "UK": 2, "USA": 3, "India": 4}
        city_map = {"Berlin": 0, "Delhi": 1, "Dhaka": 2, "Hamburg": 3, "London": 4, "Mumbai": 5, "New York": 6}
        channel_map = {"Email": 0, "Facebook Ads": 1, "Google Ads": 2, "Organic": 3, "Referral": 4}
        device_map = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
        sub_map = {"Annual": 0, "Monthly": 1}
        coupon_map = {"NEW20": 0, "No_Coupon": 1, "REF10": 2, "SALE15": 3}
        payment_map = {"BKash": 0, "Card": 1, "PayPal": 2, "SEPA": 3, "UPI": 4}

        input_row = {
            'gender': gender_map.get(gender, 1),
            'age': age,
            'country': country_map.get(country, 0),
            'city': city_map.get(city, 0),
            'acquisition_channel': channel_map.get(acquisition_channel, 3),
            'device_type': device_map.get(device_type, 1),
            'subscription_type': sub_map.get(subscription_type, 1),
            'is_premium_user': is_premium_user,
            'total_visits': total_visits,
            'avg_session_time': avg_session_time,
            'pages_per_session': pages_per_session,
            'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'discount_used': discount_used,
            'coupon_code': coupon_map.get(coupon_code, 1),
            'support_tickets': support_tickets,
            'refund_requested': refund_requested,
            'delivery_delay_days': delivery_delay_days,
            'payment_method': payment_map.get(payment_method, 1),
            'satisfaction_score': satisfaction_score,
            'nps_score': nps_score,
            'marketing_spend_per_user': marketing_spend_per_user,
            'lifetime_value': lifetime_value,
            'last_3_month_purchase_freq': last_3_month_purchase_freq,
            'days_to_last_purchase': days_to_last_purchase,
            'signup_year': signup_year,
            'signup_month': signup_month
        }

        feature_order = [
            'gender', 'age', 'country', 'city', 'acquisition_channel', 'device_type',
            'subscription_type', 'is_premium_user', 'total_visits', 'avg_session_time',
            'pages_per_session', 'email_open_rate', 'email_click_rate', 'total_spent',
            'avg_order_value', 'discount_used', 'coupon_code', 'support_tickets',
            'refund_requested', 'delivery_delay_days', 'payment_method', 'satisfaction_score',
            'nps_score', 'marketing_spend_per_user', 'lifetime_value', 
            'last_3_month_purchase_freq', 'days_to_last_purchase', 'signup_year', 'signup_month'
        ]

        input_df = pd.DataFrame([input_row])[feature_order]
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            with st.container(border=True):
                st.write("### 📊 Probability Meter")
                st.metric(label="Churn Probability Score", value=f"{prediction_proba:.2%}")

        with col_res2:
            with st.container(border=True):
                st.write("### 🎯 Verdict & Action Items")
                if prediction[0] == 1:
                    st.error(f"⚠️ **Warning: Pelanggan terindikasi kuat akan CHURN.**")
                    st.markdown("""
                    * **Rekomendasi Tindakan:**
                        * Segera hubungi pelanggan secara personal melalui tim CRM.
                        * Berikan insentif berupa kode promo eksklusif atau diskon loyalitas sebelum langganan berakhir.
                    """)
                else:
                    st.success(f"✅ **Success: Pelanggan diprediksi tetap LOYAL.**")
                    st.markdown("""
                    * **Rekomendasi Tindakan:**
                        * Pertahankan pola layanan dan kualitas interaksi saat ini.
                        * Pelanggan masuk kategori ideal untuk ditawarkan program *Upselling* produk premium.
                    """)