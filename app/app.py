import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('C:\\Users\\hanao\\Downloads\\uas-bengkod\\bengkod-churn-prediction\\models\\best_model.joblib')
    scaler = joblib.load('C:\\Users\\hanao\\Downloads\\uas-bengkod\\bengkod-churn-prediction\\models\\scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Pastikan file .joblib ada di direktori yang benar. Error: {e}")

#  JUDUL APLIKASI 
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Aplikasi berbasis Machine Learning untuk menganalisis dan memprediksi potensi kehilangan pelanggan (*churn*) secara akurat.")
st.write("")

#  PEMBAGIAN HALAMAN MENGGUNAKAN TABS 
tab_info, tab_input, tab_result = st.tabs(["📈 Model Info", "📋 Prediction Form", "🔮 Result Analysis"])

# TAB 1: MODEL INFO (Menggunakan Catatan Evaluasimu)
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
        st.metric(label="Overall Accuracy", value="85.13%")
        st.metric(label="F1-Score Macro", value="0.7681", delta="+0.2126 vs Baseline")
        st.metric(label="Recall Kelas Minoritas (Churn)", value="81.00%", delta="+69.00% vs Baseline")

    st.info("""
    💡 **Key Insight Evaluasi:** Melalui kombinasi Preprocessing dan Hyperparameter Tuning, model berhasil mengatasi kendala *overfitting* dan tidak lagi 'buta' terhadap kelas minoritas. Dengan nilai *Recall* mencapai **81%**, model sangat sensitif dalam menjaring pelanggan yang berpotensi *churn* dengan tingkat kegagalan deteksi (*miss*) yang sangat minim.
    """)


# TAB 2: INPUT PARAMETER PELANGGAN
with tab_input:
    st.header("📋 Input Parameter Pelanggan")
    st.caption("Silakan lengkapi data profil dan aktivitas pelanggan di bawah ini untuk memulai analisis.")

    with st.form("churn_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### **Demografi & Akun**")
            gender = st.selectbox("Gender", options=["Male", "Female", "Unknown"])
            country = st.selectbox("Country", options=["Germany", "India", "Bangladesh", "USA", "UK"])
            city = st.selectbox("City", options=["London", "Mumbai", "Dhaka", "New York", "Delhi", "Berlin", "Hamburg"])
            acquisition_channel = st.selectbox("Acquisition Channel", options=["Organic", "Google Ads", "Facebook Ads", "Referral", "Email"])
            device_type = st.selectbox("Device Type", options=["Tablet", "Mobile", "Desktop"])
            subscription_type = st.selectbox("Subscription Type", options=["Monthly", "Annual"])
            coupon_code = st.selectbox("Coupon Code", options=["No_Coupon", "REF10", "SALE15", "NEW20"])
            payment_method = st.selectbox("Payment Method", options=["UPI", "PayPal", "SEPA", "BKash", "Card"])
            is_premium_user = st.radio("Is Premium User?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
            discount_used = st.radio("Discount Used?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            st.markdown("##### **Aktivitas & Transaksi**")
            age = st.number_input("Age", min_value=0.0, max_value=95.0, value=30.0)
            total_visits = st.number_input("Total Visits", min_value=0, max_value=31, value=5)
            avg_session_time = st.number_input("Avg Session Time (minutes)", min_value=0.0, max_value=20.0, value=8.0)
            pages_per_session = st.number_input("Pages Per Session", min_value=0.0, max_value=11.0, value=4.0)
            email_open_rate = st.number_input("Email Open Rate (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5)
            email_click_rate = st.number_input("Email Click Rate (0.0 - 0.5)", min_value=0.0, max_value=0.5, value=0.1)
            total_spent = st.number_input("Total Spent", min_value=0.0, max_value=16000.0, value=500.0)
            avg_order_value = st.number_input("Avg Order Value", min_value=0.0, max_value=155.0, value=50.0)
            support_tickets = st.number_input("Support Tickets", min_value=0, max_value=9, value=0)
            refund_requested = st.radio("Refund Requested?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            st.markdown("##### **Metrik Kepuasan & Waktu**")
            delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0, max_value=11, value=0)
            marketing_spend_per_user = st.number_input("Marketing Spend Per User", min_value=0.0, max_value=30.0, value=5.0)
            lifetime_value = st.number_input("Lifetime Value", min_value=0.0, max_value=3800.0, value=1000.0)
            last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Freq", min_value=0, max_value=14, value=2)
            satisfaction_score = st.slider("Satisfaction Score", min_value=1.0, max_value=5.0, value=4.0, step=1.0)
            nps_score = st.slider("NPS Score", min_value=0, max_value=10, value=8)
            days_to_last_purchase = st.number_input("Days to Last Purchase", min_value=0, value=15)
            signup_year = st.number_input("Signup Year", min_value=2010, max_value=2026, value=2025)
            signup_month = st.slider("Signup Month", min_value=1, max_value=12, value=6)

        submitted = st.form_submit_button("Run Churn Prediction Analysis")


# TAB 3: HASIL EVALUASI PREDIKSI
with tab_result:
    st.header("🔮 Prediction Analysis Result")
    
    if submitted:
        # Mapping Data Kategori ke Numerik (Menambahkan default fallback value jika ada input 'Unknown')
        gender_map = {"Female": 0, "Male": 1, "Unknown": 2}
        country_map = {"Bangladesh": 0, "Germany": 1, "UK": 2, "USA": 3, "India": 4}
        city_map = {"Berlin": 0, "Delhi": 1, "Dhaka": 2, "Hamburg": 3, "London": 4, "Mumbai": 5, "New York": 6}
        channel_map = {"Email": 0, "Facebook Ads": 1, "Google Ads": 2, "Organic": 3, "Referral": 4}
        device_map = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
        sub_map = {"Annual": 0, "Monthly": 1}
        coupon_map = {"NEW20": 0, "No_Coupon": 1, "REF10": 2, "SALE15": 3}
        payment_map = {"BKash": 0, "Card": 1, "PayPal": 2, "SEPA": 3, "UPI": 4}

        input_row = {
            'gender': gender_map.get(gender, 2),
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

        # Konversi ke DataFrame & Scaling
        input_df = pd.DataFrame([input_row])[feature_order]
        input_scaled = scaler.transform(input_df)
        
        # Predict menggunakan model final
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Tampilkan Hasil di Tab Analisis
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.write("### Probability Meter")
            st.metric(label="Churn Probability Score", value=f"{prediction_proba:.2%}")

        with col_res2:
            st.write("### Verdict & Action Items")
            if prediction[0] == 1:
                st.error(f"⚠️ **Warning: Pelanggan terindikasi kuat akan CHURN.**")
                st.markdown("""
                * **Rekomendasi Tindakan:** * Hubungi pelanggan secara personal melalui tim Customer Relationship Management (CRM).
                    * Berikan penawaran/diskon loyalitas khusus sebelum masa aktif paket habis.
                """)
            else:
                st.success(f"✅ **Success: Pelanggan diprediksi tetap LOYAL.**")
                st.markdown("""
                * **Rekomendasi Tindakan:**
                    * Pertahankan kualitas interaksi saat ini.
                    * Pelanggan cocok diikutsertakan ke dalam program *Upselling* produk premium.
                """)
    else:
        st.info("Silakan isi formulir di tab **📋 Prediction Form** dan tekan tombol analisis untuk melihat hasil di sini.")