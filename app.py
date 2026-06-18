import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Page Config ---
st.set_page_config(page_title="Porous Asphalt Permeability Predictor", layout="wide")
st.title("🛣️ Porous Asphalt Permeability Predictor")
st.markdown("Enter your mix design parameters on the left to predict water permeability and check JKR compliance.")
st.markdown(""" <style> thead tr th {text-align: center !important; </style> """, unsafe_allow_html=True)

# --- DATA LOADING & TRAINING ---
@st.cache_resource
def load_data_and_train_models():
    df = pd.read_excel('ML_Data_Only.xlsx')
    df = pd.get_dummies(df, columns=['Binder_Type'])
    
    target_col = 'k_cm/s'
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    performance_data = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, predictions)
        n = len(X_test_scaled)
        p = X_test_scaled.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        performance_data.append({"Model": name,"5-Fold CV R²": round(cv_r2_mean, 4), "Test Adjusted R²": round(adj_r2, 4), "RMSE": round(rmse, 4), "MAE (cm/s)": round(mae, 4)})
        
    return models["Random Forest"], scaler, feature_cols, pd.DataFrame(performance_data), X_train_scaled

rf_model, scaler, feature_cols, perf_df, X_train_scaled = load_data_and_train_models()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔧 Mix Design Parameters")
nmas = st.sidebar.number_input("NMAS (mm)", value=14.0)
pass_475 = st.sidebar.number_input("% Passing 4.75mm sieve", value=35.0)
pass_236 = st.sidebar.number_input("% Passing 2.36mm sieve", value=13.0)
pass_118 = st.sidebar.number_input("% Passing 1.18mm sieve", value=7.0)
pass_06 = st.sidebar.number_input("% Passing 0.6mm sieve", value=5.0)
porosity = st.sidebar.number_input("% Porosity", value=20)
binder_content = st.sidebar.number_input("% Binder Content", value=5.0)
binder_names = {
    1: "Normal Bitumen: PG 60/70", 
    2: "High Viscosity Modified Asphalt: PG 88-28", 
    3: "Polymer Modified Binder: PG 68", 
    4: "Polymer Modified Binder: PG 76-22", 
    5: "Viscosity Grade: VG-30"
}
selected_binder_code = st.sidebar.selectbox("Binder Type", options=[1, 2, 3, 4, 5], format_func=lambda x: binder_names[x])
blows = st.sidebar.number_input("Compaction Effort (Blows)", value=50)
feature_name_mapping = {
    'NMAS_mm': "NMAS (mm)",
    '%_Passing_4.75mm': "% Passing 4.75mm sieve",
    '%_Passing_2.36mm': "% Passing 2.36mm sieve",
    '%_Passing_1.18mm': "% Passing 1.18mm sieve",
    '%_Passing_.0.6mm': "% Passing 0.6mm sieve",
    '%_Porosity': "% Porosity",
    '%_Binder_Content': "% Binder Content",
    'C_Effort_n_blows': "Compaction Effort (Blows)"
}
for code, name in binder_names.items():
    feature_name_mapping[f'Binder_Type_{code}'] = f"Binder Type: {name}"

# --- MAPPING INPUTS ---
user_data = {col: 0.0 for col in feature_cols}
user_data['NMAS_mm'] = nmas
user_data['%_Passing_4.75mm'] = pass_475 
user_data['%_Passing_2.36mm'] = pass_236
user_data['%_Passing_1.18mm'] = pass_118
user_data['%_Passing_.0.6mm'] = pass_06
user_data['%_Porosity'] = porosity
user_data['%_Binder_Content'] = binder_content
user_data['C_Effort_n_blows'] = blows
binder_col_name = f'Binder_Type_{selected_binder_code}'
if binder_col_name in user_data: user_data[binder_col_name] = 1.0

user_df = pd.DataFrame([user_data])[feature_cols]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction & JKR Check", "📈 Feature Importance ", "🤖 Model Performance"])

with tab1:
    st.subheader("Simulation Results")
    user_scaled = scaler.transform(user_df)
    predicted_k = rf_model.predict(user_scaled)[0]
    st.metric(label="Predicted Permeability Coefficient (k)", value=f"{predicted_k:.4f} cm/s")
    
    st.divider()
    st.subheader("JKR Standard Compliance Check")
    st.markdown("**JKR Standard Range Limits Reference**")
    jkr_table_data = {
        "Mix Parameter": [
            "% Passing 4.75mm (Grading A)", "% Passing 4.75mm (Grading B)",
            "% Passing 2.36mm (Grading A)", "% Passing 2.36mm (Grading B)",
            "Porosity (%)", "Binder Content (%)"
        ],
        "Minimum Limit": [30.0, 10.0, 5.0, 5.0, 18.0, 4.0],
        "Maximum Limit": [50.0, 25.0, 15.0, 10.0, 25.0, 6.0]
    }
    jkr_df = pd.DataFrame(jkr_table_data)
    jkr_df["Minimum Limit"] = jkr_df["Minimum Limit"].map("{:.1f}".format)
    jkr_df["Maximum Limit"] = jkr_df["Maximum Limit"].map("{:.1f}".format)
    st.table(jkr_df)

    jkr_limits1 = {'4.75': (30, 50, 10, 25), '2.36': (5, 15, 5, 10)}
    fits_A = (30 <= pass_475 <= 50) and (5 <= pass_236 <= 15)
    fits_B = (10 <= pass_475 <= 25) and (5 <= pass_236 <= 10)
    if fits_A: st.success("✅ Gradation Complies with JKR Grading A"); active = 'A'
    elif fits_B: st.success("✅ Gradation Complies with JKR Grading B"); active = 'B'
    else: st.warning("⚠️ Gradation does not comply with either JKR Grading A or Grading B."); active = None
    #
    jkr_limits2 = {'porosity': (18, 25)}
    fits_C = (18 <= porosity <= 25)
    if fits_C: st.success("✅ Porosity Complies with JKR Standards"); active = 'C'
    else: st.warning("⚠️ Porosity does not comply with JKR Standards."); active = None
    #
    jkr_limits3 = {'binder content': (4, 6)}
    fits_D = (4 <= binder_content <= 6)
    if fits_D: st.success("✅ Binder Content Complies with JKR Standards"); active = 'D'
    else: st.warning("⚠️ Binder Content does not comply with JKR Standards."); active = None
    #
    minimum_k = 0.116
    if predicted_k >= minimum_k:
        st.success("✅ **MIX APPROVED**: The simulated permeability meets the recommended value (≥ 0.116 cm/s).")
    else:
        st.error(f"❌ **MIX REJECTED**: The permeability is below the recommended value. Consider increasing porosity or adjusting the gradation.")
    #
    st.divider()
    st.subheader("Sample Gradation Curve")   
    sieve_sizes = [4.75, 2.36, 1.18, 0.6]
    passing_values = [pass_475, pass_236, pass_118, pass_06]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sieve_sizes, passing_values, marker='d', linestyle='-', color='b')
    ax.set_xscale('log') 
    ax.set_xlabel("Sieve Size (mm)")
    ax.set_ylabel("% Passing")
    ax.set_title("User Mix Gradation")
    ax.set_xticks(sieve_sizes)
    ax.set_xticklabels([str(s) for s in sieve_sizes], rotation=45)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Factors Affecting Permeability")
    importances = rf_model.feature_importances_

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        def clean_autopct(pct):
            return ('%1.1f%%' % pct) if pct > 2 else ''
        num_features = len(importances)
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, num_features)]
        wedges, texts, autotexts = ax2.pie(importances, autopct=clean_autopct, startangle=140, colors=colors)
        ax2.axis('equal')
        plt.tight_layout()
        st.pyplot(fig2)
    with col2 :
        st.markdown("**Importance Ranking Table**")
        importance_df = pd.DataFrame({
            'Parameter': feature_cols,
            'Importance (%)': np.round(importances * 100, 2)
        }).sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)
        importance_df['Parameter'] = importance_df['Parameter'].map(feature_name_mapping)
        importance_df['Importance (%)'] = importance_df['Importance (%)'].map("{:.2f}".format)
        st.table(importance_df.set_index('Parameter'))

with tab3:
    st.subheader("Machine Learning Algorithm Performance Comparison")
    st.table(perf_df.set_index('Model'))
