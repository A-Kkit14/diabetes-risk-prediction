import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class ConvertAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def map_age_group(age_code):
            if age_code <= 4:
                return 1   # 18–39
            elif age_code <= 7:
                return 2   # 40–54
            elif age_code <= 9:
                return 3   # 55–64
            else:
                return 4   # 65+

        X = X.apply(map_age_group)
        return X.to_frame(name="AgeGroup")

    def get_feature_names_out(self, input_features=None):
        return ["AgeGroup"]

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Diabetes Dashboard",
    layout="wide"
)

st.title("Diabetes Health Indicators Dashboard")
st.markdown("Phân tích dữ liệu sức khỏe & nguy cơ tiểu đường (BRFSS 2015)")

# ===============================
# LOAD DATA
# ===============================
def map_age_group(age_code):
    if age_code <= 4:
        return "18–39"
    elif age_code <= 7:
        return "40–54"
    elif age_code <= 9:
        return "55–64"
    else:
        return "65+"

@st.cache_data
def load_data():
    return pd.read_csv("Code/data_decoded.csv")  # đổi path nếu cần

df = load_data()
df2 = pd.read_csv("Code/data_clean.csv")
df2 = df2.drop(columns=['Unnamed: 0'])

ordinal_cols = ["GenHlth", "Education", "Income", "AgeGroup"]

agegroup_map = {
    "18–39": 1,
    "40–54": 2,
    "55–64": 3,
    "65+": 4
}
df2["AgeGroup"] = df2["Age"].apply(map_age_group)
df_tmp = df2.copy()
df_tmp["AgeGroup_num"] = df_tmp["AgeGroup"].map(agegroup_map)


def plot_binary_diabetes_pie(df, col):
    values = ["Yes", "No"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, val in zip(axes, values):
        subset = df[df[col] == val]

        vc = subset["Diabetes_binary"].value_counts()

        if vc.shape[0] < 2:
            ax.text(
                0.5, 0.5,
                "Not enough data",
                ha="center", va="center"
            )
            ax.axis("off")
            ax.set_title(f"{col} = {val}")
            continue

        ax.pie(
            vc.values,
            labels=vc.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "black"}
        )
        ax.set_title(f"{col} = {val}")

    plt.suptitle(f"Diabetes Distribution by {col}", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# ===============================
# COLUMN GROUPS
# ===============================
binary_cols = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'DiffWalk', 'Sex'
]

ordinal_cols = ["GenHlth", "Education", "Income", "AgeGroup"]
continuous_cols = ["BMI", "MentHlth", "PhysHlth"]

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Chọn nội dung",
    [
        "Overview",
        "Ordinal vs Diabetes",
        "Binary vs Diabetes",
        "Continuous vs Diabetes",
        "Correlation",
        "Top Risk Factors",
        "Diabetes Risk Prediction"
    ]
)

# ======================
# OVERVIEW
# ======================
if page == "Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric(
        "Diabetes (%)",
        round((df["Diabetes_binary"] == "Diabetes").mean() * 100, 2)
    )
    col3.metric(
        "No Diabetes (%)",
        round((df["Diabetes_binary"] == "No Diabetes").mean() * 100, 2)
    )

    st.subheader("Target Distribution")

    col_left, col_right = st.columns(2)

    # ===== BAR CHART =====
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(
            data=df,
            x="Diabetes_binary",
            ax=ax1
        )
        ax1.set_title("Diabetes Distribution (Count)")
        ax1.set_xlabel("")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    # ===== PIE CHART =====
    with col_right:
        target_counts = df["Diabetes_binary"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(
            target_counts.values,
            labels=target_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "black"}
        )
        ax2.set_title("Diabetes Distribution (%)")
        ax2.axis("equal")
        st.pyplot(fig2)

# ======================
# ORDINAL vs DIABETES
# ======================
elif page == "Ordinal vs Diabetes":
    st.header("Ordinal Variables vs Diabetes")

    fig, axes = plt.subplots(2, 2, figsize=(18,10))
    axes = axes.flatten()

    for ax, col in zip(axes, ordinal_cols):
        sns.countplot(
            data=df,
            x=col,
            hue="Diabetes_binary",
            ax=ax
        )
        ax.set_title(f"{col} vs Diabetes")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["No Diabetes", "Diabetes"],
        title="Diabetes",
        loc="upper right"
    )

    plt.tight_layout()
    st.pyplot(fig)

# ======================
# BINARY vs DIABETES
# ======================
elif page == "Binary vs Diabetes":
    st.header("Binary Risk Factors vs Diabetes")

    tab1, tab2 = st.tabs(["📊 Bar chart", "🥧 Pie chart (%)"])

    # ======================
    # TAB 1: BAR CHART
    # ======================
    with tab1:
        selected_cols = st.multiselect(
            "Chọn biến nhị phân (Bar chart)",
            binary_cols,
            default=binary_cols[:4]
        )

        if len(selected_cols) > 0:
            n_cols = 3
            n_rows = math.ceil(len(selected_cols) / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(18, n_rows * 4)
            )
            axes = axes.flatten()

            for ax, col in zip(axes, selected_cols):
                sns.countplot(
                    data=df,
                    x=col,
                    hue="Diabetes_binary",
                    ax=ax
                )
                ax.set_title(col)
                ax.set_xlabel("")

            for i in range(len(selected_cols), len(axes)):
                fig.delaxes(axes[i])

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                ["No Diabetes", "Diabetes"],
                title="Diabetes",
                loc="upper right"
            )

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Vui lòng chọn ít nhất 1 biến.")

    # ======================
    # TAB 2: PIE CHART
    # ======================
    with tab2:
        selected_col = st.selectbox(
            "Chọn biến nhị phân (Pie chart)",
            binary_cols
        )

        plot_binary_diabetes_pie(df, selected_col)

# ======================
# CONTINUOUS vs DIABETES
# ======================
elif page == "Continuous vs Diabetes":
    st.header("Continuous Variables vs Diabetes")

    fig, axes = plt.subplots(
        1,
        len(continuous_cols),
        figsize=(18,5)
    )

    for ax, col in zip(axes, continuous_cols):
        sns.kdeplot(
            data=df,
            x=col,
            hue="Diabetes_binary",
            fill=True,
            common_norm=False,
            alpha=0.4,
            ax=ax
        )
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)

    plt.tight_layout()
    st.pyplot(fig)

# ======================
# CORRELATION
# ======================
elif page == "Correlation":
    st.header("Correlation Analysis")

    st.markdown("""
    **Phương pháp sử dụng:**
    - Binary ↔ Binary: **Phi correlation**
    - Binary ↔ Ordinal: **Spearman correlation**
    - Ordinal ↔ Ordinal: **Spearman correlation**
    - Continuous ↔ Continuous: **Spearman correlation**
    - Binary ↔ Continuous: **Point-Biserial correlation**
    """)

    # ======================
    # PREPARE DATA
    # ======================
    df_corr = df_tmp.copy()

   

    # Encode ordinal if needed
    ordinal_cols_compare = ["GenHlth", "Education", "Income", "AgeGroup_num"]

    # ======================
    # TABS
    # ======================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Binary ↔ Binary",
        "Binary ↔ Ordinal",
        "Continuous ↔ Continuous",
        "Binary ↔ Continuous",
        "Ordinal ↔ Continuous"
    ])

    # ======================
    # TAB 1: Binary ↔ Binary (Phi)
    # ======================
    with tab1:
        corr_bb = df_corr[binary_cols].corr()

        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(
            corr_bb,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5
        )
        ax.set_title("Phi Correlation: Binary vs Binary")
        st.pyplot(fig)

    # ======================
    # TAB 2: Binary ↔ Ordinal (Spearman)
    # ======================
    with tab2:
        heatmap_cols = binary_cols + ordinal_cols_compare
        corr_bo = df_corr[heatmap_cols].corr(method="spearman")

        mask = np.triu(np.ones_like(corr_bo, dtype=bool))

        fig, ax = plt.subplots(figsize=(14,10))
        sns.heatmap(
            corr_bo,
            mask=mask,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Spearman Correlation: Binary & Ordinal Variables")
        st.pyplot(fig)

    # ======================
    # TAB 3: Continuous ↔ Continuous (Spearman)
    # ======================
    with tab3:
        corr_cc = df_corr[continuous_cols].corr(method="spearman")

        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(
            corr_cc,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            center=0
        )
        ax.set_title("Spearman Correlation: Continuous Variables")
        st.pyplot(fig)

    # ======================
    # TAB 4: Binary ↔ Continuous (Point-Biserial)
    # ======================
    with tab4:
        from scipy.stats import pointbiserialr

        corr_bc = pd.DataFrame(
            index=binary_cols,
            columns=continuous_cols,
            dtype=float
        )

        for b in binary_cols:
            for c in continuous_cols:
                corr_bc.loc[b, c] = pointbiserialr(
                    df_corr[b],
                    df_corr[c]
                ).correlation

        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(
            corr_bc,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f"
        )
        ax.set_title("Point-Biserial Correlation: Binary vs Continuous")
        st.pyplot(fig)

    # ======================
    # TAB 5: Ordinal ↔ Continuous (Spearman)
    # ======================
    with tab5:
        corr_oc = pd.DataFrame(
            index=ordinal_cols_compare,
            columns=continuous_cols,
            dtype=float
        )

        for o in ordinal_cols_compare:
            for c in continuous_cols:
                corr_oc.loc[o, c] = df_corr[[o, c]].corr(
                    method="spearman"
                ).iloc[0, 1]

        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(
            corr_oc,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f"
        )
        ax.set_title("Spearman Correlation: Ordinal vs Continuous")
        st.pyplot(fig)
#------------------------------------


elif page == "Top Risk Factors":
    st.header("Top Binary Risk Factors for Diabetes")

    st.markdown("""
    **Cách tính:**  
    Tỷ lệ Diabetes trong **nhóm Yes** của từng yếu tố nguy cơ nhị phân.
    """)

    risk_rates = []

    for col in binary_cols:
        subset = df[df[col] == "Yes"]

        if len(subset) == 0:
            continue

        rate = (subset["Diabetes_binary"] == "Diabetes").mean() * 100

        risk_rates.append({
            "Risk Factor": col,
            "Diabetes Rate (%)": rate
        })

    risk_df = (
        pd.DataFrame(risk_rates)
        .sort_values("Diabetes Rate (%)", ascending=False)
    )

    # ===== BAR CHART =====
    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        data=risk_df,
        x="Diabetes Rate (%)",
        y="Risk Factor",
        ax=ax
    )

    ax.set_title("Ranking of Binary Risk Factors by Diabetes Rate")
    ax.set_xlabel("Diabetes Rate (%)")
    ax.set_ylabel("")

    for i, v in enumerate(risk_df["Diabetes Rate (%)"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

    st.pyplot(fig)

    # ===== TABLE =====
    st.subheader("Detailed Table")
    st.dataframe(
        risk_df.style.format({"Diabetes Rate (%)": "{:.2f}%"})
    )
#---------------------------------------------------------------------------------------------------------
elif page == "Diabetes Risk Prediction":
    import joblib

    st.header("Diabetes Risk Prediction")
    st.markdown("Người dùng nhập thông tin sức khỏe để dự đoán nguy cơ mắc bệnh tiểu đường.")

    # ======================
    # LOAD MODEL
    # ======================
    @st.cache_resource
    def load_model():
        return joblib.load("Code/dt_tuned_pipeline.joblib")

    model = load_model()

    # ======================
    # USER INPUT
    # ======================
    st.subheader("📋 Điền thông tin cá nhân")

    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox("Giới tính", ["Male", "Female"])
        height_cm = st.number_input("Chiều cao (cm)", 100, 280, 165)
        weight_kg = st.number_input("Cân nặng (kg)", 30.0, 300.0, 60.0)

    with col2:
        age_group_label = st.selectbox(
            "Nhóm tuổi",
            ["18–39", "40–54", "55–64", "65+"]
        )

        genhlth = st.selectbox(
            "Đánh giá sức khỏe tổng quát",
            [
                "Excellent",
                "Very good",
                "Good",
                "Fair",
                "Poor"
            ]
        )

    st.subheader("Bạn có...")

    col3, col4 = st.columns(2)

    with col3:
        high_bp = st.radio("Huyết áp cao?", ["No", "Yes"])
    with col4:
        high_chol = st.radio("Cholesterol cao?", ["No", "Yes"])

    # ======================
    # MAP INPUT → MODEL FORMAT
    # ======================

    # ---- BMI ----
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    # ---- Age code (1–13) → pipeline tự convert ----
    age_ui_map = {
        "18–39": 3,   # đại diện nhóm 1–4
        "40–54": 6,   # đại diện nhóm 5–7
        "55–64": 8,   # đại diện nhóm 8–9
        "65+": 11     # đại diện nhóm 10–13
    }
    age_code = age_ui_map[age_group_label]

    # ---- GenHlth ----
    genhlth_map = {
        "Excellent": 1,
        "Very good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    }

    # ---- Binary ----
    binary_map = {"No": 0, "Yes": 1}

    # ======================
    # BUILD INPUT DATA (MATCH MODEL FEATURES)
    # ======================
    input_data = {
        col: df2[col].mode()[0] for col in model.feature_names_in_
    }

    input_data.update({
        "GenHlth": genhlth_map[genhlth],
        "BMI": bmi,
        "Age": age_code,
        "HighBP": binary_map[high_bp],
        "HighChol": binary_map[high_chol]
    })

    input_df = pd.DataFrame([input_data])

    # ======================
    # PREDICT
    # ======================
    if st.button("🔍 Dự đoán nguy cơ"):
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error(f"⚠️ Nguy cơ mắc tiểu đường: **{proba*100:.1f}%**")
        else:
            st.success(f"✅ Nguy cơ thấp: **{(1-proba)*100:.1f}%**")

        st.caption("⚠️ Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán y tế.")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("📊 Diabetes Health Indicators Dashboard | Streamlit")
