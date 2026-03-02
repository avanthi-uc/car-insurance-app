import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns


st.set_page_config(
    page_title="Car Insurance Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ---------------- Main Background ---------------- */
.stApp {
    background-color: #f5f3ff;   /* soft lavender */
    color: #2e1065;
}



/* ---------------- Headers ---------------- */
h1 {
    color: #4c1d95;
    font-weight: 700;
}

h2, h3 {
    color: #6d28d9;
}

/* ---------------- KPI Cards ---------------- */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 16px rgba(109, 40, 217, 0.15);
    border-left: 6px solid #7c3aed;
}

/* ---------------- Buttons ---------------- */
.stButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 8px 18px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #6d28d9;
}

/* ---------------- Download Button ---------------- */
.stDownloadButton>button {
    background-color: #8b5cf6;
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* ---------------- DataFrame ---------------- */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 8px;
}

/* ---------------- Expander ---------------- */
details {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(124, 58, 237, 0.1);
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* ---------------- Main Background ---------------- */
.stApp {
    background-color: #f5f3ff;
    color: #2e1065;
}

/* ---------------- Sidebar (Soft Violet) ---------------- */
section[data-testid="stSidebar"] {
    background-color: #ede9fe;   /* soft violet */
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #4c1d95 !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: #5b21b6 !important;
    font-weight: 700;
}

/* ---------------- Highlight selected navigation ---------------- */
div[role="radiogroup"] label {
    padding: 8px;
    border-radius: 8px;
    margin-bottom: 6px;
}

div[role="radiogroup"] input:checked + div {
    background-color: #c4b5fd !important;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚗 Car Insurance App")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Introduction", "📁 Dataset", "📊 EDA", "🔮 Predict", "📁 Predict from CSV"]
)


st.title("🚗 Car Insurance Claim Prediction")


if page == "🏠 Introduction":
    

    st.markdown("""
    ### 📌 Project Overview
    This project predicts whether a customer is likely to file a **car insurance claim**
    based on demographic, vehicle, and policy-related features.

    ### 🎯 Business Use Cases
    - Fraud prevention  
    - Risk-based pricing  
    - Better customer targeting  
    - Operational efficiency  

    ### 🧠 Models Used
    - Logistic Regression  
    - Random Forest  
    - XGBoost  
    - LightGBM

    ### 📊 Evaluation Metrics
    - Accuracy  
    - Precision 
    - Recall  
    - Confusion Matrix  

    ---
    
    """)


elif page == "🔮 Predict":

    import joblib
    import pandas as pd

    model = joblib.load("models/xgb_sm1_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    top_features = joblib.load("models/model_features.pkl")

    st.title("Insurance Claim Risk Assessment")
    st.markdown("Enter vehicle and policyholder details to evaluate claim risk.")

    st.markdown("---")

    # 🔹 Section Header
    st.subheader("Vehicle & Policy Details")


    # Encoding Maps

    segment_map = {
        'A':0, 'B1':1, 'B2':2, 'C1':3, 'C2':4, 'Utility':5
    }

    transmission_map = {
        'Manual':0,
        'Automatic':1
    }

    rear_brake_map = {
        'Drum':0,
        'Disc':1
    }

    area_clusters = [f"C{i}" for i in range(1,23)]
    area_cluster_map = {val: idx for idx, val in enumerate(area_clusters)}

    model_map = {
        'M1':0,'M10':1,'M11':2,'M2':3,'M3':4,
        'M4':5,'M5':6,'M6':7,'M7':8,'M8':9,'M9':10
    }

    # Create 3-column layout for better balance
    col1, col2, col3 = st.columns(3)

    with col1:
        age_of_car = st.number_input("Age of Car (Years)", min_value=0.0)
        displacement = st.number_input("Engine Displacement (cc)")
        cylinder = st.selectbox(
                "Number of Cylinders",
                options=[3, 4]
            )
        max_power_bhp = st.number_input("Max Power (BHP)")
        gross_weight = st.number_input("Gross Weight (kg)")

        segment_choice = st.selectbox("Segment", list(segment_map.keys()))
        segment = segment_map[segment_choice]

    with col2:
        age_of_policyholder = st.number_input("Policyholder Age", min_value=18.0)
        policy_tenure = st.number_input("Policy Tenure")
        max_torque_nm = st.number_input("Max Torque (Nm)")
        max_power_rpm = st.number_input("Max Power RPM")
        width = st.number_input("Car Width (mm)")

        transmission_choice = st.selectbox("Transmission Type", list(transmission_map.keys()))
        transmission_type = transmission_map[transmission_choice]

    with col3:
        length = st.number_input("Car Length (mm)")
        height = st.number_input("Car Height (mm)")
        max_torque_rpm = st.number_input("Max Torque RPM")
        population_density = st.number_input("Population Density")
        gear_box = st.selectbox(
                    "Gear Box",
                    options=[5, 6],
                    index=0  # default = 5
                )

        area_choice = st.selectbox("Area Cluster", area_clusters)
        area_cluster = area_cluster_map[area_choice]

        model_choice = st.selectbox("Model", list(model_map.keys()))
        model_input = model_map[model_choice]

        rear_choice = st.selectbox("Rear Brakes Type", list(rear_brake_map.keys()))
        rear_brakes_type = rear_brake_map[rear_choice]

    st.markdown("---")

    # 🔹 Predict Button Centered
    predict_col1, predict_col2, predict_col3 = st.columns([1,2,1])

    with predict_col2:
        predict_button = st.button("🔮 Evaluate Claim Risk", use_container_width=True)

    if predict_button:

        input_data = {
            "age_of_car": age_of_car,
            "age_of_policyholder": age_of_policyholder,
            "displacement": displacement,
            "max_torque_nm": max_torque_nm,
            "policy_tenure": policy_tenure,
            "length": length,
            "cylinder": cylinder,
            "height": height,
            "model": model_input,
            "gross_weight": gross_weight,
            "max_power_rpm": max_power_rpm,
            "transmission_type": transmission_type,
            "max_power_bhp": max_power_bhp,
            "width": width,
            "area_cluster": area_cluster,
            "max_torque_rpm": max_torque_rpm,
            "population_density": population_density,
            "gear_box": gear_box,
            "segment": segment,
            "rear_brakes_type": rear_brakes_type
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=top_features)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("## 📊 Risk Evaluation Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠ High Risk of Claim")
            else:
                st.success("✅ Low Risk of Claim")

        with result_col2:
            st.metric("Claim Probability", f"{probability*100:.2f}%")

        # Progress Bar Visualization
        st.progress(float(probability))

        st.markdown("---")
        st.caption("Model: XGBoost | Features: Top 20 ")


elif page == "📁 Predict from CSV":


    model = joblib.load("models/xgb_sm1_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    top_features = joblib.load("models/model_features.pkl")
 

    st.title("📁 Prediction from CSV")
    st.write("Upload the test CSV file to predict insurance claims.")

    uploaded_file = st.file_uploader(
        "Upload test.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Load CSV
        # Read file
        data = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Uploaded Data Preview")
        st.write(data.head())

        policy_ids = data["policy_id"].copy()

        
        data['policy_tenure'] = data['policy_tenure'] * 10
        data['age_of_car'] = data['age_of_car'] * 100
        data['age_of_policyholder'] = data['age_of_policyholder'] * 10

        # Extract max_torque
        data[['max_torque_nm', 'max_torque_rpm']] = (
            data['max_torque']
            .str.replace('Nm', '', regex=False)
            .str.replace('rpm', '', regex=False)
            .str.split('@', expand=True)
        )

        data['max_torque_nm'] = pd.to_numeric(data['max_torque_nm'])
        data['max_torque_rpm'] = pd.to_numeric(data['max_torque_rpm'])

        data.drop(columns=['max_torque'], inplace=True)

        # Extract max_power
        data[['max_power_bhp', 'max_power_rpm']] = (
            data['max_power']
            .str.extract(r'(\d+\.?\d*)\s*bhp.*?(\d+)')
        )

        data[['max_power_bhp', 'max_power_rpm']] = data[
            ['max_power_bhp', 'max_power_rpm']
        ].astype(float)

        data.drop(columns=['max_power'], inplace=True)

        # Apply Encodings

        # area_cluster C1–C22
        area_mapping = {f"C{i}": i-1 for i in range(1,23)}
        data['area_cluster'] = data['area_cluster'].map(area_mapping)

        # segment mapping
        data['segment'] = data['segment'].map({
            'A':0, 'B1':1, 'B2':2,
            'C1':3, 'C2':4, 'Utility':5
        })

        # model mapping
        data['model'] = data['model'].map({
            'M1':0,'M10':1,'M11':2,'M2':3,'M3':4,
            'M4':5,'M5':6,'M6':7,'M7':8,'M8':9,'M9':10
        })

        # transmission_type
        data['transmission_type'] = data['transmission_type'].map({
            'Manual':0,
            'Automatic':1
        })

        # rear_brakes_type
        data['rear_brakes_type'] = data['rear_brakes_type'].map({
            'Drum':0,
            'Disc':1
        })

        # Select Top 20 Features
        input_data = data[top_features]

        # Scale
        input_scaled = scaler.transform(input_data)

        
        # Get probabilities
        probabilities = model.predict_proba(input_scaled)[:, 1]

        # Custom threshold
        threshold = 0.45
        predictions = (probabilities > threshold).astype(int)


        # Output
        output = pd.DataFrame({
        "policy_id": policy_ids,
        "is_claim": predictions,
        "claim_probability": probabilities
    })

        st.subheader("Prediction Output")
        st.dataframe(output)

        csv = output.to_csv(index=False).encode("utf-8")
        

if page == "📁 Dataset":

    st.title("📁 Training Dataset Overview")

    train_data = pd.read_csv("train.csv")

    # KPI CARDS
    
    total_records = len(train_data)
    total_features = train_data.shape[1]
    claim_rate = train_data['is_claim'].mean() * 100
    missing_values = train_data.isnull().sum().sum()

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Total Records", f"{total_records:,}")

    with kpi2:
        st.metric("Total Features", total_features)

    with kpi3:
        st.metric("Claim Rate", f"{claim_rate:.2f}%")



    st.markdown("---")

    #  Dataset Preview Section
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(train_data.head(50), use_container_width=True)

    st.markdown("---")


        # Styled Explanation Section
    
    with st.expander("📖 Dataset Explanation (Click to Expand)"):

        st.markdown("""
        ### 🎯 Target Variable

        **is_claim** → Binary Classification Target  
        - **0** → No Claim  
        - **1** → Claim  

        ---

        ### 📋 Feature Descriptions

        **Policy Information**
        - `policy_id` → Unique policyholder ID  
        - `policy_tenure` → Policy duration  

        **Policyholder Information**
        - `age_of_policyholder` → Normalized age  
        - `area_cluster` → Area cluster  
        - `population_density` → City population density  

        **Vehicle Information**
        - `make` → Manufacturer  
        - `segment` → Vehicle segment  
        - `model` → Vehicle model  
        - `fuel_type` → Fuel type  
        - `engine_type` → Engine type  
        - `airbags` → Number of airbags  
        - `ncap_rating` → Safety rating  

        **Performance Metrics**
        - `max_torque` → Torque (Nm@rpm)  
        - `max_power` → Power (bhp@rpm)  
        - `displacement` → Engine displacement  
        - `cylinder` → Number of cylinders  
        - `gear_box` → Number of gears  

        **Dimensions**
        - `length`, `width`, `height` → Vehicle dimensions  
        - `gross_weight` → Maximum vehicle weight  

        **Safety Features (Binary Flags)**
        - ESC, TPMS, Parking Sensors, Brake Assist  
        - Power Steering, Central Locking, etc.

        ---
        """)


elif page == "📊 EDA":

    st.markdown("## 📊 Exploratory Data Analysis")

    car = pd.read_csv("train.csv")
    car['policy_tenure'] = car['policy_tenure'] * 10
    car['age_of_car'] = car['age_of_car'] * 100
    car['age_of_policyholder'] = car['age_of_policyholder'] * 10
    car['age_of_policyholder'] = car['age_of_policyholder']  * 6.25

    # -------------------------------------
    # Horizontal Tabs
    # -------------------------------------
    tab1, tab2, tab3 , tab4 = st.tabs(["📊 Counts", "📈 KPI", "🔍 Binary Features" , "🧠 Patterns"])

    # =========================================================
    # TAB 1 — COUNTS
    # =========================================================
    with tab1:

        st.subheader("Feature Count Summary")

        feature_options = {
            "Class (is_claim)": "is_claim",
            "Make": "make",
            "Segment": "segment",
            "Model": "model",
            "Fuel Type": "fuel_type",
            "Engine Type": "engine_type",
            "Airbags": "airbags",
            "Cylinders": "cylinder",
            "Transmission Type": "transmission_type",
            "Gear Box": "gear_box",
            "Steering Type": "steering_type",
            "NCAP Rating" : "ncap_rating",
            "Area" : 'area_cluster'
        }

        selected_feature = st.selectbox(
            "Select Feature",
            list(feature_options.keys())
        )

        feature_column = feature_options[selected_feature]

        counts = car[feature_column].value_counts().reset_index()
        counts.columns = [feature_column, "Count"]

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.dataframe(counts, use_container_width=True)

        with col2:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(y=feature_column, data=car, order=car[feature_column].value_counts().index, ax=ax)
            ax.set_title(f"{selected_feature} Distribution")
            st.pyplot(fig, use_container_width=True)


    # =========================================================
    # TAB 2 — KPI
    # =========================================================
        with tab2:

            st.subheader("Dataset KPIs")

            total_records = len(car)
            total_features = car.shape[1]
            claim_rate = car['is_claim'].mean() * 100
            missing_values = car.isnull().sum().sum()

            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Total Records", f"{total_records:,}")
            k2.metric("Total Features", total_features)
            k3.metric("Claim Rate", f"{claim_rate:.2f}%")
            k4.metric("Missing Values", missing_values)

            st.markdown("---")

            st.subheader("📈 Key Statistical Indicators")

            # -----------------------------
            # Prepare Required Columns
            # -----------------------------
            kpi_columns = [
                "length",
                "width",
                "height",
                "gross_weight",
                "turning_radius",
                "displacement",
                "age_of_car",
                "age_of_policyholder",
                "policy_tenure"
            ]

            # Compute statistics
            stats = car[kpi_columns].agg(['min', 'mean', 'max']).T
            stats = stats.round(2)

            st.markdown("### 🚗 Vehicle Dimensions")

            col1, col2, col3 = st.columns(3)

            col1.metric("Length (mm)", 
                        f"Min: {stats.loc['length','min']}", 
                        f"Mean: {stats.loc['length','mean']} | Max: {stats.loc['length','max']}")

            col2.metric("Width (mm)", 
                        f"Min: {stats.loc['width','min']}", 
                        f"Mean: {stats.loc['width','mean']} | Max: {stats.loc['width','max']}")

            col3.metric("Height (mm)", 
                        f"Min: {stats.loc['height','min']}", 
                        f"Mean: {stats.loc['height','mean']} | Max: {stats.loc['height','max']}")

            st.markdown("### ⚖ Vehicle Specifications")

            col4, col5, col6 = st.columns(3)

            col4.metric("Gross Weight (kg)", 
                        f"Min: {stats.loc['gross_weight','min']}", 
                        f"Mean: {stats.loc['gross_weight','mean']} | Max: {stats.loc['gross_weight','max']}")

            col5.metric("Turning Radius (m)", 
                        f"Min: {stats.loc['turning_radius','min']}", 
                        f"Mean: {stats.loc['turning_radius','mean']} | Max: {stats.loc['turning_radius','max']}")

            col6.metric("Displacement (cc)", 
                        f"Min: {stats.loc['displacement','min']}", 
                        f"Mean: {stats.loc['displacement','mean']} | Max: {stats.loc['displacement','max']}")

            st.markdown("### 👤 Policy & Driver Information")

            col7, col8, col9 = st.columns(3)

            col7.metric("Age of Car", 
                        f"Min: {stats.loc['age_of_car','min']}", 
                        f"Mean: {stats.loc['age_of_car','mean']} | Max: {stats.loc['age_of_car','max']}")

            col8.metric("Policyholder Age", 
                        f"Min: {stats.loc['age_of_policyholder','min']}", 
                        f"Mean: {stats.loc['age_of_policyholder','mean']} | Max: {stats.loc['age_of_policyholder','max']}")

            col9.metric("Policy Tenure", 
                        f"Min: {stats.loc['policy_tenure','min']}", 
                        f"Mean: {stats.loc['policy_tenure','mean']} | Max: {stats.loc['policy_tenure','max']}")
            

            st.markdown("---")
            st.subheader("🏙 Area Cluster vs Population Density")

            # Aggregate population density by cluster
            cluster_population = (
            car.groupby('area_cluster')['population_density']
            .mean()
            .reset_index()
            .rename(columns={'population_density': 'Population Density'})
            )

            st.dataframe(cluster_population, use_container_width=True)


        # =========================================================
        # TAB 3 — INTERACTIONS
        # =========================================================
        with tab3:

            st.subheader("Binary Features")

            st.markdown("## 🔘 Binary Feature Explorer")

            car = pd.read_csv("train.csv")

            binary_columns = [
                "is_esc",
                "is_adjustable_steering",
                "is_tpms",
                "is_parking_sensors",
                "is_parking_camera",
                "rear_brakes_type",
                "is_front_fog_lights",
                "is_rear_window_wiper",
                "is_rear_window_washer",
                "is_rear_window_defogger",
                "is_brake_assist",
                "is_power_door_lock",
                "is_central_locking",
                "is_power_steering",
                "is_driver_seat_height_adjustable",
                "is_day_night_rear_view_mirror",
                "is_ecw",
                "is_speed_alert"
            ]

            selected_binary = st.selectbox(
                "Select Binary Feature",
                binary_columns
            )

            st.markdown("---")

            # ===============================
            # Counts
            # ===============================
            counts = car[selected_binary].value_counts().reset_index()
            counts.columns = [selected_binary, "Count"]

            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.subheader("📋 Counts")
                st.dataframe(counts, use_container_width=True)

            # ===============================
            # Visualization
            # ===============================
            with col2:
                st.subheader("📊 Distribution")

                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(5,4))

                sns.countplot(
                    x=selected_binary,
                    data=car,
                    ax=ax
                )

                ax.set_title(f"{selected_binary} Distribution")
                ax.set_xticklabels(["No (0)", "Yes (1)"])

                st.pyplot(fig, use_container_width=True)


                st.markdown("### 🔘 Binary Feature Profile for Claim Cases")

                binary_columns = [
                    "is_esc",
                    "is_adjustable_steering",
                    "is_tpms",
                    "is_parking_sensors",
                    "is_parking_camera",
                    "rear_brakes_type",
                    "is_front_fog_lights",
                    "is_rear_window_wiper",
                    "is_rear_window_washer",
                    "is_rear_window_defogger",
                    "is_brake_assist",
                    "is_power_door_locks",
                    "is_central_locking",
                    "is_power_steering",
                    "is_driver_seat_height_adjustable",
                    "is_day_night_rear_view_mirror",
                    "is_ecw",
                    "is_speed_alert"
                ]

                # Convert Yes/No → 0/1
                for col in binary_columns:
                    if car[col].dtype == "object":
                        car[col] = car[col].map({"No": 0, "Yes": 1})

                claims_only = car[car["is_claim"] == 1]

                summary_data = []

                for col in binary_columns:
                    claim_yes_pct = claims_only[col].mean() * 100
                    overall_yes_pct = car[col].mean() * 100
                    difference = claim_yes_pct - overall_yes_pct

                    summary_data.append([
                        col,
                        round(claim_yes_pct, 2),
                        round(overall_yes_pct, 2),
                        round(difference, 2)
                    ])

                binary_claim_summary = pd.DataFrame(
                    summary_data,
                    columns=[
                        "Feature",
                        "% Yes in Claims",
                        "% Yes Overall",
                        "Difference"
                    ]
                ).sort_values("Difference", ascending=False)

                st.dataframe(binary_claim_summary, use_container_width=True)


                st.markdown("""
                ### 📘 Interpretation: Binary Feature Profile for Claim Cases

                This table compares how frequently each binary feature appears in claim cases versus the overall dataset.

                - **% Yes in Claims** → Percentage of claim cases where the feature is present.
                - **% Yes Overall** → Percentage of all policies where the feature is present.
                - **Difference** → Indicates whether the feature appears more or less frequently in claim cases compared to the overall distribution.

                - A **positive difference** means the feature appears more often among claim cases.
                - A **negative difference** means the feature appears less often among claim cases.
                - A difference close to zero suggests little to no association.

                  
                This analysis shows **association, not causation**.  
                A feature appearing more frequently in claim cases does not necessarily mean it increases risk — it may simply be more common overall.

""")


        # =========================================================
        # TAB 4 — CORRELATIONS & PATTERNS
        # =========================================================
        with tab4:

            st.subheader("🧠 Correlation & Pattern Analysis")

            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            # -----------------------------
            # 1️⃣ Correlation Heatmap
            # -----------------------------
            st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")

            numeric_cols = car.select_dtypes(include=np.number).columns
            corr_matrix = car[numeric_cols].corr()

            fig1, ax1 = plt.subplots(figsize=(10,6))
            sns.heatmap(
                corr_matrix,
                cmap="coolwarm",
                center=0,
                ax=ax1
            )

            ax1.set_title("Feature Correlation Matrix")
            st.pyplot(fig1, use_container_width=True)

            st.markdown("---")

            # -----------------------------
            # 2️⃣ Top Correlated with Target
            # -----------------------------
            st.markdown("### 🎯 Top Correlated Features with is_claim")

            target_corr = corr_matrix["is_claim"].abs().sort_values(ascending=False)

            top_corr = target_corr[1:11]  # exclude self-correlation

            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.dataframe(top_corr)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                top_corr.plot(kind="bar", ax=ax2)
                ax2.set_title("Top Correlated with is_claim")
                st.pyplot(fig2, use_container_width=True)

            st.markdown("---")

        

            st.subheader("🧠 Claim Concentration Explorer")

            # Features to analyze
            claim_features = {
                "Area Cluster": "area_cluster",
                "Make": "make",
                "Model": "model",
                "Engine Type": "engine_type"
            }

            selected_feature = st.selectbox(
                "Select Feature to Analyze Claim Concentration",
                list(claim_features.keys())
            )

            feature_col = claim_features[selected_feature]

            # ---------------------------------------
            # Compute claim counts
            # ---------------------------------------
            claim_counts = (
                car[car["is_claim"] == 1]
                .groupby(feature_col)["is_claim"]
                .count()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"is_claim": "Number of Claims"})
            )

            st.markdown("---")

            col1, col2 = st.columns([1, 1])

            # -----------------------
            # Table
            # -----------------------
            with col1:
                st.subheader("📋 Claim Counts")
                st.dataframe(claim_counts, use_container_width=True)

                if not claim_counts.empty:
                    top_category = claim_counts.iloc[0][feature_col]
                    top_claims = claim_counts.iloc[0]["Number of Claims"]

                    st.success(f"🔥 Highest Claims: {top_category} ({top_claims} claims)")

            # -----------------------
            # Visualization
            # -----------------------
            with col2:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(6,4))

                sns.barplot(
                    x="Number of Claims",
                    y=feature_col,
                    data=claim_counts,
                    ax=ax
                )

                ax.set_title(f"Number of Claims by {selected_feature}")

                st.pyplot(fig, use_container_width=True)


            st.markdown("### 👤 Age Group vs Claim Risk")

            # Reverse normalization before analysis
            car['policy_tenure'] = car['policy_tenure'] * 10
            car['age_of_car'] = car['age_of_car'] * 100
            car['age_of_policyholder'] = car['age_of_policyholder'] * 10
            car['age_of_policyholder'] = car['age_of_policyholder'] * 6.25

            # Create age groups
            car["age_group"] = pd.cut(
                car["age_of_policyholder"],
                bins=[18, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
            )

            # Compute claim rate per age group
            age_claim = (
                car.groupby("age_group")["is_claim"]
                .agg(["count", "sum", "mean"])
                .reset_index()
            )

            age_claim.columns = [
                "Age Group",
                "Total Policies",
                "Total Claims",
                "Claim Rate"
            ]

            age_claim["Claim Rate"] = (age_claim["Claim Rate"] * 100).round(2)

            col1, col2 = st.columns([1, 1.5])

            # -------------------------
            # Table
            # -------------------------
            with col1:
                st.dataframe(age_claim, use_container_width=True)

                highest_group = age_claim.sort_values(
                    "Claim Rate", ascending=False
                ).iloc[0]

                st.success(
                    f"🔥 Highest Risk Age Group: {highest_group['Age Group']} "
                    f"({highest_group['Claim Rate']}%)"
                )

            # -------------------------
            # Visualization
            # -------------------------
            with col2:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6,4))

                ax.bar(age_claim["Age Group"], age_claim["Claim Rate"])

                ax.set_title("Claim Rate by Age Group")
                ax.set_ylabel("Claim Rate (%)")
                ax.set_xlabel("Age Group")

                st.pyplot(fig, use_container_width=True)
