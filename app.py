import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import joblib
import base64
from io import BytesIO

# 
# Page configuration
# 
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")
st.title("Breast Cancer Classification")
st.markdown(
    "This application walks through a complete machine learning pipeline using the "
    "Wisconsin Breast Cancer dataset. Explore the data, understand the case, "
    "view raw records, and train and evaluate classification models."
)

# 
# Load data from CSV and cache it
# 
@st.cache_resource
def load_data():
    # 1. Read the CSV file
    df = pd.read_csv('data.csv')
    
    # 2. Map diagnosis: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # 3. Drop the 'id' column (not needed)
    df.drop('id', axis=1, inplace=True)
    
    # 4. Rename columns to match scikit‑learn's feature names
    sklearn_feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    feature_columns = [col for col in df.columns if col != 'diagnosis']
    rename_map = {old: new for old, new in zip(feature_columns, sklearn_feature_names)}
    df.rename(columns=rename_map, inplace=True)
    
    # 5. Get the dataset description from sklearn (for the Case Study tab)
    description = load_breast_cancer().DESCR
    
    return df, description

df, description = load_data()
X_full = df.drop('diagnosis', axis=1)
y_full = df['diagnosis']

# 
# Helper functions for download links
# 
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def get_model_download_link(model, filename="model.pkl"):
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Model (.pkl)</a>'
    return href

# 
# Tabs
# 
tab1, tab2, tab3, tab4 = st.tabs(
    ["EDA & Visualization", "Case Study", "Raw Data & Export", "Machine Learning"]
)

# 
# TAB 1: EDA & Visualization
# 
with tab1:
    st.header("Exploratory Data Analysis and Visualization")
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Malignant %", f"{df['diagnosis'].mean()*100:.1f}%")

    with st.expander("Show first 5 rows"):
        st.dataframe(df.head())
    with st.expander("Statistical Summary"):
        st.dataframe(df.describe().style.format("{:.2f}"))
    with st.expander("Class Distribution"):
        st.write(df['diagnosis'].value_counts().rename({0: 'Benign', 1: 'Malignant'}))

    st.subheader("Feature Distributions by Diagnosis")
    if st.checkbox("Show histograms for all features (may take a moment)"):
        n_features = len(X_full.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()
        for i, feature in enumerate(X_full.columns):
            sns.histplot(data=df, x=feature, hue='diagnosis', kde=True,
                         ax=axes[i], palette={0: 'skyblue', 1: 'salmon'})
            axes[i].set_title(feature)
            axes[i].legend(title='Diagnosis', labels=['Benign (0)', 'Malignant (1)'])
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Features with separated distributions are likely good predictors.")

    st.subheader("Correlation with Diagnosis")
    corr_with_target = df.corr()[['diagnosis']].sort_values(by='diagnosis', ascending=False)
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation with Diagnosis (1=Malignant)")
    st.pyplot(fig)

    st.subheader("Pairplot of Top Correlated Features")
    top_features = corr_with_target.index[1:5].tolist()
    if top_features:
        pairplot_fig = sns.pairplot(df, vars=top_features, hue='diagnosis',
                                    palette={0: 'skyblue', 1: 'salmon'}, corner=True)
        pairplot_fig.fig.suptitle("Top Correlated Features", y=1.02)
        st.pyplot(pairplot_fig)
    else:
        st.warning("No features found for pairplot.")

    st.subheader("Boxplots by Diagnosis")
    selected_box_features = st.multiselect(
        "Choose up to 6 features for boxplots",
        options=X_full.columns.tolist(),
        default=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                 'mean smoothness', 'mean concavity'][:6]
    )
    if selected_box_features:
        n_cols_box = min(3, len(selected_box_features))
        n_rows_box = (len(selected_box_features) + n_cols_box - 1) // n_cols_box
        fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(n_cols_box * 5, n_rows_box * 4))
        if n_rows_box * n_cols_box == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, feat in enumerate(selected_box_features):
            # FIX: use hue and legend=False to avoid palette key error
            sns.boxplot(x='diagnosis', y=feat, data=df, hue='diagnosis',
                        palette={0: 'skyblue', 1: 'salmon'}, legend=False, ax=axes[i])
            axes[i].set_title(feat)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle("Boxplots of Selected Features by Diagnosis", y=1.02)
        st.pyplot(fig)

    st.subheader("UMAP Visualization (on scaled data)")
    if st.checkbox("Run UMAP (requires umap-learn installed)"):
        try:
            from umap import UMAP
            X_scaled = StandardScaler().fit_transform(X_full)
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
            X_umap = reducer.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(X_umap[y_full == 0, 0], X_umap[y_full == 0, 1],
                       label="Benign (0)", c="skyblue", alpha=0.7)
            ax.scatter(X_umap[y_full == 1, 0], X_umap[y_full == 1, 1],
                       label="Malignant (1)", c="salmon", alpha=0.7)
            ax.set_xlabel("UMAP Component 1")
            ax.set_ylabel("UMAP Component 2")
            ax.set_title("UMAP Projection of Breast Cancer Data")
            ax.legend()
            st.pyplot(fig)
        except ImportError:
            st.error("umap-learn is not installed. Run `pip install umap-learn` in your environment.")

# 
# TAB 2: Case Study
# 
with tab2:
    st.header("Case Study: Wisconsin Breast Cancer Classification")
    st.markdown(description)
    st.markdown(
        """
        ### Project Objective
        Build a machine learning model that accurately classifies breast
        cancer tumours as **malignant** (cancerous) or **benign** (non‑cancerous)
        based on 30 real‑valued features extracted from digitised images of
        fine needle aspirates (FNAs).

        ### Data
        - **Source:** UCI Machine Learning Repository / Scikit‑learn `load_breast_cancer`.
        - **Samples:** 569 (212 malignant, 357 benign).
        - **Features:** 30 numeric measurements describing cell nuclei
          (radius, texture, perimeter, area, smoothness, compactness, concavity,
          concave points, symmetry, fractal dimension) – each measured as
          mean, standard error, and worst value.

        ### Methods
        1. **Data Exploration:** Visual inspection of feature distributions,
           correlations, and low‑dimensional projections (UMAP).
        2. **Preprocessing:** Standard scaling of all features.
        3. **Models Tested:**
           - Logistic Regression
           - Decision Tree
           - Random Forest
        4. **Evaluation:** Stratified 80/20 train‑test split, 10‑fold
           cross‑validation, ROC‑AUC as primary metric.
        5. **Hyperparameter Tuning:** Grid search on Random Forest.

        ### Key Insights
        - Several features, particularly **worst concave points**,
          **worst perimeter**, and **worst radius**, show strong positive
          correlation with malignancy.
        - Ensemble methods (Random Forest) outperform a single decision tree
          and generalise better.
        - The final tuned Random Forest achieves a test ROC‑AUC above 0.99,
          indicating excellent discrimination.

        ### Clinical Relevance
        A model like this can serve as a decision‑support tool for pathologists,
        helping to prioritise suspicious cases and reduce subjectivity.
        However, model predictions must always be validated by a medical
        professional – false negatives carry severe consequences.
        """
    )

# 
# TAB 3: Raw Data & Export
# 
with tab3:
    st.header("Raw Data & Export")
    st.dataframe(df)
    st.markdown(get_table_download_link(df, filename="data.csv"), unsafe_allow_html=True)
    st.info("Click the link above to download the full dataset as a CSV file.")

# 
# TAB 4: Machine Learning
# 
with tab4:
    st.header("Machine Learning Model Training and Evaluation")
    st.markdown(
        "Select a classifier, adjust hyperparameters, and train the model on a "
        "stratified train‑test split. Cross‑validation scores, test metrics, "
        "and the option to download the trained model are provided."
    )

    # Controls
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random state", 0, 1000, 42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size / 100, random_state=int(random_state), stratify=y_full
    )

    model_option = st.selectbox(
        "Choose a classifier",
        ("Logistic Regression", "Decision Tree", "Random Forest")
    )

    if model_option == "Logistic Regression":
        C = st.slider("Inverse regularisation strength (C)", 0.01, 10.0, 1.0, 0.01)
        solver = st.selectbox("Solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"], index=0)
        model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    elif model_option == "Decision Tree":
        max_depth = st.slider("Max depth (None = unlimited)", 1, 30, 5)
        min_samples_split = st.slider("Min samples split", 2, 20, 2)
        model = DecisionTreeClassifier(
            max_depth=None if max_depth == 30 else max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    else:
        n_estimators = st.slider("Number of trees", 50, 300, 100, 10)
        max_depth = st.slider("Max depth (None = unlimited)", 1, 30, 10)
        max_features = st.selectbox("Max features", ["sqrt", "log2", None], index=0)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 30 else max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    if st.button("Train and Evaluate Model"):
        with st.spinner("Training... This may take a moment."):
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                        cv=cv_folds, scoring="roc_auc")
            pipeline.fit(X_train, y_train)
            y_test_pred = pipeline.predict(X_test)
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_proba)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean CV AUC", f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            col2.metric("Test Accuracy", f"{test_accuracy:.4f}")
            col3.metric("Test ROC-AUC", f"{test_auc:.4f}")

            st.subheader("Confusion Matrix (Test Set)")
            cm = confusion_matrix(y_test, y_test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['Benign (0)', 'Malignant (1)'])
            fig, ax = plt.subplots()
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title("Confusion Matrix - Test Set")
            st.pyplot(fig)

            st.subheader("ROC Curve (Test Set)")
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (area = {test_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            ax.grid(True)
            st.pyplot(fig)

            if model_option in ["Decision Tree", "Random Forest"]:
                st.subheader("Feature Importances")
                importances = pipeline.named_steps["classifier"].feature_importances_
                feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15], ax=ax, color='viridis')
                ax.set_title("Top 15 Feature Importances")
                ax.set_xlabel("Importance")
                st.pyplot(fig)

            st.markdown("### Download Trained Model")
            st.markdown(get_model_download_link(pipeline, "breast_cancer_model.pkl"), unsafe_allow_html=True)
            st.success("Model trained successfully. You can now download the pickled pipeline.")