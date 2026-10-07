import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set modern page layout and custom themes
st.set_page_config(
    page_title="Facebook Live Clustering Portal", page_icon="📊", layout="wide"
)

# Custom CSS injected directly to elevate the UI design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .title-banner {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Banner UI Component
st.markdown(
    """
    <div class="title-banner">
        <h1>📊 Unsupervised K-Means Engagement Portal</h1>
        <p style="font-size:1.1rem; opacity:0.9; max-width:800px; margin: 0 auto; margin-top: 10px;">
            Grouping posts entirely by behavior metrics. As per the revised notebook methodology, 
            <b>status_type</b> is strictly reserved for evaluation context and excluded from training.
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Control Panel
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Parameters")
    st.write("---")

    st.markdown("### 1. Data File Source")
    csv_file = st.file_uploader(
        "Upload your Facebook Live CSV dataset", type=["csv"]
    )

    st.write("---")
    st.markdown("### 2. Tuning Configuration")
    num_clusters = st.slider(
        "Select Clusters (K)", min_value=2, max_value=10, value=2, step=1
    )

# ---------------------------------------------------------
# Processing Pipeline & Evaluation Logic
# ---------------------------------------------------------
if csv_file is not None:
    # 1. Import Dataset
    df = pd.read_csv(csv_file)
    data = df.copy()

    # 2. Data Cleansing
    drop_cols = [
        "status_id",
        "status_published",
        "Column1",
        "Column2",
        "Column3",
        "Column4",
    ]
    existing_drops = [col for col in drop_cols if col in data.columns]
    if existing_drops:
        data.drop(columns=existing_drops, inplace=True)

    # 3. Feature Extraction
    features = [
        "num_reactions",
        "num_comments",
        "num_shares",
        "num_likes",
        "num_loves",
        "num_wows",
        "num_hahas",
        "num_sads",
        "num_angrys",
    ]
    missing_features = [f for f in features if f not in data.columns]

    if missing_features:
        st.error(
            f"❌ Missing required numerical columns in dataset: {missing_features}"
        )
        st.stop()

    X = data[features].copy()

    # Handle Missing Values if any exist (safety fallback mechanism)
    X = X.fillna(X.median())

    # 4. Feature Transformation & Scaling (Log1p to normalize right-skew)
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # 5. Build K-Means Engine
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled_df)

    # Calculate Evaluation Scores
    inertia_score = kmeans.inertia_
    sil_score = silhouette_score(X_scaled_df, cluster_labels)

    # Append results to temporary tracking set
    result_df = data.copy()
    result_df["Cluster"] = cluster_labels

    # ---------------------------------------------------------
    # UI Component Output Structure
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Cluster Evaluation Profiles",
            "📊 Mathematical Validation",
            "📋 Interactive Data Explorer",
        ]
    )

    with tab1:
        st.subheader("Algorithmic Fit Metrics")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="Target Groupings (K)", value=num_clusters)
        with m_col2:
            st.metric(label="Cluster Inertia Score", value=f"{inertia_score:.2f}")
        with m_col3:
            st.metric(label="Silhouette Evaluation Index", value=f"{sil_score:.4f}")

        # Methodology Context Callout
        st.info(
            "💡 Unsupervised clustering utilizes mathematical metrics (Inertia, Silhouette) "
            "for internal evaluation rather than conventional cross-entropy classification labels."
        )

        st.write("---")
        st.subheader("Group Characteristics Profiles (Engagement Averages)")

        profile_metric = st.radio(
            "Display Profile By:", ["Mean (Average Summary)", "Median (Skew Safe)"]
        )

        if profile_metric == "Mean (Average Summary)":
            profile_view = result_df.groupby("Cluster")[features].mean().round(2)
        else:
            profile_view = result_df.groupby("Cluster")[features].median().round(2)

        st.dataframe(profile_view, use_container_width=True)

    with tab2:
        st.subheader("Dimensional Layout Assessment (PCA Visualization)")
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled_df)

            pca_plot_df = pd.DataFrame(
                {
                    "Principal Component 1 (PC1)": X_pca[:, 0],
                    "Principal Component 2 (PC2)": X_pca[:, 1],
                    "Cluster Assigned": cluster_labels.astype(str),
                }
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=pca_plot_df,
                x="Principal Component 1 (PC1)",
                y="Principal Component 2 (PC2)",
                hue="Cluster Assigned",
                palette="tab10",
                alpha=0.6,
                ax=ax,
            )
            ax.set_title("2D Mapping of Clusters Using PCA")
            st.pyplot(fig)

        with g_col2:
            if "status_type" in result_df.columns:
                st.markdown("**Crosstab Composition Analysis (`status_type`)**")
                crosstab = pd.crosstab(
                    result_df["Cluster"],
                    result_df["status_type"],
                    normalize="index",
                ).round(3)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                crosstab.plot(kind="bar", stacked=True, ax=ax2, colormap="viridis")
                ax2.set_title("Post-Type Share Within Clusters")
                ax2.set_ylabel("Proportion")
                ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.warning("`status_type` column missing for cross-tabulation visual.")

    with tab3:
        st.subheader("Live Operational Data Frame")
        st.dataframe(result_df, use_container_width=True)

else:
    # Initial Splash screen state
    st.markdown(
        """
        <div style="background-color: #fef08a; border-left: 6px solid #eab308; padding: 25px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #854d0e; margin-top:0;">👋 Pipeline Standby State</h3>
            <p style="font-size: 1.05rem; color: #713f12;">Please open the sidebar on the left margin and load the <b>Live.csv</b> file resource to execute the unsupervised data pipelines instantly.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )
