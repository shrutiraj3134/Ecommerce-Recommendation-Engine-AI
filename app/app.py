import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# ==============================================================================
# 1. PAGE SETUP & THEME STYLING
# ==============================================================================
st.set_page_config(page_title="NexCart Recommender Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; font-family: 'Segoe UI', sans-serif; }
    .gradient-header {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 45px; font-weight: 800; text-align: center; margin-bottom: 5px;
    }
    .sub-header { color: #8b949e; text-align: center; font-size: 18px; margin-bottom: 30px; }
    
    /* Metrics Banner */
    .metric-box {
        background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-val { color: #00f2fe; font-size: 28px; font-weight: 700; }
    .metric-lbl { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }

    /* Product Cards */
    .product-card {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 22px; margin: 15px 0px; transition: all 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-4px); border-color: #00f2fe;
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.15);
    }
    .card-tag-cluster { color: #00f2fe; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 6px; }
    .card-tag-trend { color: #ff9f43; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 6px; }
    .card-title { color: #f0f6fc; font-size: 15px; font-weight: 600; }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #0d1117 !important;
        font-weight: 700; border: none; padding: 12px 30px; border-radius: 8px; width: 100%; font-size: 16px;
    }
    div.stButton > button:first-child:hover { transform: scale(1.01); box-shadow: 0 5px 20px rgba(0, 242, 254, 0.4); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="gradient-header">NexCart Engine AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced ML Recommendation System & Analytics Dashboard • Built by Shruti Raj</div>', unsafe_allow_html=True)

# ==============================================================================
# 2. OPTIMIZED DATA PIPELINE
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('../data/7817_1.csv')
    clean_df = df[['reviews.username', 'asins', 'reviews.rating']].copy()
    clean_df.columns = ['username', 'product_id', 'rating']
    clean_df.dropna(inplace=True)
    
    product_mapping = df.dropna(subset=['asins', 'name']).set_index('asins')['name'].to_dict()
    
    # Pre-calculating Global Trends
    global_trends = clean_df.groupby('product_id').agg({'rating': ['count', 'mean']})
    global_trends.columns = ['count', 'mean']
    top_products = global_trends[global_trends['count'] >= 5].sort_values(by='mean', ascending=False).index.tolist()
    
    return clean_df, product_mapping, top_products

try:
    clean_df, product_mapping, top_products = load_and_clean_data()
except FileNotFoundError:
    st.error("❌ Data File missing! Please ensure '7817_1.csv' is saved inside 'data/' folder.")
    st.stop()

# Building User-Item Matrix & KNN
user_product_matrix = clean_df.pivot_table(index='username', columns='product_id', values='rating')
user_product_matrix.fillna(0, inplace=True)
user_pivot_sparse = csr_matrix(user_product_matrix.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_pivot_sparse)

# ==============================================================================
# 3. INTERACTIVE LIVE ANALYTICS BANNER (NEW FEATURE 1)
# ==============================================================================
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{user_product_matrix.shape[0]}</div><div class="metric-lbl">Total Profiles</div></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{user_product_matrix.shape[1]}</div><div class="metric-lbl">Unique Products</div></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{clean_df["rating"].mean().round(2)} ⭐</div><div class="metric-lbl">Average Rating</div></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{len(clean_df)}</div><div class="metric-lbl">Clean Reviews</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout: Left side for Control/ML, Right side for Visual EDA Chart
layout_col1, layout_col2 = st.columns([1, 1])

# ==============================================================================
# 4. CONTROL PANEL & INFERENCE CORNER (LEFT SIDE)
# ==============================================================================
with layout_col1:
    st.markdown("### 🎛️ AI Engine Control Room")
    all_users = list(user_product_matrix.index)
    
    # Smart Search & Select Combobox (NEW FEATURE 3)
    selected_user = st.selectbox("Search or Select a Target Profile Index:", all_users, index=all_users.index('1soni') if '1soni' in all_users else 0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Get Recommendations"):
        user_idx = user_product_matrix.index.get_loc(selected_user)
        distances, indices = model_knn.kneighbors(user_product_matrix.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=6)
        
        target_user_ratings = user_product_matrix.loc[selected_user]
        
        cluster_recs = []
        for idx in indices.flatten()[1:]:
            sim_user_name = user_product_matrix.index[idx]
            sim_user_ratings = user_product_matrix.loc[sim_user_name]
            for prod_id in user_product_matrix.columns:
                if target_user_ratings[prod_id] == 0 and sim_user_ratings[prod_id] >= 4:
                    if prod_id not in cluster_recs:
                        cluster_recs.append(prod_id)
                        
        st.success(f"🎯 Cluster match successfully activated for user profile: `{selected_user}`")
        
        rendered_count = 0
        if len(cluster_recs) > 0:
            for prod_id in cluster_recs[:4]:
                asli_naam = product_mapping.get(prod_id, prod_id)
                st.markdown(f"""<div class="product-card"><div class="card-tag-cluster">AI Personal Cluster Match #{rendered_count+1}</div><div class="card-title">🛍️ {asli_naam}</div></div>""", unsafe_allow_html=True)
                rendered_count += 1

        if rendered_count < 4:
            for prod_id in top_products:
                if target_user_ratings.get(prod_id, 0) == 0 and prod_id not in cluster_recs:
                    asli_naam = product_mapping.get(prod_id, prod_id)
                    st.markdown(f"""<div class="product-card"><div class="card-tag-trend">Trending High-Value Fallback Match</div><div class="card-title">🛍️ {asli_naam}</div></div>""", unsafe_allow_html=True)
                    rendered_count += 1
                    if rendered_count >= 4:
                        break

# ==============================================================================
# 5. DATA VISUALIZATION AREA (RIGHT SIDE - NEW FEATURE 2)
# ==============================================================================
with layout_col2:
    st.markdown("### 📊 Platform Distribution Analytics")
    
    # Pandas aggregate counting for plotting [cite: 113, 196]
    rating_counts = clean_df['rating'].value_counts().sort_index()
    chart_data = pd.DataFrame({'Total Reviews': rating_counts.values}, index=['1 ⭐', '2 ⭐', '3 ⭐', '4 ⭐', '5 ⭐'])
    
    # Interactive Native Streamlit Bar Chart [cite: 148, 351]
    st.bar_chart(chart_data, color="#00f2fe", use_container_width=True)
    st.markdown("<p style='color:#8b949e; font-size:13px; text-align:center;'>Figure 1.0: Real-time user ratings frequency spread across the marketplace.</p>", unsafe_allow_html=True)