import streamlit as st
from utils import load_data, create_scatter_plot

st.title("🔍 Data Exploration")

# Load data
df = load_data()

st.write("""
### Exploring Taylor Swift's Song Metrics
Let's start by looking at the relationships between different song features.
This will help us understand which variables might be good predictors for our regression model.
""")

# Display raw data
with st.expander("View Raw Data"):
    st.dataframe(df)

# Feature selection for visualization
st.subheader("Visualize Relationships")
col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "Select X-axis feature",
        options=['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    )

with col2:
    y_feature = st.selectbox(
        "Select Y-axis feature",
        options=['popularity', 'danceability', 'energy', 'loudness', 'valence']
    )

# Create and display scatter plot
if x_feature != y_feature:
    fig = create_scatter_plot(
        df, x_feature, y_feature,
        f"Relationship between {x_feature} and {y_feature}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Display summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Add footer
st.markdown("---")  # Add a horizontal line
st.markdown("[Made by Western AI Education](https://www.instagram.com/westernu.ai/)")