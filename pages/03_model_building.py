import streamlit as st
from utils import load_data
import numpy as np
import plotly.graph_objects as go

st.title("🔨 Build Your Regression Model")

# Load data
df = load_data()

st.write("""
### Build a Linear Regression Model
Now it's your turn! Select features to build a regression model and predict song popularity.
Adjust the slope and intercept to find the best fit line manually.
""")

# Feature selection
feature = st.selectbox("Select the feature to predict popularity",
                       options=[
                           'danceability', 'energy', 'loudness', 'valence',
                           'tempo', 'acousticness', 'instrumentalness',
                           'liveness', 'speechiness'
                       ])

# Calculate optimal parameters using np.polyfit
X = df[feature].values.astype(float)
y = df['popularity'].values.astype(float)
optimal_m, optimal_b = np.polyfit(X, y, 1)

# Manual regression controls
st.subheader("Adjust Regression Parameters")
col1, col2 = st.columns(2)

with col1:
    # Round values to avoid floating point precision issues
    slope_min = round(float(optimal_m - 0.5), 2)
    slope_max = round(float(optimal_m + 0.5), 2)
    slope_value = round(float(optimal_m), 2)
    m = st.slider("Slope (m)",
                  min_value=slope_min,
                  max_value=slope_max,
                  value=slope_value,
                  step=0.01,
                  key='slope_slider')

with col2:
    # Round values to avoid floating point precision issues
    intercept_min = round(float(optimal_b - 0.5), 2)
    intercept_max = round(float(optimal_b + 0.5), 2)
    intercept_value = round(float(optimal_b), 2)
    b = st.slider("Intercept (b)",
                  min_value=intercept_min,
                  max_value=intercept_max,
                  value=intercept_value,
                  step=0.01,
                  key='intercept_slider')

# Get data points
X = df[feature].values
y = df['popularity'].values

# Calculate predictions and metrics
X = X.astype(float)
y = y.astype(float)
y_pred = m * X + b
residuals = y - y_pred

# Calculate metrics
mse = np.mean(np.square(residuals))
ss_res = np.sum(np.square(residuals))
y_mean = np.mean(y)
ss_tot = np.sum(np.square(y - y_mean))
r2 = 1 - (ss_res / ss_tot)

# Display metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.3f}")
    st.write("(Closer to 1.0 indicates better fit)")

with col2:
    st.metric("Mean Squared Error", f"{mse:.3f}")
    st.write("(Lower is better)")

# Create figure
fig = go.Figure()

# Add scatter plot of actual data points
fig.add_trace(
    go.Scatter(x=X,
               y=y,
               mode='markers',
               name='Actual Data',
               hovertemplate=(f"Song: %{{customdata[1]}}<br>"
                              f"{feature}: %{{x}}<br>"
                              "popularity: %{y}<br>"
                              "Residual: %{customdata[0]:.2f}<br>"),
               customdata=np.column_stack((residuals, df['song_name']))))

# Add regression line
x_range = np.linspace(float(X.min()), float(X.max()), 100)
y_line = m * x_range + b

fig.add_trace(
    go.Scatter(x=x_range,
               y=y_line,
               mode='lines',
               name='Your Line',
               line=dict(color='red')))

# Add residual lines
for i in range(len(X)):
    fig.add_trace(
        go.Scatter(x=[X[i], X[i]],
                   y=[y[i], y_pred[i]],
                   mode='lines',
                   line=dict(color='gray', dash='dot', width=1),
                   showlegend=True if i == 0 else False,
                   name='Residuals' if i == 0 else None,
                   hoverinfo='skip'))

# Update layout
fig.update_layout(height=600,
                  title_text=f"Manual Regression: Popularity vs {feature}",
                  xaxis_title=feature,
                  yaxis_title='popularity',
                  showlegend=True,
                  template='simple_white',
                  hovermode='closest')

st.plotly_chart(fig, use_container_width=True)

# Model interpretation
st.subheader("Model Interpretation")
st.write(f"""
- **Slope**: {m:.3f}
- **Intercept**: {b:.3f}
- **Equation**: popularity = {m:.3f} × {feature} + {b:.3f}
""")

# Prediction interface
st.subheader("Make Predictions")

# Get min, max values for the selected feature
min_val = float(df[feature].min())
max_val = float(df[feature].max())
step_size = (max_val - min_val) / 100.0
step_size = round(step_size, 3)
min_val = round(min_val - (min_val % step_size), 3)
max_val = round(max_val + (step_size - max_val % step_size), 3)

user_input = st.slider(f"Select a {feature} value",
                       min_value=min_val,
                       max_value=max_val,
                       value=round(float(df[feature].mean()), 3),
                       step=step_size,
                       key=f'prediction_slider_{feature}')

prediction = m * user_input + b
st.write(f"Predicted popularity: {prediction:.2f}")

# Add footer
st.markdown("---")  # Add a horizontal line
st.markdown(
    "[Made by Western AI Education](https://www.instagram.com/westernu.ai/)")
