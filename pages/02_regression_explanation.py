import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.title("📚 Understanding Linear Regression")

st.write("""
### What is Linear Regression?
Linear regression is a statistical method that models the relationship between a dependent 
variable (target) and one or more independent variables (features) by fitting a linear equation 
to the observed data.
""")

# Interactive demonstration
st.subheader("Interactive Demo")

# Generate sample data with true values m=2, b=1
np.random.seed(42)
x = np.linspace(0, 10, 100)
true_m, true_b = 2.0, 1.0  # True parameter values
y = true_m * x + true_b + np.random.normal(
    0, 1.0, 100)  # Reduced noise for clearer demonstration
sample_data = pd.DataFrame({'x': x, 'y': y})

# Define step size and slider ranges around true values
step_size = 0.1

# Adjusted sliders with ranges ±0.5 around true values
m = st.slider("Slope (m)",
              min_value=true_m - 0.5,
              max_value=true_m + 0.5,
              value=true_m,
              step=step_size,
              key='slope_slider')

b = st.slider("Intercept (b)",
              min_value=true_b - 0.5,
              max_value=true_b + 0.5,
              value=true_b,
              step=step_size,
              key='intercept_slider')

# Calculate predicted values and metrics
y_pred = m * x + b
residuals = y - y_pred
mae = np.mean(np.abs(residuals))
mse = np.mean(np.square(residuals))

# Create two columns for metrics display
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Mean Absolute Error (MAE)",
        value=f"{mae:.3f}",
        help=
        "Average magnitude of prediction errors. Lower values indicate better fit."
    )

with col2:
    st.metric(
        label="Mean Squared Error (MSE)",
        value=f"{mse:.3f}",
        help=
        "Average squared magnitude of prediction errors. More sensitive to large errors. Lower values indicate better fit."
    )

# Create figure
fig = go.Figure()

# Add scatter plot of data points
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        hovertemplate='x: %{x}<br>y: %{y}<br>Residual: %{customdata:.2f}',
        customdata=residuals))

# Add regression line
fig.add_trace(
    go.Scatter(x=x,
               y=y_pred,
               mode='lines',
               name='Your Line',
               line=dict(color='red')))

# Add residual lines
for i in range(len(x)):
    fig.add_trace(
        go.Scatter(x=[x[i], x[i]],
                   y=[y[i], y_pred[i]],
                   mode='lines',
                   line=dict(color='gray', dash='dot', width=1),
                   showlegend=True if i == 0 else False,
                   name='Residuals' if i == 0 else None,
                   hoverinfo='skip'))

# Update layout
fig.update_layout(height=600,
                  title_text="Try to fit the line to minimize residuals",
                  xaxis_title="x",
                  yaxis_title="y",
                  showlegend=True,
                  template='simple_white',
                  hovermode='closest')

st.plotly_chart(fig)

st.write("""
### Understanding Residuals
Residuals are the differences between the observed values and the predicted values from our regression line.
- A residual value of 0 means the prediction exactly matches the actual value
- Positive residuals mean our line underpredicted the actual value
- Negative residuals mean our line overpredicted the actual value

### Key Concepts:

1. **Line Equation**: y = mx + b
   - m: slope (how steep the line is)
   - b: y-intercept (where the line crosses the y-axis)

2. **Goal**: Find the best-fitting line that minimizes the prediction errors (residuals)

3. **Error Measurement**: 
   - We use Mean Squared Error (MSE)
   - The smaller the MSE, the better the fit
   - MSE is calculated by squaring the residuals and taking their mean

4. **Assumptions**:
   - Linear relationship between variables
   - Independent observations
   - Normal distribution of residuals
   - Homoscedasticity (constant variance of residuals)
""")

# Add footer
st.markdown("---")  # Add a horizontal line
st.markdown(
    "[Made by Western AI Education](https://www.instagram.com/westernu.ai/)")
