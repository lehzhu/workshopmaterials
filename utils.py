import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

def load_data():
    """Load and preprocess the Taylor Swift songs dataset"""
    df = pd.read_csv('data/taylor_swift_spotify.csv')
    # Handle missing values for numeric columns
    numeric_columns = [
        'popularity', 'danceability', 'energy', 'loudness', 'valence', 'tempo', 'duration_ms',
        'acousticness', 'instrumentalness', 'liveness', 'speechiness'
    ]
    
    # Convert columns to numeric, handling any non-numeric values
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Rename columns for consistency
    df = df.rename(columns={'name': 'song_name'})
    
    return df

def create_scatter_plot(data, x_col, y_col, title):
    """Create an interactive scatter plot using Plotly"""
    fig = px.scatter(data, x=x_col, y=y_col, 
                    title=title,
                    hover_data=['song_name', 'album'])
    fig.update_layout(
        template='simple_white',
        title_x=0.5,
        title_font_size=20
    )
    return fig

def fit_regression(X, y):
    """Fit a linear regression model and return model and metrics"""
    # Reshape X if needed
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, r2, mse, X_train, X_test, y_train, y_test

def calculate_residuals(X, y, model):
    """Calculate residuals for the regression model"""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    y_pred = model.predict(X)
    residuals = y - y_pred
    return residuals, y_pred

def plot_regression_line(data, x_col, y_col, model):
    """Create a scatter plot with regression line and residual lines"""
    # Get data points
    X = data[x_col].values
    y = data[y_col].values
    
    # Calculate predicted values and residuals
    residuals, y_pred = calculate_residuals(X, y, model)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot of actual data points
    fig.add_trace(
        go.Scatter(
            x=X,
            y=y,
            mode='markers',
            name='Actual Data',
            hovertemplate=(
                f"{x_col}: %{{x}}<br>"
                f"{y_col}: %{{y}}<br>"
                "Residual: %{customdata:.2f}<br>"
            ),
            customdata=residuals
        )
    )
    
    # Add regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_range.reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        )
    )
    
    # Add residual lines
    for i in range(len(X)):
        fig.add_trace(
            go.Scatter(
                x=[X[i], X[i]],
                y=[y[i], y_pred[i]],
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                showlegend=True if i == 0 else False,
                name='Residuals' if i == 0 else None,
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Regression Analysis: {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True,
        template='simple_white',
        hovermode='closest'
    )
    
    return fig
