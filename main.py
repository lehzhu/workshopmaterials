#Made for Western AI workshop
#Author: Leon Zhu
#Date: November 2024
import streamlit as st
from utils import load_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(page_title="Taylor Swift Linear Regression Tutorial",
                   page_icon="🎵",
                   layout="wide",
                   initial_sidebar_state="expanded")


def main():
   try:
      logger.info("Starting application...")
      st.title("📊 Learning Linear Regression with Taylor Swift")
      st.write("""
        Welcome to an interactive tutorial on Linear Regression using Taylor Swift's song data! 
        This tutorial will help you understand how different characteristics of songs relate to their popularity, 
        using real data from Taylor Swift's discography.
        """)

      st.header("📐 Understanding Linear Regression")
      st.write("""
        Linear regression is a powerful statistical method that helps us understand and predict relationships 
        between variables in Taylor Swift's music. Think of it as drawing a "line of best fit" through data points 
        that helps us understand how one aspect of a song relates to another.

        #### Key Concepts:

        1. **Variables in Our Analysis**: 
           - **Dependent Variable (y)**: Usually song popularity (what we want to predict)
           - **Independent Variables (x)**: Song characteristics we use to make predictions
           - **Example**: How does a song's danceability affect its popularity?

        2. **The Linear Equation**: y = mx + b
           - **m (slope)**: The relationship strength
             - Positive slope: As x increases, y increases
             - Negative slope: As x increases, y decreases
             - Example: If m = 2 for danceability vs. popularity, a 0.1 increase in danceability suggests a 0.2 increase in popularity
           - **b (y-intercept)**: The baseline value when x = 0

        3. **Understanding Model Quality**:
           - **R² Score (0 to 1)**:
             - Closer to 1: Strong relationship (e.g., 0.8 means the model explains 80% of popularity variations)
             - Closer to 0: Weak relationship
           - **Residuals**:
             - The differences between predicted and actual values
             - Smaller residuals indicate better predictions
             - Help us understand where our model needs improvement
        """)

      st.header("🎵 Understanding Song Metrics")
      st.write("""
        Let's explore the key metrics from Taylor Swift's songs that we'll use in our analysis:

        1. **Popularity (0-100)**
           - Measures overall song popularity on Spotify
           - Higher values indicate more popular songs
           - Recent Example: "Anti-Hero" (90) is one of her most popular tracks

        2. **Acoustic Features**:
           - **Acousticness (0-1)**: Confidence measure of whether the track is acoustic
           - **Instrumentalness (0-1)**: Predicts whether a track contains no vocals
           - **Liveness (0-1)**: Detects presence of an audience in the recording
           - Example: "folklore" album tracks typically have higher acousticness values

        3. **Rhythm and Movement**:
           - **Danceability (0-1)**: How suitable a track is for dancing
           - **Tempo (BPM)**: Speed or pace of a track
           - **Energy (0-1)**: Measure of intensity and activity
           - Example: "Shake It Off" has high danceability (0.65) and energy (0.80)

        4. **Sound and Speech**:
           - **Loudness (dB)**: Overall loudness, typically -60 to 0 dB
           - **Speechiness (0-1)**: Presence of spoken words
           - Example: Tracks from "reputation" album often have higher loudness values

        5. **Emotional Content**:
           - **Valence (0-1)**: Musical positiveness
           - Higher values indicate more positive/happy sounds
           - Example: "ME!" has high valence while "exile" has lower valence

        These metrics help us understand:
        - How song characteristics evolve across albums
        - What makes certain songs more popular
        - The relationship between different musical elements
        """)

      with st.sidebar:
         st.markdown("### About the Dataset")
         st.write("""
            This dataset contains various metrics for Taylor Swift's songs, including:
            - Popularity
            - Danceability
            - Energy
            - Loudness
            - Valence
            - Tempo
            """)

      # Test data loading
      df = load_data()
      logger.info("Data loaded successfully")

   except Exception as e:
      logger.error(f"Error in main: {str(e)}")
      st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
   main()

# Add footer
st.markdown("---")  # Add a horizontal line
st.markdown(
    "[Made by Western AI Education](https://www.instagram.com/westernu.ai/)")
