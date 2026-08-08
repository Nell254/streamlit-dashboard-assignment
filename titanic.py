# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load the Titanic dataset
@st.cache_data
def load_titanic_data():
    # Using seaborn's built-in titanic dataset
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        df = pd.read_csv(url)
        return df
    except:
        # Fallback to a local sample if online access fails
        # This is just for demonstration - you'd typically use the direct URL
        st.write("Using fallback data (sample)")
        return pd.DataFrame({
            'survived': [0, 1, 1, 0, 1],
            'pclass': [3, 1, 3, 1, 2],
            'sex': ['male', 'female', 'female', 'male', 'female'],
            'age': [22, 38, 26, 35, 19],
            'sibsp': [1, 1, 0, 1, 0],
            'parch': [0, 0, 0, 0, 0],
            'fare': [7.25, 71.28, 7.92, 53.1, 12.75],
            'embarked': ['S', 'C', 'S', 'S', 'Q']
        })

# Load data
df = load_titanic_data()

# Configure app layout
st.set_page_config(page_title="Titanic Dashboard", layout="wide")
st.title("📊 Titanic Dataset Interactive Dashboard")
st.markdown("""
This interactive dashboard analyzes the famous Titanic dataset. 
Explore passenger information, survival statistics, and demographic patterns.
""")

# Sidebar filters
st.sidebar.header("Filter Data")
pclass_filter = st.sidebar.multiselect(
    "Passenger Class",
    options=df["pclass"].unique(),
    default=df["pclass"].unique()
)

sex_filter = st.sidebar.multiselect(
    "Sex",
    options=df["sex"].unique(),
    default=df["sex"].unique()
)

embarked_filter = st.sidebar.multiselect(
    "Embarkation Port",
    options=df["embarked"].unique(),
    default=df["embarked"].unique()
)

# Apply filters
filtered_df = df[
    (df["pclass"].isin(pclass_filter)) &
    (df["sex"].isin(sex_filter)) &
    (df["embarked"].isin(embarked_filter))
]

# Display dataset info
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Passengers", len(filtered_df))
col2.metric("Survival Rate", f"{(filtered_df['survived'].mean() * 100):.1f}%")
col3.metric("Average Age", f"{filtered_df['age'].mean():.1f} years")

# Main visualization area
st.subheader("Passenger Survival Analysis")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Survival Distribution", "Age Analysis", "Fare Analysis", "Demographics"])

with tab1:
    # Survival by class
    survival_by_class = filtered_df.groupby(['pclass', 'survived']).size().unstack(fill_value=0)
    survival_by_class_pct = survival_by_class.div(survival_by_class.sum(axis=1), axis=0) * 100
    
    # fig1 = px.bar(
    #     survival_by_class_pct,
    #     x=survival_by_class_pct.index,
    #     y=['0', '1'],
    #     labels={'value': 'Percentage', 'pclass': 'Passenger Class'},
    #     title='Survival Rate by Passenger Class',
    #     color_discrete_sequence=['#FF6B6B', '#4ECDC9']
    # )
    # fig1.update_layout(barmode='group')
    # st.plotly_chart(fig1, use_container_width=True)

    survival_by_class_pct_reset = survival_by_class_pct.reset_index(name='count')
    survival_by_class_pct_reset.columns = ['pclass', 'survived', 'count']

    fig1 = px.bar(
        survival_by_class_pct_reset,
        x='pclass',
        y='count',
        color='survived',
        color_discrete_sequence=['#FF6B6B', '#4ECDC9'],
        title='Survival Count by Class'
    )


with tab2:
    # Age distribution
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Age Distribution', 'Age vs Survival'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Age histogram
    fig2.add_trace(
        go.Histogram(x=filtered_df['age'], name="Passengers", 
                    marker_color='#4ECDC9', opacity=0.7),
        row=1, col=1
    )
    
    # Age vs survival scatter
    survived_data = filtered_df[filtered_df['survived'] == 1]
    not_survived_data = filtered_df[filtered_df['survived'] == 0]
    
    fig2.add_trace(
        go.Scatter(x=survived_data['age'], y=survived_data['fare'], 
                  mode='markers', name='Survived', 
                  marker=dict(color='#4ECDC9', size=8, opacity=0.7)),
        row=1, col=2
    )
    
    fig2.add_trace(
        go.Scatter(x=not_survived_data['age'], y=not_survived_data['fare'], 
                  mode='markers', name='Not Survived', 
                  marker=dict(color='#FF6B6B', size=8, opacity=0.7)),
        row=1, col=2
    )
    
    fig2.update_xaxes(title_text="Age", row=1, col=1)
    fig2.update_yaxes(title_text="Count", row=1, col=1)
    fig2.update_xaxes(title_text="Age", row=1, col=2)
    fig2.update_yaxes(title_text="Fare", row=1, col=2)
    fig2.update_layout(height=400, title_text="Age Analysis")
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Fare distribution
    fig3 = px.box(
        filtered_df,
        x="pclass",
        y="fare",
        color="survived",
        title='Fare Distribution by Class and Survival',
        color_discrete_sequence=['#FF6B6B', '#4ECDC9']
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    # Gender and class distribution
    gender_class = filtered_df.groupby(['pclass', 'sex']).size().unstack(fill_value=0)
    
    fig4 = px.imshow(
        gender_class,
        labels=dict(x="Sex", y="Passenger Class", color="Count"),
        title='Passenger Distribution by Class and Gender',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig4, use_container_width=True)

# Detailed statistics section
st.subheader("Detailed Statistics")
st.write(f"Filtered dataset contains {len(filtered_df)} passengers out of {len(df)} total")

# Survival statistics table
survival_stats = filtered_df.groupby('pclass')['survived'].agg(['count', 'sum', 'mean']).round(3)
survival_stats.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
st.dataframe(survival_stats)

# Correlation heatmap
st.subheader("Feature Correlations")
numeric_df = filtered_df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title='Feature Correlation Heatmap'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📊 Built with Streamlit and Plotly | Titanic Dataset Analysis")
