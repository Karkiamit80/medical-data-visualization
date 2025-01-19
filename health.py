import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lifelines import KaplanMeierFitter

# Load the dataset
file_path = 'healthcare.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
def preprocess_data(df):
    # Convert date columns
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Admission_Year'] = df['Date of Admission'].dt.year
    df['Admission_Month'] = df['Date of Admission'].dt.strftime('%b')  # Format as abbreviated month name (Jan, Feb, ...)
    df['Admission_Quarter'] = df['Date of Admission'].dt.quarter
    df['Admission_Day'] = df['Date of Admission'].dt.day
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Discharge_Year'] = df['Discharge Date'].dt.year
    df['Discharge_Month'] = df['Discharge Date'].dt.strftime('%b')  # Format as abbreviated month name (Jan, Feb, ...)
    df['Discharge_Quarter'] = df['Discharge Date'].dt.quarter
    df['Discharge_Day'] = df['Discharge Date'].dt.day
    df['los'] = df['Discharge Date'] - df['Date of Admission']
    
    # Define age groups
    age_bins = [0, 18, 35, 50, 65, 100]
    age_labels = ['0-18', '19-35', '36-50', '51-65', '66+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Sort by month for better plotting
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Admission_Month'] = pd.Categorical(df['Admission_Month'], categories=month_order, ordered=True)
    
    return df

df = preprocess_data(df)

# Configure the Streamlit page
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
main_pages = [
    "Overview",
    "Visualization Pages",
    "EDA",
    "Seasonal Trends",
    "Time Series Analysis",
    "Medical Condition Analysis",
    "Kaplan-Meier Analysis",
    "Correlation Heatmap",
    "Length of Stay Analysis",
    "Dimensionality Reduction"
]
selected_main_page = st.sidebar.radio("Select Main Page:", main_pages)

# Sub-page navigation for Visualization Pages
if selected_main_page == "Visualization Pages":
    viz_sub_pages = [
        "Pie Chart", "Scatter Plot", "Bubble Chart", "Bar Chart", "Line Chart",
        "Histogram", "Box Plot", "Stacked Bar Chart", "Stacked Area Chart"
    ]
    selected_viz = st.sidebar.selectbox("Select Visualization:", viz_sub_pages)

# Main Content Rendering
def render_overview(df):
    st.title("Healthcare Dashboard: Overview")
    st.write("Dataset Information")
    st.write(df.describe())
    st.write("Data Preview")
    st.write(df.head())
    if df.iloc[0].isnull().any():
        df = df.iloc[1:]

    # Preprocess data
    st.write("### Dataset Preview")
    st.write(df.head())

    # Extract numerical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_columns) > 1:
        # Display numeric columns
        st.write("### Numerical Columns")
        st.write(numeric_columns)

        # Generate correlation matrix
        correlation_matrix = df[numeric_columns].corr()

        # Plot the heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        description = df.describe()
    description = df.describe()
    # Display the description
    st.write("### Descriptive Statistics", description)

    # Create a heatmap
    numeric_df = df.select_dtypes(include='number')
    description = numeric_df.describe()

    # Display the description
    st.write("### Descriptive Statistics (Numeric Columns Only)", description)

    # Create a heatmap
    # Filter out non-numeric and timedelta columns
    numeric_df = df.select_dtypes(include=['number'])

    # Generate descriptive statistics
    
    description = numeric_df.describe()
    st.write(description.dtype)
    
    # Create a heatmap
    st.write("### Heatmap of Descriptive Statistics")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(description, annot=True, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
def render_kaplan_meier(df):
    st.title("Kaplan-Meier Survival Curves")
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in df['Test Results'].unique():
        event_data = df[df['Test Results'] == result]
        if event_data['los'].notnull().any():
            kmf.fit(durations=event_data['los'], event_observed=event_data['Test Results'].notnull())  
            kmf.plot_survival_function(ax=ax, label=result)
    plt.title('Kaplan-Meier Survival Curves for Test Results')
    st.pyplot(fig)

def render_visualization(viz_type, df):
    if viz_type == "Pie Chart":
        st.title("Pie Chart: Distribution of Medical Conditions")
        condition_counts = df['Medical Condition'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            condition_counts, labels=condition_counts.index,
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3")
        )
        ax.axis('equal')
        st.pyplot(fig)

    elif viz_type == "Scatter Plot":
        st.title("Scatter Plot: Age vs. Billing Amount")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Age', y='Billing Amount', hue='Medical Condition', palette='Set2', ax=ax)
        plt.title('Scatter Plot of Age vs. Billing Amount')
        st.pyplot(fig)

    elif viz_type == "Bubble Chart":
        st.title("Bubble Chart: Age vs. Length of Stay")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Age', y='LOS', size='Billing Amount', hue='Medical Condition',
            sizes=(20, 200), palette='Set1', ax=ax
        )
        plt.title('Bubble Chart: Age vs. Length of Stay')
        st.pyplot(fig)

    elif viz_type == "Bar Chart":
        st.title("Bar Chart: Patient Count by Medical Condition")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='Medical Condition', palette='Set3', ax=ax)
        plt.title('Bar Chart of Patient Count by Medical Condition')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif viz_type == "Line Chart":
        st.title("Line Chart: Monthly Trends for Top Medical Conditions")
        condition_monthly_trend = (
            df.groupby(['Medical Condition', 'Admission_Month'])
            .size()
            .reset_index(name='Count')
        )
        top_conditions = df['Medical Condition'].value_counts().head(5).index
        fig, ax = plt.subplots(figsize=(12, 6))
        for condition in top_conditions:
            data = condition_monthly_trend[condition_monthly_trend['Medical Condition'] == condition]
            sns.lineplot(data=data, x='Admission_Month', y='Count', marker='o', label=condition, ax=ax)
        plt.title('Line Chart of Monthly Trends')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.legend(title='Medical Condition')
        st.pyplot(fig)

    elif viz_type == "Histogram":
        st.title("Histogram: Distribution of Billing Amount")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Billing Amount', kde=True, bins=20, color='blue', ax=ax)
        plt.title('Histogram of Billing Amount')
        st.pyplot(fig)

    elif viz_type == "Box Plot":
        st.title("Box Plot: Billing Amount by Medical Condition")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='Medical Condition', y='Billing Amount', palette='Set2', ax=ax)
        plt.title('Box Plot of Billing Amount by Medical Condition')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif viz_type == "Stacked Bar Chart":
        st.title("Stacked Bar Chart: Monthly Trends by Gender")
        gender_monthly_trend = df.groupby(['Gender', 'Admission_Month']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        gender_monthly_trend.T.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette("Set2"))
        plt.title('Stacked Bar Chart of Monthly Trends by Gender')
        plt.xlabel('Month')
        plt.ylabel('Patient Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif viz_type == "Stacked Area Chart":
        st.title("Stacked Area Chart: Monthly Trends for Top Medical Conditions")
        stacked_data = condition_monthly_trend.pivot(index='Admission_Month', columns='Medical Condition', values='Count').fillna(0)
        top_conditions = stacked_data.columns[:5]  # Top 5 conditions
        stacked_data = stacked_data[top_conditions]
        fig, ax = plt.subplots(figsize=(12, 6))
        stacked_data.plot(kind='area', stacked=True, ax=ax, colormap='Set3')
        plt.title('Stacked Area Chart of Monthly Trends')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Main Content Rendering
if selected_main_page == "Overview":
    render_overview(df)

elif selected_main_page == "Visualization Pages":
    render_visualization(selected_viz, df)

elif selected_main_page == "Kaplan-Meier Analysis":
    render_kaplan_meier(df)

elif selected_main_page == "EDA":
    st.title("exploratory data analysis")
    st.subheader("Billing Amount Distribution by Medical Condition")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df, x='Medical Condition', y='Billing Amount', palette='Set3', ax=ax)
    plt.title('Billing Amount by Medical Condition')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    df = pd.read_csv("healthcare.csv")
    st.write(df.head().T)
    
    # Dataframe Summary
    st.subheader("Data Summary")
    st.write("### Basic Information:")
    buffer = st.empty()
    with buffer.container():
        st.text(df.info())

    st.write("### Missing Values:")
    st.write(df.isna().sum())

    st.write("### Descriptive Statistics:")
    st.write(df.describe())

    # Histograms for Numerical Features
    st.subheader("Numerical Feature Distributions")
    st.write("Histograms for numerical columns:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(len(numerical_cols), 1, figsize=(10, 6 * len(numerical_cols)))

    for idx, col in enumerate(numerical_cols):
        sns.histplot(df[col], bins=50, kde=True, ax=ax[idx])
        ax[idx].set_title(f"Distribution of {col}")

    st.pyplot(fig)
    desc_stats = df.describe()
    st.write(desc_stats)
    # Compute descriptive statistics for numeric columns
    
    # Calculate the correlation matrix
    

    # Categorical Features
    st.subheader("Categorical Features")
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 
                        'Insurance Provider', 'Admission Type', 
                        'Medication', 'Test Results']

    for col in categorical_cols:
        st.write(f"### {col}")
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        df[col].value_counts().plot(kind='bar', ax=ax[0], color=sns.color_palette('tab10'))
        ax[0].set_title(f"{col} - Bar Chart")
        df[col].value_counts().plot(kind='pie', autopct="%.2f%%", ax=ax[1])
        ax[1].set_title(f"{col} - Pie Chart")
        st.pyplot(fig)

    # Billing Amount Analysis
    st.subheader("Billing Amount Analysis")
    for col in categorical_cols:
        char_bar = df.groupby(col)[['Billing Amount']].sum().reset_index()
        char_bar = char_bar.sort_values(by="Billing Amount", ascending=False)
        fig = px.bar(char_bar, x=col, y="Billing Amount", title=f"Billing Amount by {col}")
        st.plotly_chart(fig)

    # Days Hospitalized
    st.subheader("Days Hospitalized")
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Days Hospitalized'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    st.write("### Days Hospitalized Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Days Hospitalized'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Days Hospitalized")
    st.pyplot(fig)

    # Average Days Hospitalized by Categorical Features
    st.subheader("Average Days Hospitalized by Features")
    for col in categorical_cols:
        avg_days = df.groupby(col)[['Days Hospitalized']].mean().reset_index()
        fig = px.bar(avg_days, x=col, y="Days Hospitalized", title=f"Average Days Hospitalized by {col}")
        st.plotly_chart(fig)

    # Custom Insights
    st.subheader("Billing vs. Medical Conditions and Medications")
    df_trans = df.groupby(['Medical Condition', 'Medication'])[['Billing Amount']].sum().reset_index()
    fig = px.bar(df_trans, x='Medical Condition', y='Billing Amount', color='Medication', 
                title="Billing Amount by Medical Condition and Medication")
    st.plotly_chart(fig)

    st.subheader("Billing vs. Medical Conditions and Test Results")
    df_trans = df.groupby(['Medical Condition', 'Test Results'])[['Billing Amount']].sum().reset_index()
    fig = px.bar(df_trans, x='Medical Condition', y='Billing Amount', color='Test Results', 
    title="Billing Amount by Medical Condition and Test Results")
    st.plotly_chart(fig)
elif selected_main_page == "Time Series Analysis":
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    import streamlit as st

    # Assuming 'df' is your data, load or prepare the data here
    # For example: df = pd.read_csv('your_data.csv')

    # Convert 'Date of Admission' to datetime format
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])

    # Aggregate data by date
    daily_counts = df.groupby('Date of Admission').size().reset_index(name='Count')

    # Set the date as the index for time series analysis
    daily_counts.set_index('Date of Admission', inplace=True)

    # Streamlit App Title
    st.title("Time Series Analysis of Daily Admissions")

    # Plot the time series data
    st.subheader("Time Series Plot of Daily Admissions")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(daily_counts.index, daily_counts['Count'], label='Daily Admissions', color='blue')
    ax1.set_title("Time Series of Daily Admissions")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Number of Admissions")
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition of Admissions Data")

    decomposition = seasonal_decompose(daily_counts['Count'], model='additive', period=30)  # Assuming monthly seasonality

    # Plot the decomposition
    fig2 = decomposition.plot()
    st.pyplot(fig2)

    # Optionally, you can add more interactive features like parameter sliders, dropdowns, etc.


    
elif selected_main_page == "Seasonal Trends":
    if 'Admission_Month' in df.columns:
        st.write("### Monthly Trends in Admissions by Gender")
        monthly_trend = df.groupby(['Admission_Month', 'Gender']).size().reset_index(name='Count')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=monthly_trend, x='Admission_Month', y='Count', hue='Gender', marker='o', ax=ax, sort=False)
        ax.set_xticks(range(len(month_order)))
        ax.set_xticklabels(month_order)
        plt.title('Monthly Admission Trends by Gender')
        plt.xlabel('Month')
        plt.ylabel('Patient Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

elif selected_main_page == "Correlation Heatmap":
    st.title("Correlation Heatmap")
    
    desc_stats = df.describe()
    st.write(desc_stats)

    # Calculate the correlation matrix
    st.subheader("Correlation Matrix Heatmap")
    corr_matrix = df.corr()

    # Calculate the correlation matrix
    st.subheader("Correlation Matrix Heatmap")
    corr_matrix = desc_stats.corr()

    st.subheader("Descriptive Statistics")
    desc_stats = df.describe()
    st.write(desc_stats)

    df['Date of Admission'] = pd.to_datetime(data['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(data['Discharge Date'], errors='coerce')
    
    # Calculate Length of Stay (LOS) in days
    data['Length of Stay'] = (data['Discharge Date'] - data['Date of Admission']).dt.days
    stats_data = df[['Age', 'Billing Amount', 'Room Number', 'Length of Stay']].describe()
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(stats_data, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=0.5, ax=ax)
    ax.set_title("Statistic Heatmap of Numeric Features", fontsize=16)
    ax.set_xlabel("Statistics", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    plt.xticks(rotation=45)

        # Display the heatmap
    st.pyplot(fig)

elif selected_main_page == "Length of Stay Analysis":
    # Preprocessing
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Admission_Month'] = pd.Categorical(df['Date of Admission'].dt.strftime('%b'), categories=month_order, ordered=True)
    df['Admission_Year'] = df['Date of Admission'].dt.year
    df['Admission_Quarter'] = df['Date of Admission'].dt.quarter
    df['Admission_Day'] = df['Date of Admission'].dt.day

    # Length of Stay Distribution by Medical Condition
    st.subheader("Average Length of Stay by Medical Condition")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=df.groupby('Medical Condition')['LOS'].mean().reset_index(),
        x='Medical Condition',
        y='LOS',
        palette='Set1',
        ax=ax
    )
    ax.set_title('Average Length of Stay by Medical Condition')
    ax.set_ylabel('Average LOS (days)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Top 10 Doctors by Revenue
    st.subheader("Top 10 Doctors by Revenue Generated")
    revenue_df = df.groupby("Doctor")["Billing Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=revenue_df, x='Doctor', y='Billing Amount', palette="viridis", ax=ax)
    ax.set_title("Top 10 Doctors by Revenue Generated")
    ax.set_ylabel("Total Revenue ($)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Patient Outcomes Distribution by Top 10 Doctors
    st.subheader("Patient Outcomes Distribution by Top 10 Doctors")
    top_doctors = revenue_df['Doctor']
    outcomes_data = df[df["Doctor"].isin(top_doctors)]
    outcomes_pivot = outcomes_data.pivot_table(index="Doctor", columns="Test Results", aggfunc="size", fill_value=0).loc[top_doctors]

    fig, ax = plt.subplots(figsize=(12, 6))
    outcomes_pivot.plot(kind="bar", stacked=True, ax=ax, color=["#5DADE2", "#58D68D", "#F5B041"])
    ax.set_title("Patient Outcomes Distribution by Top 10 Doctors")
    ax.set_ylabel("Number of Patients")
    ax.set_xlabel("Doctor")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Specialization Analysis by Top 10 Doctors
    st.subheader("Specialization Analysis by Top 10 Doctors")
    specialization_pivot = outcomes_data.pivot_table(index="Doctor", columns="Medical Condition", aggfunc="size", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    specialization_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Specialization Analysis by Top 10 Doctors (Medical Conditions)")
    ax.set_ylabel("Number of Cases")
    ax.set_xlabel("Doctor")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Average Length of Stay by Top 10 Doctors
    st.subheader("Average Length of Stay by Top 10 Doctors")
    avg_length_stay = df.groupby("Doctor")["LOS"].mean().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=avg_length_stay, x='Doctor', y='LOS', palette="coolwarm", ax=ax)
    ax.set_title("Average Length of Stay by Top 10 Doctors")
    ax.set_ylabel("Average LOS (days)")
    ax.set_xlabel("Doctor")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
elif selected_main_page == "Dimensionality Reduction":
    # Dimensionality Reduction code (PCA + t-SNE)
    selected_columns = ['Age', 'Billing Amount', 'Room Number', 'Gender', 'Medical Condition']
    subset_data = df[selected_columns]

    # Encoding categorical variables
    label_encoders = {}
    for col in ['Gender', 'Medical Condition']:
        le = LabelEncoder()
        subset_data[col] = le.fit_transform(subset_data[col])
        label_encoders[col] = le

    # Normalizing the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(subset_data)

    # PCA & t-SNE
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(normalized_data)
    tsne = TSNE(n_components=2)
    tsne_components = tsne.fit_transform(pca_components)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x=tsne_components[:, 0], y=tsne_components[:, 1], hue=df['Medical Condition'], palette='Set2', ax=ax)
    plt.title('t-SNE Visualization After PCA')
    st.pyplot(fig)
#Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**© 2024 Healthcare Dashboard**")
