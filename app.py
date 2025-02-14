# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pandas_datareader import data as web
# from datetime import datetime
# from sklearn.linear_model import LinearRegression, LassoCV
# from sklearn.metrics import mean_squared_error, r2_score

# # Set up seaborn style for plots
# sns.set(style='whitegrid')

# # =============================================================================
# # Helper Functions
# # =============================================================================
# @st.cache_data(show_spinner=True)
# def load_data(start_date, end_date):
#     """
#     Download FRED data and resample to a common monthly frequency.
#     """
#     # Download data from FRED
#     cs_index = web.DataReader('CSUSHPISA', 'fred', start_date, end_date)
#     gdp = web.DataReader('GDP', 'fred', start_date, end_date)
#     unrate = web.DataReader('UNRATE', 'fred', start_date, end_date)
#     mortgage = web.DataReader('MORTGAGE30US', 'fred', start_date, end_date)
#     income = web.DataReader('W875RX1', 'fred', start_date, end_date)
#     cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

#     # Resample each series to monthly frequency using month start ('MS')
#     cs_index_monthly = cs_index.resample('MS').ffill()
#     gdp_monthly = gdp.resample('MS').ffill()         # GDP is quarterly; resample to monthly.
#     unrate_monthly = unrate.resample('MS').ffill()
#     mortgage_monthly = mortgage.resample('MS').ffill()
#     income_monthly = income.resample('MS').ffill()
#     cpi_monthly = cpi.resample('MS').ffill()

#     # Merge all data into a single DataFrame
#     df = pd.DataFrame({
#         'CSUSHPISA': cs_index_monthly['CSUSHPISA'],
#         'GDP': gdp_monthly['GDP'],
#         'UNRATE': unrate_monthly['UNRATE'],
#         'MORTGAGE30US': mortgage_monthly['MORTGAGE30US'],
#         'INCOME': income_monthly['W875RX1'],
#         'CPI': cpi_monthly['CPIAUCSL']
#     })

#     # Drop any rows with missing values
#     df.dropna(inplace=True)
#     return df

# def feature_engineering(df):
#     """
#     Compute month-over-month percentage changes for key variables.
#     """
#     df['CSUSHPISA_pct'] = df['CSUSHPISA'].pct_change() * 100
#     df['GDP_pct'] = df['GDP'].pct_change() * 100
#     df['INCOME_pct'] = df['INCOME'].pct_change() * 100
#     df['CPI_pct'] = df['CPI'].pct_change() * 100
#     df = df.dropna()
#     return df

# def run_models(df):
#     """
#     Build and evaluate Linear Regression and Lasso Regression models.
#     Returns a dictionary of metrics and figures.
#     """
#     # Define predictors and target variable
#     X = df[['GDP_pct', 'UNRATE', 'MORTGAGE30US', 'INCOME_pct', 'CPI_pct']]
#     y = df['CSUSHPISA_pct']

#     # Split the data chronologically (80% train, 20% test)
#     split_index = int(len(df) * 0.8)
#     X_train = X.iloc[:split_index]
#     X_test = X.iloc[split_index:]
#     y_train = y.iloc[:split_index]
#     y_test = y.iloc[split_index:]

#     # Linear Regression
#     lr_model = LinearRegression()
#     lr_model.fit(X_train, y_train)
#     y_pred_lr = lr_model.predict(X_test)
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)

#     # Lasso Regression with Cross-Validation
#     lasso_model = LassoCV(cv=5, random_state=42)
#     lasso_model.fit(X_train, y_train)
#     y_pred_lasso = lasso_model.predict(X_test)
#     mse_lasso = mean_squared_error(y_test, y_pred_lasso)
#     r2_lasso = r2_score(y_test, y_pred_lasso)

#     # Package results in a dictionary
#     results = {
#         'lr': {
#             'model': lr_model,
#             'mse': mse_lr,
#             'r2': r2_lr,
#             'y_test': y_test,
#             'y_pred': y_pred_lr
#         },
#         'lasso': {
#             'model': lasso_model,
#             'mse': mse_lasso,
#             'r2': r2_lasso,
#             'y_test': y_test,
#             'y_pred': y_pred_lasso,
#             'coefficients': pd.DataFrame({
#                 'Feature': X.columns,
#                 'Coefficient': lasso_model.coef_
#             })
#         }
#     }
#     return results

# def plot_actual_vs_pred(y_test, y_pred, title):
#     """
#     Generate a matplotlib plot of actual vs predicted values.
#     """
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(y_test.index, y_test, label='Actual', marker='o')
#     ax.plot(y_test.index, y_pred, label='Predicted', marker='x')
#     ax.set_title(title)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('% Change')
#     ax.legend()
#     plt.tight_layout()
#     return fig

# # =============================================================================
# # Streamlit App
# # =============================================================================

# st.title("US Home Price Analysis with FRED Data")
# st.markdown("""
# This Streamlit app downloads economic data from FRED, merges and processes it, 
# and builds regression models to explain the monthly percentage changes in the S&P/Case–Shiller Home Price Index.
# """)

# # Sidebar options for date range
# st.sidebar.header("Settings")
# start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2003-01-01")
# end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2023-12-31")

# st.sidebar.markdown("---")
# if st.sidebar.button("Load and Process Data"):
#     with st.spinner("Loading data from FRED..."):
#         raw_df = load_data(start_date, end_date)
#         st.write("### Raw Merged Data")
#         st.write(raw_df.head())
#         st.write("Shape:", raw_df.shape)
        
#         processed_df = feature_engineering(raw_df)
#         st.write("### Data after Feature Engineering")
#         st.write(processed_df.head())
#         st.write("Shape:", processed_df.shape)
        
#         st.session_state['df'] = processed_df
#         st.success("Data loaded and processed!")

# if 'df' in st.session_state:
#     df = st.session_state['df']

#     st.write("#### S&P/Case–Shiller Home Price Index")
#     fig1, ax1 = plt.subplots(figsize=(12, 4))
#     ax1.plot(df.index, df['CSUSHPISA'], label="CSUSHPISA")
#     ax1.set_xlabel("Date")
#     ax1.set_ylabel("Index Value")
#     ax1.legend()
#     st.pyplot(fig1)

#     st.write("#### Correlation Heatmap")
#     corr_fig, ax_corr = plt.subplots(figsize=(8, 6))
#     cols_for_corr = ['CSUSHPISA_pct', 'GDP_pct', 'UNRATE', 'MORTGAGE30US', 'INCOME_pct', 'CPI_pct']
#     sns.heatmap(df[cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
#     st.pyplot(corr_fig)

#     # Model selection
#     st.header("Regression Modeling")
#     model_choice = st.radio("Choose a model to evaluate", ("Linear Regression", "Lasso Regression"))

#     results = run_models(df)

#     if model_choice == "Linear Regression":
#         st.subheader("Linear Regression Results")
#         st.write(f"Mean Squared Error: {results['lr']['mse']:.4f}")
#         st.write(f"R² Score: {results['lr']['r2']:.4f}")
#         fig_lr = plot_actual_vs_pred(results['lr']['y_test'], results['lr']['y_pred'], 
#                                      "Linear Regression: Actual vs Predicted % Change")
#         st.pyplot(fig_lr)
#     else:
#         st.subheader("Lasso Regression Results")
#         st.write(f"Mean Squared Error: {results['lasso']['mse']:.4f}")
#         st.write(f"R² Score: {results['lasso']['r2']:.4f}")
#         st.write("#### Feature Coefficients (Lasso)")
#         st.write(results['lasso']['coefficients'])
#         fig_lasso = plot_actual_vs_pred(results['lasso']['y_test'], results['lasso']['y_pred'], 
#                                         "Lasso Regression: Actual vs Predicted % Change")
#         st.pyplot(fig_lasso)
        
#     st.markdown("---")
#     st.write("This app was developed to showcase how to deploy a data science model using Streamlit.")












import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web
from datetime import datetime
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Set up seaborn style for plots
sns.set(style='whitegrid')

# =============================================================================
# Helper Functions
# =============================================================================
@st.cache_data(show_spinner=True)
def load_data(start_date, end_date):
    """
    Download FRED data and resample to monthly frequency.
    """
    cs_index = web.DataReader('CSUSHPISA', 'fred', start_date, end_date)
    gdp = web.DataReader('GDP', 'fred', start_date, end_date)
    unrate = web.DataReader('UNRATE', 'fred', start_date, end_date)
    mortgage = web.DataReader('MORTGAGE30US', 'fred', start_date, end_date)
    income = web.DataReader('W875RX1', 'fred', start_date, end_date)
    cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

    # Resample each series to a monthly frequency using month start
    cs_index_monthly = cs_index.resample('MS').ffill()
    gdp_monthly = gdp.resample('MS').ffill()
    unrate_monthly = unrate.resample('MS').ffill()
    mortgage_monthly = mortgage.resample('MS').ffill()
    income_monthly = income.resample('MS').ffill()
    cpi_monthly = cpi.resample('MS').ffill()

    # Merge into one DataFrame
    df = pd.DataFrame({
        'CSUSHPISA': cs_index_monthly['CSUSHPISA'],
        'GDP': gdp_monthly['GDP'],
        'UNRATE': unrate_monthly['UNRATE'],
        'MORTGAGE30US': mortgage_monthly['MORTGAGE30US'],
        'INCOME': income_monthly['W875RX1'],
        'CPI': cpi_monthly['CPIAUCSL']
    })

    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    """
    Compute month-over-month percentage changes for selected variables.
    """
    df['CSUSHPISA_pct'] = df['CSUSHPISA'].pct_change() * 100
    df['GDP_pct'] = df['GDP'].pct_change() * 100
    df['INCOME_pct'] = df['INCOME'].pct_change() * 100
    df['CPI_pct'] = df['CPI'].pct_change() * 100
    df = df.dropna()
    return df

def run_models(df):
    """
    Build and evaluate Linear Regression and Lasso Regression models.
    Returns a dictionary containing models, metrics, and predictions.
    """
    X = df[['GDP_pct', 'UNRATE', 'MORTGAGE30US', 'INCOME_pct', 'CPI_pct']]
    y = df['CSUSHPISA_pct']

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Lasso Regression with CV
    lasso_model = LassoCV(cv=5, random_state=42)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    results = {
        'lr': {
            'model': lr_model,
            'mse': mse_lr,
            'r2': r2_lr,
            'y_test': y_test,
            'y_pred': y_pred_lr
        },
        'lasso': {
            'model': lasso_model,
            'mse': mse_lasso,
            'r2': r2_lasso,
            'y_test': y_test,
            'y_pred': y_pred_lasso,
            'coefficients': pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lasso_model.coef_
            })
        }
    }
    return results

def plot_actual_vs_pred(y_test, y_pred, title):
    """
    Create a matplotlib plot of actual vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual', marker='o')
    ax.plot(y_test.index, y_pred, label='Predicted', marker='x')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('% Change')
    ax.legend()
    plt.tight_layout()
    return fig

# =============================================================================
# Streamlit App Layout
# =============================================================================
st.title("US Home Price Analysis and Prediction")
st.markdown("""
This application downloads economic data from FRED, processes it, and builds regression models 
to explain and predict the monthly percentage change in the S&P/Case–Shiller Home Price Index.
""")

# Sidebar for date range input
st.sidebar.header("Data Settings")
start_date_input = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2003-01-01")
end_date_input = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2023-12-31")

# Button to load and process data
if st.sidebar.button("Load and Process Data"):
    with st.spinner("Downloading and processing data..."):
        raw_df = load_data(start_date_input, end_date_input)
        st.write("### Raw Merged Data")
        st.dataframe(raw_df.head())
        st.write("Shape:", raw_df.shape)
        
        processed_df = feature_engineering(raw_df)
        st.write("### Data after Feature Engineering")
        st.dataframe(processed_df.head())
        st.write("Shape:", processed_df.shape)
        st.session_state['df'] = processed_df
        st.success("Data loaded and processed!")

# If data is loaded, display plots and run models
if 'df' in st.session_state:
    df = st.session_state['df']
    
    st.write("#### S&P/Case–Shiller Home Price Index")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df.index, df['CSUSHPISA'], label="CSUSHPISA")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Value")
    ax1.legend()
    st.pyplot(fig1)
    
    st.write("#### Correlation Heatmap")
    corr_fig, ax_corr = plt.subplots(figsize=(8, 6))
    cols_for_corr = ['CSUSHPISA_pct', 'GDP_pct', 'UNRATE', 'MORTGAGE30US', 'INCOME_pct', 'CPI_pct']
    sns.heatmap(df[cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(corr_fig)
    
    st.header("Regression Modeling")
    model_choice = st.radio("Choose a model to evaluate", ("Linear Regression", "Lasso Regression"))
    
    results = run_models(df)
    
    if model_choice == "Linear Regression":
        st.subheader("Linear Regression Results")
        st.write(f"Mean Squared Error: {results['lr']['mse']:.4f}")
        st.write(f"R² Score: {results['lr']['r2']:.4f}")
        fig_lr = plot_actual_vs_pred(results['lr']['y_test'], results['lr']['y_pred'], 
                                     "Linear Regression: Actual vs Predicted % Change")
        st.pyplot(fig_lr)
    else:
        st.subheader("Lasso Regression Results")
        st.write(f"Mean Squared Error: {results['lasso']['mse']:.4f}")
        st.write(f"R² Score: {results['lasso']['r2']:.4f}")
        st.write("#### Feature Coefficients (Lasso)")
        st.dataframe(results['lasso']['coefficients'])
        fig_lasso = plot_actual_vs_pred(results['lasso']['y_test'], results['lasso']['y_pred'], 
                                        "Lasso Regression: Actual vs Predicted % Change")
        st.pyplot(fig_lasso)
    
    # =============================================================================
    # 5. Prediction Section
    # =============================================================================
    st.header("Predict New Output")
    st.markdown("""
    Enter values for the predictor variables to predict the monthly % change in the Home Price Index.
    The predictors are:
    - **GDP_pct**: Month-over-month percentage change in GDP.
    - **UNRATE**: Unemployment Rate.
    - **MORTGAGE30US**: 30-Year Mortgage Rate.
    - **INCOME_pct**: Month-over-month percentage change in disposable personal income.
    - **CPI_pct**: Month-over-month percentage change in CPI.
    """)
    
    predict_model_choice = st.radio("Select Model for Prediction", ("Linear Regression", "Lasso Regression"))
    
    with st.form("prediction_form"):
        gdp_pct_input = st.number_input("GDP Percentage Change", value=0.0, format="%.2f")
        unrate_input = st.number_input("Unemployment Rate", value=0.0, format="%.2f")
        mortgage_input = st.number_input("30-Year Mortgage Rate", value=0.0, format="%.2f")
        income_pct_input = st.number_input("Income Percentage Change", value=0.0, format="%.2f")
        cpi_pct_input = st.number_input("CPI Percentage Change", value=0.0, format="%.2f")
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        new_data = pd.DataFrame({
            'GDP_pct': [gdp_pct_input],
            'UNRATE': [unrate_input],
            'MORTGAGE30US': [mortgage_input],
            'INCOME_pct': [income_pct_input],
            'CPI_pct': [cpi_pct_input]
        })
        if predict_model_choice == "Linear Regression":
            pred_value = results['lr']['model'].predict(new_data)[0]
        else:
            pred_value = results['lasso']['model'].predict(new_data)[0]
        st.success(f"Predicted % Change in Home Price Index: {pred_value:.2f}%")
    
    st.markdown("---")
   