import streamlit as st
from data_processing import load_data, preprocess_data, display_data_analysis
from recommendation import display_product_recommendation

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("E-commerce Product Recommendation")

    dataset_path = 'flipkart_com-ecommerce_sample.csv'
    df = load_data(dataset_path)
    
    if df is not None:
        refined_df = preprocess_data(df)

        option = st.sidebar.selectbox("Select an option", ("Data Analysis", "Product Recommendation"))

        if option == "Data Analysis":
            display_data_analysis(refined_df)
        elif option == "Product Recommendation":
            display_product_recommendation(refined_df)

if __name__ == '__main__':
    main()