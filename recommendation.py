import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

def process_data(refined_df):
    """
    Process the refined dataset and create the vector store.
    
    Args:
        refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
        
    Returns:
        vectorstore (FAISS): Vector store containing the processed data.
    """
    refined_df['combined_info'] = refined_df.apply(lambda row: f"Product ID: {row['pid']}. Product URL: {row['product_url']}. Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. Brand: {row['brand']}. Gender: {row['gender']}", axis=1)

    loader = DataFrameLoader(refined_df, page_content_column="combined_info")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def save_vectorstore(vectorstore, directory):
    """
    Save the vector store to a directory.
    
    Args:
        vectorstore (FAISS): Vector store to be saved.
        directory (str): Directory to save the vector store.
    """
    vectorstore.save_local(directory)

def load_vectorstore(directory, embeddings):
    """
    Load the vector store from a directory.
    
    Args:
        directory (str): Directory containing the saved vector store.
        embeddings (OpenAIEmbeddings): Embeddings object.
        
    Returns:
        vectorstore (FAISS): Loaded vector store.
    """
    vectorstore = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization = True  )
    return vectorstore

def display_product_recommendation(refined_df):
    """
    Display product recommendation section.
    
    Args:
        refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
    """
    st.header("Product Recommendation")

    vectorstore_dir = 'vectorstore'

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(vectorstore_dir):
        vectorstore = load_vectorstore(vectorstore_dir, embeddings)
    else:
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    manual_template = """
    Kindly suggest three similar products based on the description I have provided below:

    Product Department: {department},
    Product Category: {category},
    Product Brand: {brand},
    Maximum Price range: {price}.

    Please provide complete answers including product department name, product category, product name, price, and stock quantity.
    """
    prompt_manual = PromptTemplate(
        input_variables=["department", "category", "brand", "price"],
        template=manual_template,
    )

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     model_name='gpt-3.5-turbo', temperature=0)

    chain = LLMChain(
        llm=llm,
        prompt=prompt_manual,
        verbose=True)

    chatbot_template = """
    You are a friendly, conversational retail shopping assistant that helps customers find products that match their preferences.
    From the following context and chat history, assist customers in finding what they are looking for based on their input.
    For each question, suggest three products, including their category, price, and current stock quantity.
    Sort the answer by the cheapest product.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Chat history: {history}

    Input: {question}
    Your Response:
    """
    chatbot_prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=chatbot_template,
    )

    memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": chatbot_prompt,
            "memory": memory}
    )

    department = st.text_input("Product Department")
    category = st.text_input("Product Category")
    brand = st.text_input("Product Brand")
    price = st.text_input("Maximum Price Range")

    if st.button("Get Recommendations"):
        response = chain.run(
            department=department,
            category=category,
            brand=brand,
            price=price
        )
        st.write(response)