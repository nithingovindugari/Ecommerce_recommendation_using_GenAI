
# E-commerce Product Recommendation System with Gen AI

This project is an end-to-end e-commerce product recommendation system that utilizes the power of natural language processing (NLP) , RAG Techniques,  machine learning to provide personalized product recommendations to users. The system is built using Python and leverages various libraries and frameworks such as Streamlit, Pandas, Seaborn, Matplotlib, and Langchain. Fundamental different from traditional recommendation systems, this project uses RAG techniques to generate recommendations based on user input and preference rather than collaborative filtering or content-based filtering. The system also understands user queries and provides relevant product suggestions using NLP techniques. The front-end of the system is built using Streamlit, allowing users to easily interact with the recommendation system through an intuitive user interface and options 
are not limited to the dataset predefined categories and brands, users can input their own preferences such as product department, category, brand, and maximum price range, and the system generates personalized product recommendations based on their input understand semantic meaning of the user input and provide relevant product suggestions. 

## Features

- **Data Analysis:** The system performs comprehensive data analysis on the e-commerce dataset, including visualizations of price distribution across top categories and discount percentage distribution.
- **Product Recommendation:** Users can input their preferences, such as product department, category, brand, and maximum price range, and the system generates personalized product recommendations based on their input.
- **NLP-powered Search:** The system utilizes NLP techniques to understand user queries and provide relevant product suggestions.
- **Efficient Data Processing:** The dataset is preprocessed and tokenized to extract relevant information and create a vector store for efficient retrieval of product recommendations.
- **Persistent Storage:** The processed data and vector store are saved to disk, eliminating the need for repeated data processing and reducing API call costs.
- **Interactive UI:** The system features an intuitive and user-friendly interface built with Streamlit, allowing users to easily interact with the recommendation system.

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/ecommerce-product-recommendation.git
```

Navigate to the project directory:
```bash
cd ecommerce-product-recommendation
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Set up the OpenAI API key:
- Create a .env file in the project root directory.
- Add your OpenAI API key to the .env file in the following format:
```
OPENAI_API_KEY=your-api-key
```

## Usage

- Prepare your e-commerce dataset in CSV format and place it in the project directory.
- Update the `dataset_path` variable in the `app.py` file with the path to your dataset file.
- Run the Streamlit app:
```bash
streamlit run app.py
```
- Access the application through the provided URL in your web browser.
- Explore the data analysis visualizations and interact with the product recommendation system by providing your preferences.

## Dataset

The project utilizes the Flipkart e-commerce dataset, which contains information about various products sold on the Flipkart platform. The dataset includes details such as product name, description, category, price, brand, and more. You can replace the dataset with your own e-commerce dataset in CSV format.

## Set up the vector database file:

The project uses a large vector database file (vectorstore/index.faiss) for efficient product recommendation retrieval.
Due to the file size limitation on GitHub, the vectorstore/index.faiss file is not included in the repository.
To obtain the vectorstore/index.faiss file just select product recommendation and the file will be generated in the project directory ( it is only generated once and can be used for future recommendations ).



## Project Structure

```
ecommerce-product-recommendation/
├── app.py
├── data_processing.py
├── recommendation_utils.py
├── requirements.txt
├── .env
└── README.md
```
- `app.py`: The main Streamlit application file that handles the user interface and integrates different components of the system.
- `data_processing.py`: Contains functions for data preprocessing, cleaning, and analysis.
- `recommendation_utils.py`: Implements the product recommendation system using NLP and machine learning techniques.
- `requirements.txt`: Lists the required Python dependencies for the project.
- `.env`: Environment file to store the OpenAI API key.
- `README.md`: Provides an overview of the project, installation instructions, and usage guidelines.

## Dependencies

The project relies on the following major dependencies:
- **Streamlit**: For building the interactive user interface.
- **Pandas**: For data manipulation and analysis.
- **Seaborn and Matplotlib**: For data visualization.
- **Langchain**: For building the language model and recommendation system.
- **OpenAI**: For leveraging pre-trained language models and embeddings.

For a complete list of dependencies, please refer to the `requirements.txt` file.


## Future Enhancements

- Implement user authentication and personalized user profiles.
- Expand the recommendation system to include user reviews and ratings.
- Integrate with a real-time e-commerce platform for seamless product recommendations.
- Optimize the system for scalability and performance to handle large-scale datasets.
- Explore advanced NLP techniques and deep learning models for enhanced recommendation accuracy.

## Contributing

Contributions to the project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Make sure to follow the project's code of conduct.


## License

This project is licensed under the MIT License.
