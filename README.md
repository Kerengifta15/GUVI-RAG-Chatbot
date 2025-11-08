# 🤖 GUVI RAG Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions about **GUVI** — including FAQs, course details, and other information — using **Google Gemini** and **FAISS** for intelligent search and response generation.


## 🚀 Features

- 🧠 Uses **Gemini API** for accurate and natural AI responses  
- 🔍 Retrieves real information from GUVI data files  
- ⚡ Fast semantic search powered by **FAISS**  
- 💬 Simple, interactive chat UI built with **Streamlit**  
- 💾 Maintains chat history with downloadable transcripts  

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | Web UI framework for the chatbot |
| **Sentence Transformers** | To generate embeddings for text search |
| **FAISS** | Vector similarity search for fast information retrieval |
| **Google Gemini API** | For generating natural language answers |

## 🗂️ Project Structure
GUVI_RAG_Chatbot/
│
├── app/
│ ├── streamlit_app.py # Main Streamlit application
│ ├── extract_guvi_data.py # Script to extract GUVI text data
│ └── data_processing.py # Handles text cleaning and formatting
│
├── data/
│ ├── www_guvi_in_faq.txt # FAQ data source
│ └── guvi_faq.txt # Additional GUVI text data
│
├── venv/ # Virtual environment (not uploaded to GitHub)
├── requirements.txt # Project dependencies
└── README.md # Project documentation


## 🧩 How It Works

1. **Loads GUVI data** (FAQs, blogs, and text files)  
2. **Splits text** into smaller chunks for better context retrieval  
3. **Embeds chunks** into vector space using Sentence Transformers  
4. **Searches relevant chunks** using FAISS  
5. **Generates accurate answers** using Google Gemini AI  

## ⚙️ Installation & Usage

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/GUVI_RAG_Chatbot.git
cd GUVI_RAG_Chatbot

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set your Gemini API key

Create a .env file or use environment variables:

set GEMINI_API_KEY=your_api_key_here

5️⃣ Run the Streamlit app
streamlit run app/streamlit_app.py

📊 Evaluation Metrics

Retrieval Accuracy: Precision@K, Recall@K

Response Relevance: BLEU / ROUGE / Human Evaluation

Latency: System response time tracking

Code Quality: PEP8 compliance and modular structure

🧑‍💻 Developer
Keren Gifta A

🏁 Acknowledgements

Special thanks to GUVI for providing data and resources, and to Google Gemini for powering the AI responses.
