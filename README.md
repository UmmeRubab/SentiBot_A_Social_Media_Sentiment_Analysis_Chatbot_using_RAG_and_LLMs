# SentiBot 2.0 — Social Media Sentiment Intelligence Using RAG and LLMs

**An AI-powered application combining transformer-based sentiment analysis, semantic search, and Retrieval-Augmented Generation (RAG).**

**Live Demo:** https://sentibott.streamlit.app/

**GitHub Repository:** https://github.com/UmmeRubab/SentiBot_A_Social_Media_Sentiment_Analysis_Chatbot_using_RAG_and_LLMs

## Overview

SentiBot 2.0 is an AI-powered social media sentiment analysis application that integrates transformer-based sentiment classification, semantic search, and Retrieval-Augmented Generation (RAG).

The application uses RoBERTa to classify sentiment, SentenceTransformers and FAISS to retrieve relevant social media posts, and Google Gemini to generate natural-language responses grounded in retrieved evidence.

Built with Python and Streamlit, SentiBot allows users to upload datasets, analyze opinions, ask questions, inspect supporting posts, and export results.

## Project Background

My previous academic research focused on aspect-based sentiment analysis through my MS Computer Science thesis, **“Towards an Aspect-Based Sentiment Classifier for Health and Fitness Apps.”**

Building on my experience in NLP and sentiment classification, I developed SentiBot to explore modern AI technologies, including transformer-based language models, semantic search, vector retrieval, and Retrieval-Augmented Generation (RAG).

While my thesis focused on aspect extraction and sentiment classification using traditional machine learning techniques, SentiBot extends my practical experience into transformer-based NLP and LLM-powered applications.

## Project Objective

The objective of SentiBot is to develop an interactive application that goes beyond basic positive, negative, and neutral sentiment classification.

By integrating semantic retrieval with LLM-generated responses, the application enables users to explore social media opinions through natural-language questions while examining the posts supporting the generated insights.

## Key Features

### 1. Transformer-Based Sentiment Analysis

- Uses the pretrained `cardiffnlp/twitter-roberta-base-sentiment-latest` model.
- Classifies retrieved social media posts as positive, negative, or neutral.
- Displays predicted sentiment labels alongside supporting posts.
- Uses pretrained transformer inference without additional fine-tuning.

### 2. Semantic Search

- Uses `sentence-transformers/all-MiniLM-L6-v2` to generate dense sentence embeddings.
- Converts social media posts and user questions into vector representations.
- Retrieves semantically relevant content rather than relying solely on keyword matching.

### 3. FAISS Vector Retrieval

- Builds a FAISS index from uploaded social media posts.
- Retrieves the top-K most relevant posts for each question.
- Uses normalized embeddings and inner-product similarity for semantic retrieval.

### 4. Retrieval-Augmented Generation

- Integrates Google Gemini with retrieved social media posts.
- Generates contextual answers based on the retrieved evidence.
- References supporting posts using identifiers such as `[P1]` and `[P2]`.
- Instructs the LLM to avoid unsupported claims and acknowledge insufficient evidence.

### 5. Interactive Streamlit Application

- Upload social media datasets in CSV format.
- Paste individual posts manually.
- Ask natural-language questions about user opinions.
- Adjust the number of retrieved posts.
- View sentiment summaries and supporting evidence.
- Export retrieved posts and sentiment predictions as CSV files.

## System Architecture

```text
              Social Media Posts
                       |
                       v
           Data Cleaning & Deduplication
                       |
                       v
             SentenceTransformers
             (MiniLM Embeddings)
                       |
                       v
                FAISS Vector Index
                       |
                       v
               User's NLP Question
                       |
                       v
              Semantic Retrieval
                 (Top-K Posts)
                       |
                       v
             RoBERTa Sentiment Model
                       |
                       v
          Retrieved Posts + Sentiment
                       |
                       v
                  Google Gemini
                       |
                       v
          Evidence-Grounded AI Answer
                       |
                       v
             Streamlit User Interface
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Frontend | Streamlit |
| NLP and Transformers | Hugging Face Transformers, RoBERTa |
| Sentence Embeddings | SentenceTransformers, MiniLM |
| Vector Search | FAISS |
| Generative AI | Google Gemini |
| Data Processing | Pandas, NumPy |
| Deep Learning Runtime | PyTorch |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

## How SentiBot Works

1. **Data input:** Users upload a CSV dataset or manually enter social media posts.
2. **Preprocessing:** Empty and duplicate posts are removed.
3. **Embedding generation:** SentenceTransformers converts posts into numerical vector representations.
4. **Semantic retrieval:** FAISS identifies the posts most relevant to the user's question.
5. **Sentiment classification:** RoBERTa predicts sentiment labels for the retrieved posts.
6. **LLM integration:** Gemini receives the question, retrieved evidence, and sentiment predictions.
7. **Response generation:** Gemini generates an answer referencing the supporting posts.
8. **Results:** Streamlit displays the generated answer, sentiment summaries, and retrieved evidence.

## Example Usage

**Sample social media posts:**

- "The new update is amazing and the app runs much faster now."
- "Customer support is terrible. Nobody responds to my complaints."
- "The interface looks beautiful, but the subscription is too expensive."
- "I love the new features, especially the personalized recommendations."
- "The app crashes whenever I try to make a payment."

**Example question:**

> What are the main positive and negative opinions about this application? Cite the supporting posts.

SentiBot retrieves relevant posts, predicts their sentiment, and generates a contextual response referencing the available evidence.

Generated responses depend on the input data, retrieval results, transformer predictions, and LLM output.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/UmmeRubab/SentiBot_A_Social_Media_Sentiment_Analysis_Chatbot_using_RAG_and_LLMs.git
```

```bash
cd SentiBot_A_Social_Media_Sentiment_Analysis_Chatbot_using_RAG_and_LLMs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API

Create a `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "YOUR_SUPPORTED_GEMINI_MODEL"
```

Use a Gemini model available to your API account.

Never upload your actual API key to GitHub.

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Live Application

SentiBot 2.0 is deployed on Streamlit Community Cloud.

**Try SentiBot:** https://sentibott.streamlit.app/

## Current Limitations

SentiBot is an applied AI application and has not undergone formal benchmarking.

Current limitations include:

- Sentiment classification relies on a pretrained model and may be less reliable for multilingual or code-mixed posts.
- Retrieved posts may not represent the entire dataset or broader social media discussions.
- Generated responses may contain inaccuracies despite evidence-grounding instructions.
- Source references identify retrieved posts but do not independently verify every generated claim.
- Model output scores are not calibrated probabilities.
- LLM response time depends on API availability, model performance, and usage limits.
- The application does not currently include a systematic evaluation of retrieval accuracy or answer faithfulness.

## Potential Improvements

Future enhancements may include:

- Evaluating retrieval relevance and citation accuracy.
- Comparing RAG-generated responses with LLM-only responses.
- Supporting multilingual and code-mixed sentiment analysis.
- Improving handling of sarcasm and ambiguous opinions.
- Adding larger-scale batch sentiment analytics.
- Measuring latency and performance across different retrieval settings.

These are potential enhancements, not completed experimental findings.

## Project Status

**Deployed AI Application**

The current implementation integrates transformer-based sentiment classification, semantic retrieval, FAISS vector search, Gemini-powered response generation, and an interactive Streamlit interface.

## Author

**Umm E Rubab**

**Live Demo:** https://sentibott.streamlit.app/

**GitHub:** https://github.com/UmmeRubab/SentiBot_A_Social_Media_Sentiment_Analysis_Chatbot_using_RAG_and_LLMs
