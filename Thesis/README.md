# Mental Health Chatbot

A professional-grade chatbot designed to answer mental health questions using Retrieval-Augmented Generation (RAG) and LangChain. This project leverages state-of-the-art language models and custom data sources to provide accurate, empathetic, and contextually relevant responses to users seeking mental health support.

## Features
- **Retrieval-Augmented Generation (RAG):** Combines generative AI with document retrieval for informed answers.
- **LangChain Integration:** Utilizes LangChain for flexible chaining of language model and retrieval components.
- **Custom Data Sources:** Answers are grounded in curated mental health resources and documents.
- **Question Suggestion:** Proactively suggests relevant questions to users.
- **MongoDB Support:** Stores user interactions and retrieved documents for improved context and analytics.
- **Extensible Architecture:** Modular design for easy integration of new models, data sources, or features.

## Project Structure
```
app/
    answer_generator.py      # Generates answers using RAG and LangChain
    app.py                  # Main application entry point
    configuration.py        # Configuration management
    data_processor.py       # Data preprocessing and ingestion
    embedding_utils.py      # Embedding and vector search utilities
    main.py                 # Application runner
    model_loader.py         # Loads and manages language models
    mongo_manager.py        # MongoDB interface
    question_suggester.py   # Suggests questions to users
    search_engine.py        # Document retrieval engine
    utils.py                # Utility functions
```

## Setup Instructions
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/ttdat1712/Ai-mental-health-chatbot.git
   cd Ai-mental-health-chatbot
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Edit `app/configuration.py` to set up model paths, database URIs, and other settings.
4. **Prepare data:**
   - Place your mental health documents in the `data/` folder.
   - Run `app/data_processor.py` to ingest and preprocess data.
5. **Run the chatbot:**
   ```powershell
   python app/app.py
   ```

## Usage

Interact with the chatbot via the web interface (Streamlit app).
Ask mental health-related questions and receive informed, empathetic answers.
Use suggested questions for guided conversations.

## Web Application Details
This chatbot features a modern, interactive web interface built with Streamlit:

- **User Experience:**
  - Clean chat UI with avatars, message bubbles, and a fixed input bar.
  - Responsive design and custom CSS for accessibility and aesthetics.
  - Sidebar for conversation history, model selection, and usage tips.
- **Session Management:**
  - Each session tracks the selected model, conversation history, and current messages.
  - Users can start new chats or delete all history with one click.
- **Model Switching:**
  - Easily switch between available AI models from the sidebar dropdown.
  - Models are loaded/unloaded dynamically for efficient resource use.
- **MongoDB Integration:**
  - All conversations are stored and retrieved from MongoDB.
  - View, select, and manage chat history directly in the sidebar.
- **Accessibility & Guidance:**
  - Keyboard and click handlers for suggested questions.
  - Sidebar tips for effective mental health queries.
- **Disclaimer:**
  - A visible disclaimer reminds users to verify information and seek professional help for urgent concerns.


1. User submits a question via the web interface.
2. The app retrieves relevant documents using BM25 and embeddings.
3. LangChain combines retrieval and generation to produce an answer.
4. Conversation is stored in MongoDB for future reference.
5. User can view, select, or delete past conversations from the sidebar.

## Technologies Used
To launch the chatbot web interface, use:
```powershell
streamlit run app/app.py
```
This will start the Streamlit server and open the chatbot in your browser.

- **Python 3.11+**
- **Add new models:** Update `AVAILABLE_MODELS` in `app/configuration.py` and provide model files.
- **Change UI themes:** Edit the CSS in `app/app.py` for colors, layout, and style.
- **Expand data sources:** Place new documents in the `data/` folder and run the data processor.

- **LangChain**
- **Frontend:** Streamlit web app with custom CSS/JS for enhanced UI/UX.
- **Backend:** Python modules for model loading, data processing, retrieval, and answer generation.
- **Database:** MongoDB for persistent conversation storage.
- **AI Models:** RAG pipeline with LangChain, BM25, and embeddings.

- **MongoDB**
**Python 3.11+**
**LangChain**
**MongoDB**
**RAG (Retrieval-Augmented Generation)**
- **RAG (Retrieval-Augmented Generation)**

