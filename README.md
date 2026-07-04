
# Nutrition ChatBot

A Streamlit-based RAG chatbot that answers questions about nutrition/diet content using Google Generative AI embeddings and a Groq LLM. Spun out from the [QaChatBot](https://github.com/ofi5/QaChatBot) FIFA demo.

## Project Structure

```
NutritionChatBot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── data/               # Data directory for PDF files
│   └── nutritiondata.pdf  # Add your nutrition/diet PDF here (not included yet)
├── .env                # Environment variables (create this file)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Status

This is currently a scaffold: the app expects a PDF at `data/nutritiondata.pdf`, which is **not included yet**. Drop in a real nutrition/diet-plan PDF before running the app.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Keys:**

   **For Local Development:**
   Create a `.env` file in the root directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   **For Website Deployment (Streamlit Cloud):**
   Use Streamlit secrets. In your Streamlit Cloud dashboard:
   - Go to your app settings
   - Navigate to "Secrets"
   - Add your API keys in this format:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

3. **Add Your Nutrition PDF:**
   Place a nutrition/diet-plan PDF at `data/nutritiondata.pdf`.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Features

- PDF document processing and text extraction
- Vector embeddings using Google Generative AI
- Question answering using Groq LLM
- Document similarity search
- Streamlit web interface
