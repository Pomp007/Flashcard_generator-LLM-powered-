# Flashcard_generator-LLM-powered-
🧠 AI Flashcard Generator
Transform educational content into smart, structured flashcards using AI. Built with Streamlit, this app supports PDF/text input, OpenAI integration, and export to CSV, JSON, Anki, or Quizlet formats.

## 🖼️ App Screenshot

🚀 Features
✍️ Text & PDF Input
Paste content or upload files directly.

🤖 AI Provider Integration
Use OpenAI (via API key) or a mock provider for testing.

🧠 Content-Aware Flashcard Generation
Detects sections and produces flashcards by topic and difficulty.

🎯 Flashcard Quality Filtering
Filters out generic or low-quality questions.

📤 Export Options
Export your flashcards as CSV, JSON, Anki, or Quizlet-compatible formats.

🧰 Requirements
Python 3.8+

pip install -r requirements.txt

📦 Installation
bash
git clone https://github.com/Pomp007/Flashcard_generator-LLM-powered-.git
pip install -r requirements.txt
🔑 OpenAI API Key (Optional)
To use OpenAI-powered generation:

Get your API key from platform.openai.com.

Paste it into the sidebar field inside the app.

▶️ Run the App
bash
streamlit run flashcard_generator.py
🧪 File Support
.pdf – Extracts text using PDF processing.

.txt – Cleaned and parsed directly.

Section headers are auto-detected for better context-aware generation.

📤 Export Formats
Format	Description
CSV	Easy to open in Excel/Sheets
JSON	Structured, for developers
Anki	Anki-compatible plain text export
Quizlet	Ready for bulk import to Quizlet

🧪 Testing
A MockProvider is included for local testing without using tokens. You can extend this for testing other LLMs.

🧱 Folder Structure
bash
Copy
Edit
📁 ai-flashcard-generator/
│
├── flashcard_generator.py     # Main Streamlit app
├── providers/                 # AI provider modules
├── processor/                 # Content extraction/cleaning
├── flashcard.py               # Flashcard dataclass
├── requirements.txt
└── README.md
✅ TODO / Future Ideas
🗂️ Tag-based flashcard filtering

🌐 Hugging Face LLM support

📚 Save session history

🔄 Regenerate single card

📈 Flashcard difficulty analysis graphs
