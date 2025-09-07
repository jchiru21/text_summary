# Text Summary (Abstractive Summarizer)

A Flask web application that performs **abstractive text summarization** using the `T5` transformer model.  
It supports:
- Summarization of raw text or uploaded PDF documents
- Translation of summaries into multiple languages (via Google Translate)
- Simple UI with Bootstrap
- Maintains a short history of previous summaries

---

## 🚀 Features
- **Abstractive Summarization**: Powered by Hugging Face `transformers` (T5-small by default, can switch to T5-base).
- **PDF Support**: Extracts text from uploaded PDFs (text-based only).
- **Multi-language Translation**: Translate summaries into any supported language code.
- **Lightweight UI**: Bootstrap-styled input form and output display.


---

## 🛠️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/text_summary.git
cd text_summary
````

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

👉 For CPU-only users, replace Torch with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Run the app

```bash
python app.py
```

Visit: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

