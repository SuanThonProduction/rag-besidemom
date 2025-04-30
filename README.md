# 🧠 RAG with Typhoon API Integration

This project implements a Retrieval-Augmented Generation (RAG) system using the Typhoon AI API, `sentence-transformers` for embeddings, and PostgreSQL for document storage and retrieval.

## 🚀 Features

- Vector search using SentenceTransformer
- Integration with Typhoon LLM (e.g., `typhoon-v2-70b-instruct`)
- PostgreSQL for document indexing
- Simple and customizable RAG pipeline

---

## 🛠️ Setup Instructions

Follow these steps to get your environment ready:

### 1. Clone the Repository

```bash
git clone https://your-repo-url
cd your-project-directory
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

**On macOS/Linux:**

```bash
source .venv/bin/activate
```

**On Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install Required Dependencies

Make sure you have a `requirements.txt` file with your dependencies. Then run:

```bash
pip install -r requirements.txt
```

If you haven’t generated the `requirements.txt` yet, you can do so with:

```bash
pip freeze > requirements.txt
```

---

## 📦 Dependencies

The main dependencies (already in `requirements.txt`) include:

```text
sentence-transformers
psycopg2
ollama
requests
```

Add any others you use.

---

## 🧪 Running the App

Once everything is set up, simply run your main script:

```bash
python your_script.py
```

Replace `your_script.py` with your actual script name.

---

## 🧰 Notes

- Make sure your PostgreSQL server is running and accessible.
- Replace any placeholder values like API keys or model names in the script as needed.

---

## 📞 Support

For help, feel free to open an issue or contact the maintainer.


create docker

docker run -d --name pgvector-db -e POSTGRES_USER=myuser -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=mydb -p 5432:5432 pgvector/pgvector:pg17
