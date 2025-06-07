import chromadb
import pypdf
import requests

chromadb_client = chromadb.Client()
collection = chromadb_client.get_or_create_collection("pdfs")

def upload_pdf(file_path):
    """
    Uploads a PDF file to the ChromaDB collection.
    """
    with open(file_path, "rb") as f:
        pdf = pypdf.PdfReader(f)
        text = ""
        id = 0
        for page in pdf.pages:
            text += page.extract_text()
        collection.add(
           documents=[page.extract_text()],
           ids = [f"{file_path}_{id}"],
        )
      

upload_pdf("FAQ_for_Chatbot.pdf")
 
query = input("Enter your query: ")
closetPages = collection.query(
    query_texts=[query],
    n_results=5
)

api_key = "sk-titCpkply6rFBcc6326yqTJzL20JeJJ9ACekZEij4nNCpClA"
api_url = "https://api.opentyphoon.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "typhoon-v2-70b-instruct",
    "messages": [
        {"role": "system", "content": closetPages['documents'][0][0]},
        {"role": "user", "content": query}
    ]
}

response = requests.post(api_url, headers=headers, json=payload).json()
print(closetPages)
print("Response:", response["choices"][0]["message"]["content"])

