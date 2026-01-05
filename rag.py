from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

PDF_PATH = "data/EAB_6.Auflage.pdf"
DB_DIR = "chroma_db"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
vectordb.persist()

llm = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

def answer(q: str, k: int = 4) -> str:
    hits = vectordb.similarity_search(q, k=k)
    context = "\n\n".join(h.page_content for h in hits)

    prompt = (
        "Beantworte die Frage ausschließlich auf Basis des folgenden Kontexts.\n"
        "Wenn die Information nicht im Kontext steht, antworte exakt: "
        "\"Nicht in den Dokumenten enthalten.\"\n\n"
        f"KONTEXT:\n{context}\n\n"
        f"FRAGE:\n{q}\n\n"
        "ANTWORT:"
    )
    return llm(prompt)[0]["generated_text"]

print("RAG bereit. Frage eingeben oder 'exit'.")
while True:
    q = input("\nFrage: ").strip()
    if q.lower() == "exit":
        break
    print("\nAntwort:\n", answer(q))
