import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def run_ingestion(pdf_filename: str, max_pages: int = 15):
    raw_dir = os.path.join("data", "raw")
    vectorstore_dir = os.path.join("data", "vectorstore")
    pdf_path = os.path.join(raw_dir, pdf_filename)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF file at {pdf_path}")

    print(
        f"[1/4] Loading '{pdf_filename}' (Limiting to first {max_pages} pages for speed)..."
    )

    loader = PyPDFLoader(pdf_path)
    all_pages = loader.load()
    pages = all_pages[:max_pages]
    print(f"      Loaded {len(pages)} pages successfully.")

    print("[2/4] Chunking document into semantic text segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(pages)
    print(f"      Created {len(chunks)} chunks.")

    print("[3/4] Generating local HuggingFace vector embeddings (Free)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[4/4] Building and saving FAISS index locally...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(os.path.join(vectorstore_dir, "faiss_index"))

    print(
        "\n✅ Ingestion Complete! Vector store saved at 'data/vectorstore/faiss_index'"
    )


if __name__ == "__main__":
    raw_files = [
        f
        for f in os.listdir(os.path.join("data", "raw"))
        if f.endswith(".pdf")
    ]

    if not raw_files:
        print("❌ No PDF found in data/raw/. Please place a PDF there first.")
    else:
        target_pdf = raw_files[0]
        print(f"Found PDF: {target_pdf}")
        run_ingestion(pdf_filename=target_pdf, max_pages=15)