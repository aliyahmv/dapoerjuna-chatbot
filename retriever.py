import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tools import df, build_block

load_dotenv()

# Ubah nama direktori untuk menghindari konflik dengan file FAISS yang lama
PERSIST_DIR = "database/chroma_recipes_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def load_retriever():
    # Validasi eksistensi direktori dan isinya
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading ChromaDB index dari penyimpanan lokal...")
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )
    else:
        print("Membuat ChromaDB index (11k dokumen)...")
        print("Proses ini akan memakan waktu dan komputasi memori.")

        docs = [
            Document(
                page_content=build_block(r),
                metadata={"loves": int(r.loves)}  
            )
            for r in df.itertuples(index=False)
        ]

        # Chroma.from_documents secara otomatis akan melakukan persist (penyimpanan) 
        # ke dalam PERSIST_DIR menggunakan basis SQLite, bukan pickle.
        vs = Chroma.from_documents(
            documents=docs, 
            embedding=embedding_model,
            persist_directory=PERSIST_DIR
        )

        print("ChromaDB selesai dibuat dan persisten!")

    return vs.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()