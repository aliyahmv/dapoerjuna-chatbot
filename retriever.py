import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.schema import Document
from tools import df, build_block

load_dotenv()

INDEX_DIR = "database/faiss_recipes_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def load_retriever():

    # Kalau index sudah ada → load
    if os.path.exists(INDEX_DIR):
        print("Loading FAISS index...")
        vs = FAISS.load_local(
            INDEX_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:
        print("Membuat FAISS index (11k dokumen)...")

        docs = [
            Document(
                page_content=build_block(r),
                metadata={"loves": int(r.loves)}  # pastikan huruf besar
            )
            for r in df.itertuples(index=False)
        ]

        vs = FAISS.from_documents(docs, embedding_model)
        vs.save_local(INDEX_DIR)

        print("FAISS selesai dibuat!")

    return vs.as_retriever(search_kwargs={"k": 4})


retriever = load_retriever()
