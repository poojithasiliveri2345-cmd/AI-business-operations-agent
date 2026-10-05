import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#loading all the pdf from the folder
all_docs = []
for file in os.listdir("rag_docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader("rag_docs/" + file)
        all_docs = all_docs + loader.load()
print("total pages:", len(all_docs))

#splitting into chunks of 500 characters
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print("total chunks:", len(chunks))

#embeddings + faiss
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

#testings some queries
questions = [
    "what is the refund policy for furniture?",
    "why might my order be delayed?",
    "what approval id needed for a 40% discount?",
    "when is an order escalated to Tier 2?",
    "Is Same Day shipping available in Africa?"
]
for q in questions:
    print("Quetions:", q)
    results = db.similarity_search(q, k=2)
    for r in results:
        print(r.page_content)
        print("---")
    print("===========")
