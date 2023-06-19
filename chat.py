from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


# declare sentence transformer embeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open("data/devotionals_raw.txt") as f:
    devotional = f.read()

# split text into sentences
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=1)
texts = text_splitter.split_text(devotional)

# search through embeddings
docsearch = FAISS.from_texts(
    texts, embeddings
)

while True:
    query = input("Enter a query: ")
    docs = docsearch.similarity_search(query, k=1)

    for doc in docs:
        print(doc)




