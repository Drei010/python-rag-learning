from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
import pandas as pd


EXCEL_PATH = Path(__file__).parent / "data" / "internshiplist.xlsx"
DB_LOCATION = Path("./chrome_langchain_db")

df = pd.read_excel(EXCEL_PATH)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="internshiplist",
    embedding_function=embeddings,
    persist_directory=str(DB_LOCATION),
)

documents = []
ids = []

for i, row in df.dropna(how="all").iterrows():
    fields = []
    metadata = {"source": EXCEL_PATH.name, "row": int(i)}

    for column, value in row.items():
        if pd.isna(value):
            continue

        text = str(value).strip()
        if not text:
            continue

        fields.append(f"{column}: {text}")

    if fields:
        document = Document(
            page_content="\n".join(fields),
            metadata=metadata,
            id=str(i),
        )
        ids.append(str(i))
        documents.append(document)

if vector_store._collection.count() == 0 and documents:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
