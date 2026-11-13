import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

REQUIRED_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)

missing = [name for name in REQUIRED_VARS if not os.getenv(name)]
if missing:
    sys.exit(f"Variáveis de ambiente ausentes no .env: {', '.join(missing)}")

PDF_PATH = os.getenv("PDF_PATH")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def ingest_pdf():
    if not os.path.isfile(PDF_PATH):
        sys.exit(f"PDF não encontrado: {os.path.abspath(PDF_PATH)}")

    docs = PyPDFLoader(PDF_PATH).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        sys.exit("Nenhum texto extraído do PDF.")

    # Remove metadados vazios para não gravar lixo no JSONB.
    chunks = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    # IDs estáveis: rodar a ingestão de novo sobrescreve em vez de duplicar.
    ids = [f"doc-{i}" for i in range(len(chunks))]

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    store.add_documents(documents=chunks, ids=ids)
    print(
        f"{len(chunks)} chunks de {len(docs)} páginas salvos "
        f"na coleção '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    ingest_pdf()
