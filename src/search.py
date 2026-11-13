import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

REQUIRED_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)

# Modelo leve e barato que aceita temperature=0. Os modelos gpt-5* não servem
# aqui: o langchain-openai 0.3.x descarta temperature != 1 para eles.
LLM_MODEL = "gpt-4o-mini"

TOP_K = 10


def search_prompt(question=None):
    try:
        missing = [name for name in REQUIRED_VARS if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                f"Variáveis de ambiente ausentes no .env: {', '.join(missing)}"
            )

        store = PGVector(
            embeddings=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL")),
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    def ask(pergunta: str) -> str:
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        contexto = "\n\n".join(doc.page_content for doc, _ in results)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
        return llm.invoke(prompt).content

    if question is not None:
        return ask(question)
    return ask
