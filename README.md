# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio de ingestão e busca semântica do MBA em IA da Full Cycle. O enunciado original está em [devfullcycle/mba-ia-desafio-ingestao-busca](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).

O projeto faz duas coisas:

- **Ingestão:** lê o `document.pdf`, divide o texto em chunks, gera os embeddings e salva tudo no PostgreSQL com pgVector.
- **Busca:** um chat no terminal que responde perguntas usando **apenas** o conteúdo do PDF. Se a resposta não estiver no documento, ele responde *"Não tenho informações necessárias para responder sua pergunta."*

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da Empresa SuperTechIABrazil é R$ 10.000.000,00.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Pré-requisitos

- **Python 3.12 ou 3.13.** Evite o 3.14: algumas versões fixadas no `requirements.txt` (como `numpy`, `pydantic_core` e `asyncpg`) podem não ter pacotes prontos para ele, e o `pip install` tenta compilar do código-fonte.
- **Docker** com Docker Compose.
- **Uma API Key da OpenAI.** O projeto usa só a OpenAI, tanto para os embeddings quanto para a LLM.

## Como executar

Todos os comandos abaixo devem ser rodados na **raiz do repositório**.

### 1. Criar o ambiente virtual e instalar as dependências

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<details>
<summary>Não tem o Python 3.12 instalado? Use o <code>uv</code></summary>

```bash
uv python install 3.12
uv venv --python 3.12 --seed venv
source venv/bin/activate
pip install -r requirements.txt
```

</details>

### 2. Configurar o `.env`

```bash
cp .env.example .env
```

Preencha assim:

| Variável | Valor |
|---|---|
| `OPENAI_API_KEY` | sua chave da OpenAI |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` (já vem no exemplo) |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | qualquer nome, por exemplo `documento_pdf` |
| `PDF_PATH` | `document.pdf` |
| `GOOGLE_API_KEY`, `GOOGLE_EMBEDDING_MODEL` | podem ficar vazias, não são usadas |

O `+psycopg` no `DATABASE_URL` é obrigatório: o `langchain_postgres` usa o driver psycopg 3.

### 3. Subir o banco de dados

```bash
docker compose up -d
docker compose ps -a
```

O serviço `postgres` deve aparecer como **healthy** e o `bootstrap_vector_ext` deve sair com código 0, depois de criar a extensão `vector`.

### 4. Fazer a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
67 chunks de 34 páginas salvos na coleção 'documento_pdf'.
```

Rodar a ingestão de novo é seguro: cada chunk tem um ID fixo (`doc-0`, `doc-1`, ...), então os registros são sobrescritos em vez de duplicados.

### 5. Rodar o chat

```bash
python src/chat.py
```

Digite a pergunta depois de `PERGUNTA:`. Para sair, digite `sair` ou `exit`, ou use Ctrl+C / Ctrl+D.

## Decisões de implementação

### Modelos

| Uso | Modelo | Por quê |
|---|---|---|
| Embeddings | `text-embedding-3-small` | O modelo de embeddings mais barato da OpenAI, gera vetores de 1536 dimensões. Configurado pelo `.env`. |
| LLM | `gpt-4o-mini` | Modelo leve e barato que aceita `temperature=0`. Fica fixo em `src/search.py` (`LLM_MODEL`), para o projeto rodar com um `.env` criado a partir do `.env.example`, sem variáveis extras. |

Os modelos `gpt-5*` (como o `gpt-5.4-nano`) foram descartados de propósito: o `langchain-openai` fixado no `requirements.txt` (0.3.30) remove qualquer `temperature` diferente de 1 para modelos cujo nome começa com `gpt-5`. Na prática, o `temperature=0` seria ignorado sem aviso.

### Parâmetros

- **Chunks de 1000 caracteres com overlap de 150**, como pede o enunciado.
- **`k=10`** na busca por similaridade, como pede o enunciado. O PDF é uma tabela longa de empresas com nomes parecidos, e trazer 10 chunks aumenta a chance de a linha certa estar no contexto.
- **`temperature=0`** na LLM. As respostas ficam determinísticas, e a frase de recusa sai sempre exatamente igual, sem variações de texto.

### Estrutura

- `src/ingest.py` carrega o PDF com `PyPDFLoader`, divide com `RecursiveCharacterTextSplitter`, remove metadados vazios e salva os chunks no `PGVector` com IDs estáveis.
- `src/search.py`: `search_prompt()` monta o `PGVector` e o `ChatOpenAI`, e retorna uma função que vetoriza a pergunta, busca os 10 chunks mais próximos, monta o `PROMPT_TEMPLATE` com eles e devolve a resposta da LLM. Se a inicialização falhar, retorna `None`.
- `src/chat.py` é o loop do terminal. Uma pergunta com erro (de rede ou da API, por exemplo) mostra a mensagem e não derruba o chat.
- Os dois scripts validam as variáveis de ambiente no início e mostram quais estão faltando.

### Correção no `docker-compose.yml`

O `docker-compose.yml` original tinha um bug no serviço `bootstrap_vector_ext`: o `command` em bloco dobrado (`>`) era dividido em várias palavras pelo Compose, e o `sh -c` recebia só `PGPASSWORD=postgres` como script. O container saía com código 0 sem rodar o `psql`, e a extensão `vector` não era criada. O comando agora é passado como uma única string.

## Testes realizados

| Pergunta | Resposta |
|---|---|
| Qual o faturamento da Empresa SuperTechIABrazil? | R$ 10.000.000,00 |
| Qual o faturamento da Alfa Agronegócio Indústria? (início do PDF) | R$ 85.675.568,77 |
| Qual o faturamento da Nexum Fármacos Holding? (meio do PDF) | R$ 524.331.781,47 |
| Qual o faturamento da Ímpar Petróleo LTDA? (fim do PDF) | R$ 1.563.536.272,88 |
| Em que ano foi fundada a Ímpar Química LTDA? | 2017 |
| Qual o faturamento da Alfa Energia Holding? / da Alfa Energia S.A.? (nomes parecidos) | R$ 858.537,02 / R$ 722.875.391,46 |
| Qual é a capital da França? | Não tenho informações necessárias para responder sua pergunta. |
| Quantos clientes temos em 2024? | Não tenho informações necessárias para responder sua pergunta. |
| Você acha isso bom ou ruim? | Não tenho informações necessárias para responder sua pergunta. |
| Qual o faturamento da Zeta Robótica S.A.? (empresa que não existe no PDF) | Não tenho informações necessárias para responder sua pergunta. |

## Problemas comuns

- **`Bind for 127.0.0.1:5432 failed: port is already allocated`:** outro Postgres já está usando a porta 5432. Pare esse container ou serviço e rode `docker compose up -d --force-recreate`. Outra opção é trocar a porta do host no `docker-compose.yml` (por exemplo, `"5433:5432"`) e ajustar o `DATABASE_URL`.
- **Troca do modelo de embeddings:** a dimensão dos vetores é definida na primeira ingestão. Se você trocar de modelo, apague o banco com `docker compose down -v` e rode `docker compose up -d` e `python src/ingest.py` de novo.
- **`401 expired_secret_key` ou `invalid_api_key`:** a `OPENAI_API_KEY` está expirada ou errada. Gere uma nova chave e atualize o `.env`.
- **Chamadas à OpenAI muito lentas (minutos):** isso pode acontecer em redes com IPv6 quebrado, porque o Python tenta IPv6 primeiro e só cai para IPv4 depois do timeout. Teste com `curl -6 -m 10 https://api.openai.com`. Se travar, desative o IPv6 da interface de rede (no macOS: `networksetup -setv6off Wi-Fi`).
