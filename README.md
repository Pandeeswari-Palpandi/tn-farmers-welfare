# tn-farmers-welfare

Tools for exploring Tamil Nadu Agriculture and Farmers Welfare Department schemes
using a Neo4j knowledge graph and language-model-based question answering.

## Setup

Use a supported Python version and create a virtual environment from the project
root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies required by the script you plan to run. The graph query
and guarded graph query scripts use Neo4j, LangChain OpenAI integration, and
python-dotenv:

```powershell
python -m pip install neo4j langchain-openai langchain-core python-dotenv
```

Create a `.env` file in the project root with your own credentials:

```dotenv
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
OPENAI_API_KEY=your_openai_api_key
```

Do not commit `.env` or share its credentials.

## Running the scripts

- `graph-loader.py` extracts scheme information and stores it in Neo4j.
- `graph-query.py` queries the knowledge graph and generates answers.
- `guarded_graph_query.py` checks whether a question is in scope before querying
  the graph.
- `networkx_rag.py` contains an alternative guarded RAG implementation.

Run a script from the project root after setting up its dependencies and `.env`,
for example:

```powershell
python graph-query.py
```

## Windows installation note for NeMo Guardrails

The NeMo Guardrails package depends on `annoy`, which may need to compile from
source on Windows. If `pip install nemoguardrails` fails with
`Microsoft Visual C++ 14.0 or greater is required`, install **Build Tools for
Visual Studio** and select **Desktop development with C++**, including the MSVC
C++ build tools and a Windows SDK. Restart the terminal and retry:

```powershell
python -m pip install nemoguardrails
```

If you do not want to install the compiler, try a Python version for which pip
can find a compatible prebuilt `annoy` wheel. Availability depends on the Python
version and platform; using a different version is not guaranteed to avoid a
source build.
