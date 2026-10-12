import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([URI, PASSWORD, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables in .env file.")

# Initialize Neo4j and ChatOpenAI
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


def simple_guardrail_check(user_query):
    """
    Guardrail function to filter out off-topic queries.
    Returns: (is_safe: bool, reason: str)
    """
    # Layer 1: Simple keyword blocklist for obviously unrelated queries
    banned_keywords = ["python code", "write a script",
                       "movie", "cricket", "weather", "recipe", "joke"]
    query_lower = user_query.lower()
    for kw in banned_keywords:
        if kw in query_lower:
            return False, f"Blocked by guardrail: Detected restricted topic keyword ('{kw}')."

    # Layer 2: LLM semantic validation check
    guardrail_prompt = (
        "You are a strict guardrail classifier. Your job is to determine if a user query "
        "is related to Tamil Nadu agriculture, farming, crops, soil, or government agricultural schemes. "
        "Answer with ONLY 'YES' if it is relevant, or 'NO' if it is completely off-topic.\n\n"
        f"User Query: {user_query}"
    )

    response = llm.invoke(
        [HumanMessage(content=guardrail_prompt)]).content.strip().upper()

    if "NO" in response:
        return False, "Blocked by guardrail: Query is outside the scope of Tamil Nadu agricultural schemes."

    return True, "Passed"


def query_knowledge_graph(question):
    """Retrieve relevant context from Neo4j AuraDB using Cypher."""
    cypher_query = """
    MATCH (s:Scheme)
    OPTIONAL MATCH (s)-[r]-(connected)
    RETURN s.title AS scheme, type(r) AS relation, connected.name AS target_entity
    LIMIT 25
    """
    context_data = []
    with driver.session() as session:
        result = session.run(cypher_query)
        for record in result:
            scheme = record["scheme"]
            rel = record["relation"]
            target = record["target_entity"]
            if rel and target:
                context_data.append(f"Scheme '{scheme}' {rel} '{target}'")
            else:
                context_data.append(f"Scheme: {scheme}")
    return list(set(context_data))


def run_guarded_rag(user_question):
    print(f"\nUser Query: '{user_question}'")

    # Run Guardrail Check
    is_safe, message = simple_guardrail_check(user_question)
    if not is_safe:
        print(f"🛡️ [Guardrail Triggered]: {message}")
        print("🤖 Assistant Response: I am specialized strictly in Tamil Nadu agricultural schemes. I cannot answer queries outside this domain.")
        return

    print("🛡️ [Guardrail Status]: PASSED. Proceeding to GraphRAG...")

    # Execute GraphRAG pipeline if safe
    graph_context = query_knowledge_graph(user_question)
    context_str = "\n".join(graph_context[:30])

    system_prompt = (
        "You are an intelligent assistant specialized in Tamil Nadu agricultural schemes. "
        "Answer the user's question accurately using ONLY the provided Knowledge Graph context below.\n\n"
        f"Knowledge Graph Context:\n{context_str}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]

    response = llm.invoke(messages)
    print(f"🤖 Assistant Response:\n{response.content}")


if __name__ == "__main__":
    # Test 1: Valid agricultural query (Should pass)
    run_guarded_rag(
        "What are the available agriculture schemes and subsidies?")

    # Test 2: Off-topic query (Should be caught and blocked by guardrail)
    run_guarded_rag("Can you write a python script to sort a list?")

    driver.close()
