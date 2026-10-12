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


def query_knowledge_graph(question):
    """
    Step A: Retrieve relevant context from Neo4j AuraDB.
    We search for schemes or entities matching keywords in the user question,
    then fetch their direct relationships.
    """
    # Simple keyword-based or text-match retrieval from the graph
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

    return list(set(context_data))  # Remove duplicates


def answer_with_graph_rag(user_question):
    """
    Step B: Feed the graph context into the LLM for RAG generation.
    """
    print(f"\n🔍 User Question: {user_question}")
    print(" Retrieving context from Neo4j Knowledge Graph...")

    graph_context = query_knowledge_graph(user_question)
    context_str = "\n".join(graph_context[:30])  # Limit context window size

    system_prompt = (
        "You are an intelligent assistant specialized in Tamil Nadu agricultural schemes. "
        "Answer the user's question accurately using ONLY the provided Knowledge Graph context below.\n\n"
        f"Knowledge Graph Context:\n{context_str}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]

    print(" Generating answer via ChatOpenAI...")
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    # Test your GraphRAG query
    sample_question = "What are some available schemes and what equipment or subsidies do they offer?"

    answer = answer_with_graph_rag(sample_question)

    print("\n--- GraphRAG Response ---")
    print(answer)

    driver.close()
