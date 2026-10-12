import os
import json
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file.")

extracted_graphs_folder = "extracted_graph_data"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


def build_networkx_graph():
    """Builds an in-memory NetworkX directed graph from local JSON extraction files."""
    G = nx.DiGraph()

    if not os.path.exists(extracted_graphs_folder):
        print(
            f"Directory '{extracted_graphs_folder}' not found! Run your extractor first.")
        return G

    json_files = [f for f in os.listdir(
        extracted_graphs_folder) if f.endswith('.json')]
    print(f"Loading {len(json_files)} extracted JSON files into NetworkX...")

    for filename in json_files:
        file_path = os.path.join(extracted_graphs_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)

                # Add Nodes
                for node in data.get("nodes", []):
                    name = node.get("name")
                    label = node.get("label", "Entity")
                    if name:
                        G.add_node(name, label=label)

                # Add Edges (Relationships)
                for rel in data.get("relationships", []):
                    source = rel.get("source")
                    target = rel.get("target")
                    relation = rel.get("relation", "RELATED_TO")
                    if source and target:
                        G.add_edge(source, target, relation=relation)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(
        f"Graph built successfully! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")
    return G


def query_networkx_graph(G, question):
    """
    Retrieves relevant subgraphs or connections from NetworkX 
    based on keyword matching between the user query and node names.
    """
    query_tokens = set(question.lower().split())
    relevant_context = []

    # Search for nodes that match query words
    for node, attrs in G.nodes(data=True):
        node_lower = node.lower()
        # Check if any query token matches parts of the node name
        if any(token in node_lower for token in query_tokens if len(token) > 3):
            # Fetch neighbors and edges
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                rel = edge_data.get(
                    "relation", "RELATED_TO") if edge_data else "RELATED_TO"
                relevant_context.append(f"'{node}' [{rel}] '{neighbor}'")

            # Also check incoming edges
            for predecessor in G.predecessors(node):
                edge_data = G.get_edge_data(predecessor, node)
                rel = edge_data.get(
                    "relation", "RELATED_TO") if edge_data else "RELATED_TO"
                relevant_context.append(f"'{predecessor}' [{rel}] '{node}'")

    # If no specific keyword match is found, fallback to general sample graph structure
    if not relevant_context:
        for u, v, data in list(G.edges(data=True))[:20]:
            rel = data.get("relation", "RELATED_TO")
            relevant_context.append(f"'{u}' [{rel}] '{v}'")

    return list(set(relevant_context))


def run_networkx_rag(user_question):
    print(f"\n🔍 User Query: {user_question}")

    # 1. Initialize NetworkX Graph from local cache
    G = build_networkx_graph()

    # 2. Retrieve Local Context
    graph_context = query_networkx_graph(G, user_question)
    context_str = "\n".join(graph_context[:30])

    # 3. Generate Answer using ChatOpenAI
    system_prompt = (
        "You are an intelligent assistant specialized in Tamil Nadu agricultural schemes. "
        "Answer the user's question accurately using ONLY the provided Knowledge Graph context below.\n\n"
        f"Knowledge Graph Context:\n{context_str}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]

    print("🤖 Generating answer via ChatOpenAI using NetworkX context...")
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    sample_question = "What equipment or subsidies are available for plant protection?"
    answer = run_networkx_rag(sample_question)

    print("\n--- NetworkX GraphRAG Response ---")
    print(answer)
