import os
import streamlit as st
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

# Streamlit Page Configuration
st.set_page_config(
    page_title="TN Farmers Welfare - Knowledge Graph RAG",
    page_icon="🌾",
    layout="wide"
)

# App Header
st.title("🌾 Tamil Nadu Agricultural Schemes Knowledge Graph Assistant")
st.markdown("Ask questions about TN government agricultural schemes, subsidies, and equipment. Powered by **Neo4j AuraDB**, **LangChain**, and **OpenAI (gpt-4o-mini)**.")

# Validate credentials
if not all([URI, PASSWORD, OPENAI_API_KEY]):
    st.error("⚠️ Missing required environment variables in your `.env` file (`NEO4J_URI`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`).")
    st.stop()

# Initialize Neo4j Driver & ChatOpenAI (Cached to prevent re-connecting on every click)


@st.cache_resource
def init_neo4j_driver():
    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))


@st.cache_resource
def init_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


driver = init_neo4j_driver()
llm = init_llm()

# Guardrail Check Function


def simple_guardrail_check(user_query):
    # Layer 1: Keyword check
    banned_keywords = ["python code", "write a script",
                       "movie", "cricket", "weather", "recipe", "joke", "coding"]
    query_lower = user_query.lower()
    for kw in banned_keywords:
        if kw in query_lower:
            return False, f"Blocked: Restricted keyword detected ('{kw}')."

    # Layer 2: LLM semantic validation
    guardrail_prompt = (
        "You are a strict guardrail classifier. Determine if the user query "
        "is related to Tamil Nadu agriculture, farming, crops, soil, or government agricultural schemes. "
        "Answer with ONLY 'YES' if relevant, or 'NO' if off-topic.\n\n"
        f"User Query: {user_query}"
    )

    response = llm.invoke(
        [HumanMessage(content=guardrail_prompt)]).content.strip().upper()
    if "NO" in response:
        return False, "Blocked: Query falls outside Tamil Nadu agricultural schemes domain."

    return True, "Passed"

# Graph Context Retrieval Function


def query_knowledge_graph(question):
    cypher_query = """
    MATCH (s:Scheme)
    OPTIONAL MATCH (s)-[r]-(connected)
    RETURN s.title AS scheme, type(r) AS relation, connected.name AS target_entity
    LIMIT 30
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


# Sidebar Settings & Diagnostics
with st.sidebar:
    st.header("🛠️ System Status")
    st.success("Connected to Neo4j AuraDB")
    st.info("Guardrail: Active (TN Agriculture Scope)")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to help you explore Tamil Nadu agricultural schemes. What would you like to know?"}
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if user_input := st.chat_input("Ask about schemes, subsidies, or equipment..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Checking safety guardrails & querying Knowledge Graph..."):

            # 1. Guardrail Check
            is_safe, guard_message = simple_guardrail_check(user_input)

            if not is_safe:
                response_text = f"🛡️ **[Guardrail Triggered]**: I am specialized strictly in Tamil Nadu agricultural schemes and cannot answer questions outside this topic. *({guard_message})*"
                st.markdown(response_text)
            else:
                # 2. Retrieve Graph Context
                graph_context = query_knowledge_graph(user_input)
                context_str = "\n".join(graph_context[:25])

                # 3. Generate RAG Response
                system_prompt = (
                    "You are an intelligent assistant specialized in Tamil Nadu agricultural schemes. "
                    "Answer the user's question accurately using ONLY the provided Knowledge Graph context below.\n\n"
                    f"Knowledge Graph Context:\n{context_str}"
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input)
                ]

                response = llm.invoke(messages)
                response_text = response.content

                st.markdown(response_text)

                # Optional: Expandable drawer to inspect raw graph context
                with st.expander("🔍 View Retrieved Knowledge Graph Context"):
                    st.code(context_str, language="text")

        # Save assistant response to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text})
