import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([URI, PASSWORD, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables in .env file.")

input_folder = "tn_schemes_pure_content"
extracted_graphs_folder = "extracted_graph_data"
main_url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
domain = "https://www.tn.gov.in/"

# Ensure folders exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(extracted_graphs_folder, exist_ok=True)

# Initialize ChatOpenAI and Neo4j Driver
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def sanitize_filename(name):
    """Remove unsafe characters for file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def sanitize_rel_type(rel_name):
    """Ensure relationship names are safe uppercase Neo4j types."""
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel_name).upper()
    return clean if clean else "RELATED_TO"


def run_data_loader():
    """Scrapes scheme lists and inner details if text files are missing."""
    print(f"Data not found locally. Running Data Loader from: {main_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(main_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching main page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    scheme_container = None
    for table in soup.find_all('table'):
        if table.find_all('a', href=lambda h: h and 'scheme_details' in h.lower()):
            scheme_container = table
            break
    if not scheme_container:
        scheme_container = soup

    links = scheme_container.find_all('a', href=True)
    scheme_items = []
    seen_urls = set()
    ignored_keywords = ['home', 'government', 'departments', 'districts',
                        'documents', 'services', 'news', 'interact', 'others', 'login']

    for link in links:
        title = link.get_text(strip=True)
        href = link['href']
        if title and len(title) > 3 and not href.startswith('#') and 'scheme' in href.lower() and not any(kw in title.lower() for kw in ignored_keywords):
            full_url = urljoin(domain, href)
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                scheme_items.append((title, full_url))

    print(
        f"Found {len(scheme_items)} valid scheme links. Downloading content...")

    for index, (title, url) in enumerate(scheme_items, start=1):
        safe_title = sanitize_filename(title) or f"scheme_{index}"
        file_path = os.path.join(
            input_folder, f"{index}_{safe_title[:50]}.txt")

        try:
            detail_res = requests.get(url, headers=headers, timeout=10)
            detail_res.raise_for_status()
            detail_soup = BeautifulSoup(detail_res.text, 'html.parser')

            detail_content = None
            for td in detail_soup.find_all(['td', 'div']):
                if (td.get('id') == 'content' or td.get('class') == ['content']) and len(td.get_text(strip=True)) > 50:
                    detail_content = td
                    break
            if not detail_content:
                detail_content = detail_soup.body

            for unwanted in detail_content.find_all(['header', 'footer', 'nav', 'script', 'style']):
                unwanted.decompose()

            scheme_text = detail_content.get_text(separator='\n', strip=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Scheme Title: {title}\n")
                f.write(f"Source URL: {url}\n")
                f.write("="*50 + "\n\n")
                f.write(scheme_text)
        except Exception as e:
            print(f" -> Failed to fetch details for {title}: {e}")
    print("Data loading completed successfully!\n")


def extract_entities_and_relationships(text):
    """Uses LangChain ChatOpenAI to extract nodes and structured relationships."""
    system_prompt = (
        "You are an expert knowledge graph builder. Analyze the agricultural scheme text "
        "and extract key entities and relationships. Allowed Node Labels: Scheme, Department, "
        "Beneficiary, Subsidy, Equipment, Criterion.\n\n"
        "Return ONLY a valid JSON object with keys:\n"
        "1. 'nodes': list of objects with keys 'name' and 'label'\n"
        "2. 'relationships': list of objects with keys 'source', 'target', and 'relation' (uppercase like TARGETS, OFFERS, REQUIRES)"
    )
    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=text)]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    return json.loads(response.content)


def process_and_store_graph():
    # Step 1: Check if raw text data is available locally
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(
        '.txt')] if os.path.exists(input_folder) else []

    if not txt_files:
        run_data_loader()
        txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    print(f"Processing {len(txt_files)} scheme files...")

    with driver.session() as session:
        for idx, filename in enumerate(txt_files, start=1):
            file_path = os.path.join(input_folder, filename)
            json_filename = filename.replace('.txt', '.json')
            json_file_path = os.path.join(
                extracted_graphs_folder, json_filename)

            print(f"[{idx}/{len(txt_files)}] Checking: {filename}")

            try:
                # Step 2: Check if entities and relationships are already extracted in JSON cache
                if os.path.exists(json_file_path):
                    print(f"   -> Cached JSON found. Skipping LLM Entity Extraction.")
                    with open(json_file_path, 'r', encoding='utf-8') as jf:
                        graph_data = json.load(jf)
                else:
                    print(
                        f"   -> No cache found. Calling ChatOpenAI for Entity Extraction...")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    graph_data = extract_entities_and_relationships(content)

                    # Save newly extracted data to JSON cache
                    with open(json_file_path, 'w', encoding='utf-8') as jf:
                        json.dump(graph_data, jf, indent=4, ensure_ascii=False)

                # Step 3: Load nodes and relationships into Neo4j
                for node in graph_data.get("nodes", []):
                    valid_labels = {
                        "Scheme", "Department", "Beneficiary", "Subsidy", "Equipment", "Criterion"}
                    node_label = node.get("label") if node.get(
                        "label") in valid_labels else "Entity"
                    query = f"MERGE (n:{node_label} {{name: $name}})"
                    session.run(query, name=node["name"])

                for rel in graph_data.get("relationships", []):
                    source = rel.get("source")
                    target = rel.get("target")
                    rel_type = sanitize_rel_type(
                        rel.get("relation", "RELATED_TO"))
                    if source and target:
                        query = f"""
                        MATCH (a {{name: $source}})
                        MATCH (b {{name: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        """
                        session.run(query, source=source, target=target)

                print(f"   -> Mapped into Neo4j successfully!")

            except Exception as e:
                print(f"   -> Error processing {filename}: {e}")

    driver.close()
    print("\n🎉 Pipeline execution completed successfully!")


if __name__ == "__main__":
    process_and_store_graph()
