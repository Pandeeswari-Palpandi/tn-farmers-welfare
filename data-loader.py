import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 1. Configuration
main_url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
domain = "https://www.tn.gov.in/"
output_folder = "tn_schemes_pure_content"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}


def sanitize_filename(name):
    """Remove unsafe characters for file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def scrape_schemes():
    print(f"Fetching main scheme list from: {main_url}")
    try:
        response = requests.get(main_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching main page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Specifically target tables or table cells that hold the main data list,
    # avoiding the global outer layout tables.
    # Usually, scheme items are listed inside table data cells (td) or lists.
    scheme_container = None

    # Look for tables containing scheme links directly
    tables = soup.find_all('table')
    for table in tables:
        # Check if this table has links pointing to scheme details
        if table.find_all('a', href=lambda h: h and 'scheme_details' in h.lower()):
            scheme_container = table
            break

    if not scheme_container:
        # Fallback to body if specific table isn't isolated, but we will filter links strictly
        scheme_container = soup

    # Extract all links safely from the isolated container
    links = scheme_container.find_all('a', href=True)

    scheme_items = []
    seen_urls = set()

    for link in links:
        title = link.get_text(strip=True)
        href = link['href']

        # Strict validation: must look like an individual scheme link
        # and avoid global navigation keywords like Home, Departments, Districts, etc.
        ignored_keywords = ['home', 'government', 'departments', 'districts',
                            'documents', 'services', 'news', 'interact', 'others', 'login']
        if (
            title
            and len(title) > 3
            and not href.startswith('#')
            and 'scheme' in href.lower()
            and not any(keyword in title.lower() for keyword in ignored_keywords)
        ):
            full_url = urljoin(domain, href)
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                scheme_items.append((title, full_url))

    print(
        f"Found {len(scheme_items)} valid scheme links. Fetching inside contents...")

    for index, (title, url) in enumerate(scheme_items, start=1):
        safe_title = sanitize_filename(title)
        if not safe_title:
            safe_title = f"scheme_{index}"

        file_name = f"{index}_{safe_title[:50]}.txt"
        file_path = os.path.join(output_folder, file_name)

        print(f"[{index}/{len(scheme_items)}] Fetching scheme: {title}")

        try:
            detail_res = requests.get(url, headers=headers, timeout=10)
            detail_res.raise_for_status()
            detail_soup = BeautifulSoup(detail_res.text, 'html.parser')

            # On detail pages, headers/footers are typically wrapped in specific layout rows.
            # We look for the main central content container block or cell.
            detail_content = None

            # Try to find a table cell or div that holds the actual article text
            for td in detail_soup.find_all(['td', 'div']):
                # Look for common content wrappers or attributes
                if td.get('id') == 'content' or td.get('class') == ['content'] or 'scheme' in td.get_text().lower():
                    # Ensure it's not the outer layout table by checking text length or depth
                    if len(td.get_text(strip=True)) > 50:
                        detail_content = td
                        break

            if not detail_content:
                detail_content = detail_soup.body

            # Strip any remaining standard template artifacts (nav, header, footer tags if present)
            for unwanted in detail_content.find_all(['header', 'footer', 'nav', 'script', 'style']):
                unwanted.decompose()

            scheme_text = detail_content.get_text(separator='\n', strip=True)

            # Save the clean scheme details into individual text files
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Scheme Title: {title}\n")
                f.write(f"Source URL: {url}\n")
                f.write("="*50 + "\n\n")
                f.write(scheme_text)

            print(f" -> Saved cleanly to {file_path}")

        except Exception as e:
            print(f" -> Failed to fetch details for {title}: {e}")


if __name__ == "__main__":
    scrape_schemes()
    print("All tasks completed successfully!")
