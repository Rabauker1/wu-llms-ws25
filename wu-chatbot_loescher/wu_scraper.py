import argparse
import json
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

# ------------- CONFIG ------------- #

DEFAULT_START_URLS = [
    "https://www.wu.ac.at/en/",
]

ALLOWED_DOMAINS = [
    "wu.ac.at",
    "www.wu.ac.at",
]

USER_AGENT = "User Agent"
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 1.0 

def is_allowed_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ALLOWED_DOMAINS


def normalize_url(base_url: str, link: str) -> str | None:
    """
    - Resolve relative URLs
    - Remove URL fragments (#section)
    - Filter non-http(s)
    """
    if not link:
        return None
    url = urljoin(base_url, link)
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None
    if not is_allowed_domain(url):
        return None

    return url


def setup_robots(start_url: str) -> robotparser.RobotFileParser:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt can't be read, be extra careful or abort.
        print(f"[WARN] Could not read robots.txt from {robots_url}.")
    return rp


def extract_text(html: str) -> tuple[str, str]:
    """
    Extract (title, main_text) from HTML.
    Very simple heuristic-based extraction.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    # Remove common layout elements
    for selector in [".cookie", ".nav", ".footer", ".header", ".banner"]:
        for tag in soup.select(selector):
            tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")

    text = soup.get_text(separator=" ", strip=True)
    # Optionally shorten very long pages 
    # max_len = 100_000
    # if len(text) > max_len:
    #     text = text[:max_len]

    return title, text


def fetch(url: str, session: requests.Session) -> str | None:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"[WARN] {url} -> HTTP {resp.status_code}")
            return None
        return resp.text
    except Exception as e:
        print(f"[ERROR] fetching {url}: {e}")
        return None


def crawl_wu(
    start_urls: list[str],
    max_pages: int,
    output_path: str,
    max_depth: int = 10,
):
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for url in start_urls:
        queue.append((url, 0))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Use robots.txt of the first start URL
    robots = setup_robots(start_urls[0])

    with open(output_path, "w", encoding="utf-8") as f_out:
        pages_crawled = 0

        while queue and pages_crawled < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if depth > max_depth:
                continue

            # Check robots.txt
            if robots and not robots.can_fetch(USER_AGENT, url):
                print(f"[SKIP] Disallowed by robots.txt: {url}")
                continue

            print(f"[CRAWL] ({pages_crawled+1}/{max_pages}) depth={depth} {url}")

            html = fetch(url, session)
            if html is None:
                continue

            try:
                title, text = extract_text(html)
            except Exception as e:
                print(f"[ERROR] parsing HTML at {url}: {e}")
                # Skip this page and continue with the crawl
                continue

            # Skip pages with almost no content
            if len(text) < 100:
                print(f"[SKIP] Too little text at {url}")
            else:
                doc = {
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "wu.ac.at",
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                f_out.flush()
                pages_crawled += 1

            # Parse links and enqueue (fail-safe: skip problematic links/pages)
            try:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    try:
                        next_url = normalize_url(url, a["href"])
                        if not next_url:
                            continue
                        if next_url in visited:
                            continue
                        queue.append((next_url, depth + 1))
                    except Exception as e:
                        print(f"[WARN] failed to normalize/enqueue link from {url}: {e}")
                        # Skip this link but continue processing other links
                        continue
            except Exception as e:
                print(f"[WARN] failed to parse links in {url}: {e}")
                # Continue crawling other queued URLs
                pass

            time.sleep(REQUEST_DELAY)

    print(f"[DONE] Crawled {pages_crawled} pages. Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WU Vienna web scraper for RAG chatbot.")
    parser.add_argument(
        "--start",
        nargs="*",
        default=DEFAULT_START_URLS,
        help="Start URLs (default: https://www.wu.ac.at/en/)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10_000,
        help="Maximum number of pages to crawl (default: 10,000)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum crawl depth from start URLs (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wu_docs_v1.jsonl",
        help="Output JSONL file path (default: wu_docs_v1.jsonl)",
    )
    args = parser.parse_args()

    crawl_wu(
        start_urls=args.start,
        max_pages=args.max_pages,
        output_path=args.output,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()