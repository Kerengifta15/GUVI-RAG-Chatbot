# app/extract_guvi_data.py
import os, time, re, requests
from bs4 import BeautifulSoup

# pages you want to collect
URLS = [
    "https://www.guvi.in/faq",
    "https://www.guvi.in/about",
    "https://www.guvi.in/blog/",
]

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """Remove extra spaces and URLs."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fetch_and_save(url):
    print(f"Fetching {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    # remove script/style content
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = clean_text(soup.get_text(separator=" "))
    name = url.replace("https://", "").replace("/", "_").replace(".", "_")
    filepath = os.path.join(SAVE_DIR, f"{name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved {filepath}")

def main():
    for url in URLS:
        fetch_and_save(url)
        time.sleep(2)   # be polite

if __name__ == "__main__":
    main()
