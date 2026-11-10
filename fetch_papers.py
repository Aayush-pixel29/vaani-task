import os
import httpx
import fitz  # PyMuPDF
import json

PAPERS = {
    "attention_1706.03762v7": "https://arxiv.org/pdf/1706.03762v7.pdf",
    "moshi_2410.00037v1": "https://arxiv.org/pdf/2410.00037v1.pdf",
    "dynamic_chunking_2507.07955v2": "https://arxiv.org/pdf/2507.07955v2.pdf",
}

os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/parsed", exist_ok=True)

def download_and_parse():
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for name, url in PAPERS.items():
            pdf_path = f"data/pdfs/{name}.pdf"
            if not os.path.exists(pdf_path):
                print(f"Downloading {name}...")
                resp = client.get(url)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
            else:
                print(f"Already downloaded {name}.")
            
            # Extract text with page numbers
            doc = fitz.open(pdf_path)
            pages = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                pages.append({
                    "paper_id": name,
                    "page": page_num + 1,
                    "text": text
                })
            
            json_path = f"data/parsed/{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pages, f, indent=2)
            print(f"Parsed {name}: {len(pages)} pages.")

if __name__ == "__main__":
    download_and_parse()

