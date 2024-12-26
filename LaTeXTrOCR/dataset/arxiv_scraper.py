import os
import tarfile
import requests
from tqdm import tqdm
import arxiv

class ArxivScraper:
    def __init__(self, query, max_results, output_file):
        self.query = query
        self.max_results = max_results
        self.output_file = output_file

    def fetch_papers(self):
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return search.results()

    def download_source(self, paper):
        try:
            # Get the full paper ID (e.g., 'math/9204225v1')
            paper_id = paper.get_short_id()
            # Construct the source files URL
            source_url = f"https://arxiv.org/e-print/{paper_id}"
            response = requests.get(source_url, stream=True)

            if response.status_code != 200:
                print(f"Failed to download source for {paper_id}: HTTP {response.status_code}")
                return None

            # Replace '/' with '_' to avoid directory issues
            tar_filename = f"{paper_id.replace('/', '_')}.tar.gz"
            tar_path = tar_filename

            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return tar_path
        except Exception as e:
            print(f"Error downloading paper {paper_id}: {e}")
            return None

    def extract_tex_content(self, tar_path):
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.tex'):
                        f = tar.extractfile(member)
                        if f:
                            return f.read().decode('utf-8')
            print(f"No .tex file found in {tar_path}")
            return None
        except Exception as e:
            print(f"Error extracting .tex from {tar_path}: {e}")
            return None

    def process_tex(self, tex_content):
        # Add any processing logic if needed
        return tex_content

    def save_content(self, processed_text):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"<|latex|>{processed_text}<|endtext|>\n")

    def scrape(self):
        papers = self.fetch_papers()
        for paper in tqdm(papers, desc="Processing papers"):
            tar_path = self.download_source(paper)
            if tar_path:
                tex_content = self.extract_tex_content(tar_path)
                processed = self.process_tex(tex_content)
                if processed:
                    self.save_content(processed)
                try:
                    os.remove(tar_path)
                except OSError as e:
                    print(f"Error removing file {tar_path}: {e}")

if __name__ == "__main__":
    scraper = ArxivScraper(query="cat:math.AG", max_results=100, output_file="papers.txt")
    scraper.scrape()