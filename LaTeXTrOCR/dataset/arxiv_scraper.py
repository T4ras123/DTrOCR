import arxiv
import os
import tarfile
import shutil
import re
import tqdm

def download_arxiv_papers(query, max_results=10, output_dir="arxiv_papers"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching for papers with query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in search.results():
        paper_id = result.entry_id.split("/")[-1]
        tar_path = os.path.join(output_dir, f"{paper_id}.tar.gz")
        result.download_source(filename=tar_path)

def extract_and_process_tar_files(input_dir, output_file):
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".tar.gz"):
                tar_path = os.path.join(input_dir, file_name)
                temp_dir = os.path.join("temp", file_name.replace(".tar.gz", ""))
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    with tarfile.open(tar_path) as tar:
                        tar.extractall(path=temp_dir)
                except tarfile.ReadError:
                    continue
                count = 0
                # Process LaTeX files
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".tex"):
                            tex_path = os.path.join(root, file)
                            try:
                                with open(tex_path, "r", encoding="utf-8") as texfile:
                                    content = texfile.read()
                                outfile.write(content + "\n<|endtext|>\n")
                                count += 1
                            except Exception as e:
                                print(f"Error processing {tex_path}: {e}")
                
                
                print(f"Processed {count} LaTeX files from {tar_path}")
                shutil.rmtree(temp_dir)
                
                os.remove(tar_path)

def main():
    queries = [
    "mathematics for beginners",
    "advanced calculus",
    "philosophy of science",
    "history of technology",
    "modern physics",
    "quantum mechanics",
    "climate change research",
    "econometrics",
    "psychology and behavior",
    "linguistics and semantics",
    "AI for healthcare",
    "robotics in agriculture",
    "computational biology",
    "data ethics",
    "education technology",
    "artificial creativity",
    "digital humanities",
    "sociology of the internet",
    "media studies and AI",
    "political science and machine learning",
    "introduction to computer science",
    "basic physics explanations",
    "simple algorithms",
    "non-linear dynamics",
    "high-energy particle physics",
    "stochastic processes in finance",
    "general science",
    "interdisciplinary studies",
    "scientific review papers",
    "research methodology",
    "scientific writing",
    ]
    output_dir = "arxiv_papers"
    output_file = "data.txt"
    
    with tqdm.tqdm(total=len(queries), desc="Processing Queries") as pbar:
        for query in queries:
            print(f"\nProcessing papers for query: {query}")
            download_arxiv_papers(query, max_results=100, output_dir=output_dir)
            extract_and_process_tar_files(output_dir, output_file)
            pbar.update(1)

    print(f"Processed LaTeX files and saved to {output_file}")

if __name__ == "__main__":
    main()
