"""Record upstream revisions and file metadata, without downloading bulk data."""
import json
from pathlib import Path
from urllib.request import Request, urlopen


def get_json(url):
    request = Request(url, headers={"User-Agent": "gpu-course-project-metadata/0.1"})
    with urlopen(request, timeout=40) as response:
        return json.load(response)


def main():
    hub = "https://huggingface.co/api/datasets/pisa-engine/ciff-hub"
    dataset = get_json(hub + "/revision/main")
    dataset_sha = dataset["sha"]
    files = get_json(hub + f"/tree/{dataset_sha}/splade?recursive=true")
    github = "https://api.github.com/repos/pisa-engine/BMP"
    bmp = get_json(github + "/commits/main")
    bmp_sha = bmp["sha"]
    tree = get_json(github + f"/git/trees/{bmp_sha}?recursive=1")
    manifest = {
        "dataset": "pisa-engine/ciff-hub", "dataset_revision": dataset_sha,
        "splade_files": files, "bmp_revision": bmp_sha,
        "bmp_input_sources": [item for item in tree["tree"] if
                              item["path"].startswith(("src/ciff/", "src/query/", "proto/"))],
        "status": "Metadata only; contents, ID mapping, query arithmetic, and local hashes remain unverified.",
    }
    output = Path(__file__).resolve().parents[1] / "results" / "input-source-metadata.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
