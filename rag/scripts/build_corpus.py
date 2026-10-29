import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from world.retrieval.corpus import build_corpus

if __name__ == "__main__":
    docs = build_corpus()
    print(f"Built archive corpus with {len(docs)} documents.")
    print("Hidden truth and forged flags were not exported.")
