import hashlib
import os
import sys
from pathlib import Path

import chromadb

import chunker

IGNORE_DIRS = {
    ".git",
}
LANGUAGES = {
    ".c": "c",
    ".h": "c",
}

MAX_FILE_BYTES = 500_000


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]  # prune in place
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in LANGUAGES and p.stat().st_size <= MAX_FILE_BYTES:
                yield p


def index_codebase(root: Path, db_path: str = "./code_index", name: str = "codebase"):
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    # What is already indexed: path -> file hash
    existing = {}
    col_meta = col.get(include=["metadatas"])["metadatas"]
    if col_meta is not None:
        for m in col_meta:
            existing[m["path"]] = m["file_hash"]

    seen = set()
    for path in iter_files(root):
        rel = str(path.relative_to(root))
        seen.add(rel)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # skip binary or oddly encoded files

        file_hash = hashlib.sha256(text.encode()).hexdigest()
        if existing.get(rel) == file_hash:
            continue  # unchanged, skip

        col.delete(where={"path": rel})  # drop old chunks of this file

        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunker.chunk_source(path.suffix, text)):
            ids.append(f"{rel}#{i}")
            docs.append(ch.embed_text(rel))
            metas.append(
                {
                    "path": rel,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "language": LANGUAGES[path.suffix],
                    "kind": ch.kind,
                    "name": ch.name or "",  # Chroma metadata can't hold None
                    "parent": ch.parent or "",
                    "file_hash": file_hash,
                }
            )
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
        print(f"indexed {rel} ({len(ids)} chunks)")

    # Remove chunks for files that were deleted
    for rel in set(existing) - seen:
        col.delete(where={"path": rel})
        print(f"removed {rel}")

    return col


if __name__ == "__main__":
    collection = index_codebase(Path(sys.argv[1]))
    res = collection.query(
        query_texts=[
            "command line syntax validation",
        ],
        n_results=5,
        where={
            "$and": [
                {"kind": "function"},
                {"language": "c"},
            ]
        },
    )
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        print(
            "-" * 8,
            f"{meta['path']}:{meta['start_line']}-{meta['end_line']}  (distance {dist:.3f})",
            "-" * 8,
            sep="\n",
        )
        print(doc[:300])
        print("\n\n")
