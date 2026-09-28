import hashlib
import sys
import threading
from pathlib import Path

import chromadb
import pathspec
from watchfiles import watch

from chunker import chunk_source

IGNORE_DIRS = {
    ".git",
}
LANGUAGES = {
    ".c": "c",
    ".h": "c",
}

MAX_FILE_BYTES = 500_000
FULL_SYNC_THRESHOLD = 500  # bigger bursts (branch switch) -> reconcile everything


class Indexer:
    def __init__(self, root, db_path="./code_index", name="codebase"):
        self.root = Path(root).resolve()
        self.col = chromadb.PersistentClient(path=db_path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
        gi = self.root / ".gitignore"
        self.ignore = pathspec.PathSpec.from_lines(
            "gitwildmatch", gi.read_text().splitlines() if gi.exists() else []
        )
        self.lock = threading.Lock()
        self.manifest = self._load_manifest()  # path -> file hash, kept in memory

    # ---------------------------------------------------------------- state
    def _load_manifest(self):
        manifest, offset = {}, 0
        while True:
            res = self.col.get(include=["metadatas"], limit=5000, offset=offset)
            if not res["ids"]:
                return manifest
            for m in res["metadatas"]:
                manifest[m["path"]] = m["file_hash"]
            offset += 5000

    def wanted(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False
        if any(part in IGNORE_DIRS for part in rel.parts):
            return False
        return not self.ignore.match_file(rel.as_posix())

    def iter_files(self, top: Path):
        for p in top.rglob("*"):
            if p.suffix in LANGUAGES and p.is_file() and self.wanted(p):
                yield p

    # ---------------------------------------------------------------- per-file ops
    def index_file(self, path: Path):
        if path.suffix not in LANGUAGES:
            return
        rel = path.relative_to(self.root).as_posix()
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return self.remove(rel)
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return  # vanished mid-read, or binary

        file_hash = hashlib.sha256(text.encode()).hexdigest()
        if self.manifest.get(rel) == file_hash:
            return  # touched/reformatted to same content

        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunk_source(path.suffix, text)):
            ids.append(f"{rel}#{i}")
            docs.append(ch.embed_text(rel))
            metas.append(
                {
                    "path": rel,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "language": LANGUAGES[path.suffix],
                    "kind": ch.kind,
                    "name": ch.name or "",
                    "parent": ch.parent or "",
                    "file_hash": file_hash,
                }
            )
        self.col.delete(where={"path": rel})
        for i in range(0, len(ids), 200):  # stay under Chroma's batch limit
            self.col.add(
                ids=ids[i : i + 200],
                documents=docs[i : i + 200],
                metadatas=metas[i : i + 200],
            )
        self.manifest[rel] = file_hash
        print(f"indexed {rel} ({len(ids)} chunks)")

    def remove(self, rel: str):
        self.col.delete(where={"path": rel})
        self.manifest.pop(rel, None)
        print(f"removed {rel}")

    def remove_tree(self, rel: str):
        """Handles a deleted file or a deleted/moved directory."""
        for p in [p for p in self.manifest if p == rel or p.startswith(rel + "/")]:
            self.remove(p)

    # ---------------------------------------------------------------- bulk + live
    def sync_all(self):
        """Reconcile index with disk. Cheap when nothing changed (hash check only)."""
        seen = set()
        for p in self.iter_files(self.root):
            seen.add(p.relative_to(self.root).as_posix())
            self.index_file(p)
        for rel in set(self.manifest) - seen:
            self.remove(rel)

    def apply(self, changes):
        paths = {Path(p) for _, p in changes}
        with self.lock:
            if len(paths) > FULL_SYNC_THRESHOLD:
                return self.sync_all()
            for p in sorted(paths):
                if p.is_dir():  # directory moved in / created
                    for f in self.iter_files(p):
                        self.index_file(f)
                elif p.is_file():
                    self.index_file(p)
                else:  # gone: file or whole directory
                    self.remove_tree(p.relative_to(self.root).as_posix())

    def _event_filter(self, _change, path):
        p = Path(path)
        # keep source files, plus extensionless paths (possible directories)
        return self.wanted(p) and (p.suffix in LANGUAGES or not p.suffix)

    def run(self):
        with self.lock:
            self.sync_all()  # catch up on edits made while offline
        for changes in watch(
            self.root, watch_filter=self._event_filter, debounce=2000, step=500
        ):
            self.apply(changes)


if __name__ == "__main__":
    Indexer(sys.argv[1]).run()
