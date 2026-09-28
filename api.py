"""Run with:  uvicorn api:app --port 8000     (one worker, see notes)"""

import os
import secrets
import threading
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from hybrid import extract_identifiers, rg_search
from watcher import Indexer

ROOT = Path(os.environ.get("CODE_ROOT", "codebase")).resolve()
DB_PATH = os.environ.get("CHROMA_PATH", "./code_index")
COLLECTION = os.environ.get("COLLECTION", "codebase")
WATCH = os.environ.get("WATCH", "1") == "1"
API_KEY = os.environ.get("API_KEY")  # unset = no auth
RRF_K = 60
MAX_RG_FILES = 20

# ------------------------------------------------------------------ app setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    idx = Indexer(ROOT, DB_PATH, COLLECTION)
    app.state.idx, app.state.watcher = idx, None
    try:
        idx.col.query(
            query_texts=["warmup"], n_results=1
        )  # load the embedding model now
    except Exception:
        pass
    if WATCH:
        t = threading.Thread(target=idx.run, daemon=True, name="watcher")
        t.start()
        app.state.watcher = t
    yield


app = FastAPI(title="Code search API", lifespan=lifespan)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_key(key: str | None = Security(api_key_header)):
    if API_KEY and not secrets.compare_digest(key or "", API_KEY):
        raise HTTPException(401, "invalid or missing API key")


def get_idx(request: Request) -> Indexer:
    return request.app.state.idx


router = APIRouter(dependencies=[Depends(check_key)])

# ------------------------------------------------------------------ schemas


class Filters(BaseModel):
    """Each field is a list of accepted values (OR within a field, AND across fields)."""

    language: list[str] | None = None
    kind: list[str] | None = None
    path: list[str] | None = None
    name: list[str] | None = None
    parent: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    n_results: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "semantic"] = "hybrid"
    filters: Filters | None = None
    max_distance: float | None = Field(
        None, gt=0, description="drop vector hits farther than this"
    )
    include_text: bool = True


class Hit(BaseModel):
    id: str
    path: str
    start_line: int
    end_line: int
    language: str
    kind: str
    name: str
    parent: str
    score: float | None = None
    distance: float | None = None
    matched_by: list[str] = []
    text: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[Hit]


class ChunkPage(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool
    chunks: list[Hit]


# ------------------------------------------------------------------ helpers


def and_where(*parts):
    parts = [p for p in parts if p]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def build_where(f: Filters | None):
    if f is None:
        return None
    clauses = [{field: {"$in": vals}} for field, vals in f.model_dump().items() if vals]
    return and_where(*clauses)


def to_hit(cid, meta, text=None, score=None, distance=None, matched_by=None) -> Hit:
    return Hit(
        id=cid,
        path=meta["path"],
        start_line=meta["start_line"],
        end_line=meta["end_line"],
        language=meta["language"],
        kind=meta["kind"],
        name=meta["name"],
        parent=meta["parent"],
        text=text,
        score=score,
        distance=distance,
        matched_by=matched_by or [],
    )


def fetch_texts(col, ids):
    if not ids:
        return {}
    res = col.get(ids=ids, include=["documents"])
    return dict(zip(res["ids"], res["documents"]))


def retrieve(idx: Indexer, req: SearchRequest) -> list[Hit]:
    col, where = idx.col, build_where(req.filters)
    pool = req.n_results * 3

    # 1. vector ranking
    vec = col.query(
        query_texts=[req.query],
        n_results=pool,
        where=where,
        include=["metadatas", "distances"],
    )
    metas = dict(zip(vec["ids"][0], vec["metadatas"][0]))
    dists = dict(zip(vec["ids"][0], vec["distances"][0]))
    vec_ids = [
        i
        for i in vec["ids"][0]
        if req.max_distance is None or dists[i] <= req.max_distance
    ]

    # 2. keyword ranking (ripgrep), mapped to chunks with one query per file
    rg_ids: list[str] = []
    if req.mode == "hybrid":
        lines_by_file = defaultdict(list)
        for rel, line in rg_search(idx.root, extract_identifiers(req.query)):
            lines_by_file[rel].append(line)
        chunk_hits = Counter()
        for rel, lines in list(lines_by_file.items())[:MAX_RG_FILES]:
            res = col.get(where=and_where(where, {"path": rel}), include=["metadatas"])
            for cid, m in zip(res["ids"], res["metadatas"]):
                n = sum(m["start_line"] <= ln <= m["end_line"] for ln in lines)
                if n:
                    chunk_hits[cid] += n
                    metas[cid] = m
        rg_ids = [cid for cid, _ in chunk_hits.most_common(pool)]

    # 3. reciprocal rank fusion
    scores = Counter()
    for ranking in (vec_ids, rg_ids):
        for rank, cid in enumerate(ranking):
            scores[cid] += 1 / (RRF_K + rank + 1)
    top = [cid for cid, _ in scores.most_common(req.n_results)]

    texts = fetch_texts(col, top) if req.include_text else {}
    vec_set, rg_set = set(vec_ids), set(rg_ids)
    return [
        to_hit(
            cid,
            metas[cid],
            texts.get(cid),
            round(scores[cid], 5),
            dists.get(cid),
            [
                s
                for s, present in (
                    ("vector", cid in vec_set),
                    ("ripgrep", cid in rg_set),
                )
                if present
            ],
        )
        for cid in top
    ]


# ------------------------------------------------------------------ endpoints


@app.get("/health")
def health(request: Request):
    idx, t = request.app.state.idx, request.app.state.watcher
    return {
        "status": "ok",
        "chunks": idx.col.count(),
        "files": len(idx.manifest),
        "watching": bool(t and t.is_alive()),
    }


# Plain `def` (not `async def`): FastAPI runs these in a thread pool, so the blocking
# embedding and Chroma calls don't stall the event loop.
@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, idx: Indexer = Depends(get_idx)):
    t0 = time.perf_counter()
    hits = retrieve(idx, req)
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        hits=hits,
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@router.get("/chunk", response_model=list[Hit])
def get_chunk(
    chunk_id: str = Query(..., alias="id"),
    neighbors: int = Query(0, ge=0, le=5, description="adjacent chunks on each side"),
    include_text: bool = True,
    idx: Indexer = Depends(get_idx),
):
    col = idx.col
    found = col.get(ids=[chunk_id], include=["metadatas"])
    if not found["ids"]:
        raise HTTPException(404, "chunk not found")
    path = found["metadatas"][0]["path"]

    res = col.get(
        where={"path": path},
        include=["metadatas", "documents"] if include_text else ["metadatas"],
    )
    docs = res["documents"] if include_text else [None] * len(res["ids"])
    rows = sorted(
        zip(res["ids"], res["metadatas"], docs),
        key=lambda r: (r[1]["start_line"], r[0]),
    )
    pos = next(i for i, r in enumerate(rows) if r[0] == chunk_id)
    window = rows[max(0, pos - neighbors) : pos + neighbors + 1]
    return [to_hit(cid, m, d) for cid, m, d in window]


@router.get("/files")
def list_files(
    prefix: str = "",
    suffix: str = "",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    idx: Indexer = Depends(get_idx),
):
    paths = sorted(
        p for p in list(idx.manifest) if p.startswith(prefix) and p.endswith(suffix)
    )  # list(): watcher mutates it
    return {"total": len(paths), "files": paths[offset : offset + limit]}


@router.get("/files/chunks", response_model=list[Hit])
def file_chunks(path: str, include_text: bool = False, idx: Indexer = Depends(get_idx)):
    res = idx.col.get(
        where={"path": path},
        include=["metadatas", "documents"] if include_text else ["metadatas"],
    )
    if not res["ids"]:
        raise HTTPException(404, "file not indexed")
    docs = res["documents"] if include_text else [None] * len(res["ids"])
    rows = sorted(
        zip(res["ids"], res["metadatas"], docs), key=lambda r: r[1]["start_line"]
    )
    return [to_hit(cid, m, d) for cid, m, d in rows]


@router.post("/reindex")
def reindex(idx: Indexer = Depends(get_idx)):
    """Full reconcile with disk. Blocks until done; cheap when nothing changed."""
    with idx.lock:
        idx.sync_all()
    return {"status": "ok", "files": len(idx.manifest)}


@router.get("/chunks", response_model=ChunkPage)
def list_chunks(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_text: bool = False,
    language: list[str] | None = Query(None),
    kind: list[str] | None = Query(None),
    path: list[str] | None = Query(None),
    name: list[str] | None = Query(None),
    parent: list[str] | None = Query(None),
    idx: Indexer = Depends(get_idx),
):
    col = idx.col
    where = build_where(
        Filters(language=language, kind=kind, path=path, name=name, parent=parent)
    )

    # Total matching chunks: O(1) when unfiltered, ids-only scan otherwise
    total = (
        col.count() if where is None else len(col.get(where=where, include=[])["ids"])
    )

    res = col.get(
        where=where,
        limit=limit,
        offset=offset,
        include=["metadatas", "documents"] if include_text else ["metadatas"],
    )
    docs = res["documents"] if include_text else [None] * len(res["ids"])
    chunks = [
        to_hit(cid, m, d) for cid, m, d in zip(res["ids"], res["metadatas"], docs)
    ]

    return ChunkPage(
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(chunks) < total,
        chunks=chunks,
    )


app.include_router(router)
