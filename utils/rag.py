# utils/rag.py
from __future__ import annotations
import json, os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

# --- pick a client class that exists in this Chroma version ---
try:
    from chromadb import PersistentClient as ChromaClient  # old/new persistent client
except Exception:  # fallback for versions that only expose Client
    from chromadb import Client as ChromaClient  # type: ignore[assignment]

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/Users/taherpanbiharwala/Desktop/Win/Flow/.chroma_menu")
DEFAULT_MENU_PATH   = os.getenv("MENU_JSON_PATH", "/Users/taherpanbiharwala/Desktop/Win/Flow/menu.json")
EMBED_MODEL         = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME     = os.getenv("CHROMA_COLLECTION", "menu_items")

_client: Optional[ChromaClient] = None
_coll: Any = None  # chroma's Collection (version-dependent)
_embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

def init_db(menu_json_path: str = DEFAULT_MENU_PATH, persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Create (or recreate) the collection and (re)index menu.json if empty or FORCE_REINDEX=1."""
    global _client, _coll

    os.makedirs(persist_dir, exist_ok=True)
    # instantiate client
    _client = ChromaClient(path=persist_dir)  # both Client/PersistentClient accept path in recent versions
    _coll = _client.get_or_create_collection(  # type: ignore[attr-defined]
        name=COLLECTION_NAME,
        embedding_function=_embed_func,
        metadata={"hnsw:space": "cosine"},
    )

    count = 0
    try:
        count = int(_coll.count())
    except Exception:
        count = 0

    if count and not os.getenv("FORCE_REINDEX"):
        return

    # drop & recreate the collection instead of delete(where={})
    try:
        _client.delete_collection(name=COLLECTION_NAME)  # type: ignore[attr-defined]
    except Exception:
        pass
    _coll = _client.get_or_create_collection(  # type: ignore[attr-defined]
        name=COLLECTION_NAME,
        embedding_function=_embed_func,
        metadata={"hnsw:space": "cosine"},
    )

    # --- load & normalize menu ---
    with open(menu_json_path, "r") as f:
        data = json.load(f)

    items: List[Dict[str, Any]]
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("menu.json must be a list, or an object with an 'items' list")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, it in enumerate(items):
        _id = str(it.get("id") or it.get("sku") or it.get("code") or f"ITEM_{i:04d}")
        name = str(it.get("name") or it.get("title") or "Unknown Item")
        desc = str(it.get("description") or it.get("desc") or "")
        category = str(it.get("category") or it.get("type") or "Menu")
        price = float(it.get("price") or it.get("cost") or 0)
        raw_tags = it.get("tags") or it.get("labels") or []
        tags_str = ", ".join(map(str, raw_tags))

        doc = f"{name}. {desc} Category: {category}. Tags: {tags_str}"

        ids.append(_id)
        documents.append(doc)
        metadatas.append({
            "id": _id,
            "name": name,
            "description": desc,
            "category": category,
            "price": price,
            "tags": tags_str,  # must be scalar for Chroma metadata
            **{k: v for k, v in it.items() if k not in
               {"id","name","description","desc","category","type","price","cost","tags","labels"}}
        })

    if ids:
        _coll.upsert(ids=ids, documents=documents, metadatas=metadatas)

def _ensure() -> None:
    """Create client/collection if missing."""
    if _client is None or _coll is None:
        init_db()


def _debug_count() -> int:
    try:
        return _coll.count()
    except Exception:
        return -1

def _pack_results(res: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not res or not getattr(res, "ids", None):
        return out
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]
    for _id, meta, doc, dist in zip(ids, metas, docs, dists):
        item = dict(meta or {})
        item["_doc"] = doc
        item["_distance"] = dist
        out.append(item)
    return out

def search_menu(query: str, top_k: int = 20):
    _ensure()
    print(f"[RAG] query={query!r}")
    try:
        print(f"[RAG] collection count={_coll.count()}")
    except Exception as e:
        print("[RAG] count() error:", repr(e))

    try:
        res = _coll.query(
            query_texts=[query],
            n_results=top_k,
            # 👇 remove "ids" from include (it is not allowed on your version)
            include=["metadatas", "documents", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]              # still present even if not in include
        mds = (res.get("metadatas") or [[]])[0]
        print(f"[RAG] hits={len(ids)}")

        out = []
        for i, md in enumerate(mds):
            rid = (ids[i] if i < len(ids) else "") or md.get("id") or md.get("sku") or md.get("uuid") or ""
            out.append({
                "id": rid,
                "name": (md.get("name") or md.get("title") or "").strip(),
                "price": md.get("price") or md.get("cost") or 0,
                "category": (md.get("category") or "").lower(),
                "tags": md.get("tags") or md.get("labels") or [],
            })
        return out
    except Exception as e:
        print("[RAG] query error:", repr(e))
        return []

def find_by_name(name: str) -> Optional[Dict[str, Any]]:
    _ensure()
    res = _coll.query(query_texts=[name], n_results=1)
    out = _pack_results(res)
    return out[0] if out else None

def debug_status():
    _ensure()
    try:
        print("[RAG] collection:", _coll.name)
        print("[RAG] count:", _coll.count())
        sample = _coll.get(include=["metadatas","ids"], limit=3)
        print("[RAG] sample metadatas:", sample.get("metadatas"))
    except Exception as e:
        print("[RAG] debug_status error:", repr(e))

def load_full_menu() -> List[Dict[str, Any]]:
    # if you already keep menu cached for search_menu, just return the list here
    return _ALL_MENU_ITEMS[:]  # however you store them