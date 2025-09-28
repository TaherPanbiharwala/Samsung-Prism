# utils/rag.py
from __future__ import annotations
import json, os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection as ChromaCollection

# ---- Config ----
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma_menu")
DEFAULT_MENU_PATH   = os.getenv("MENU_JSON_PATH", "/Users/taherpanbiharwala/Desktop/Win/Flow/menu.json")
EMBED_MODEL         = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME     = os.getenv("CHROMA_COLLECTION", "menu_items")

# Global singletons
_client: Optional[chromadb.PersistentClient] = None
_coll: Optional[ChromaCollection] = None
_embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


def _client_ok() -> bool:
    return _client is not None and _coll is not None

def _to_primitive(v: Any) -> Any:
    """Chroma `metadatas` must be primitives: str|int|float|bool|None."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    # dicts/objects → readable string
    return str(v)

def init_db(menu_json_path: str = DEFAULT_MENU_PATH, persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """
    Initializes Chroma (persistent) and (re)indexes menu.json into collection if needed.
    Safe to call multiple times.
    """
    global _client, _coll

    os.makedirs(persist_dir, exist_ok=True)
    _client = chromadb.PersistentClient(path=persist_dir)
    _coll = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_func,
        metadata={"hnsw:space": "cosine"},
    )

    # Tell the type checker this is non-null now
    assert _coll is not None

    count = _coll.count()
    if count and not os.getenv("FORCE_REINDEX"):
        return

    if count:
        _coll.delete(where={})

    # Load menu.json
    with open(menu_json_path, "r") as f:
        data = json.load(f)

    # Expect either {"items":[...]} or "[...]"
    items: List[Dict[str, Any]]
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("menu.json must be a list of items or an object with 'items' list.")

    # Normalize & upsert
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, it in enumerate(items):
        # Required-ish fields (you can extend freely)
        _id = str(it.get("id") or it.get("sku") or it.get("code") or f"ITEM_{i:04d}")
        name = str(it.get("name") or it.get("title") or "Unknown Item")
        desc = str(it.get("description") or it.get("desc") or "")
        category = str(it.get("category") or it.get("type") or "Menu")
        price_raw = it.get("price") or it.get("cost") or 0
        try:
            price = float(price_raw)
        except Exception:
            price = 0.0
        tags_val = it.get("tags") or it.get("labels") or []
        tags_str = ", ".join(map(str, tags_val)) if isinstance(tags_val, (list, tuple)) else str(tags_val)

        # Embedding text: name + description + category + tags + extra flattened fields
        extra_pairs = []
        for k, v in it.items():
            if k in {"id","sku","code","name","title","description","desc","category","type","price","cost","tags","labels"}:
                continue
            extra_pairs.append(f"{k}: {_to_primitive(v)}")
        extras_txt = ". ".join(extra_pairs)

        doc = f"{name}. {desc} Category: {category}. Tags: {tags_str}."
        if extras_txt:
            doc += f" {extras_txt}."

        ids.append(_id)
        documents.append(doc)

        meta: Dict[str, Any] = {
            "id": _id,
            "name": name,
            "description": desc,
            "category": category,
            "price": price,
            "tags": tags_str,  # string (not list)
        }
        # keep other fields, but sanitized to primitives
        for k, v in it.items():
            if k in {"id","name","description","desc","category","type","price","cost","tags","labels"}:
                continue
            meta[k] = _to_primitive(v)

        metadatas.append(meta)

    if ids:
        _coll.upsert(ids=ids, documents=documents, metadatas=metadatas)

def _ensure():
    """Ensure client/collection exist; initialize if needed."""
    if not _client or not _coll:
        init_db()

def _pack_results(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Chroma returns lists per query; we query one text → take index 0 arrays
    out: List[Dict[str, Any]] = []
    if not res or not res.get("ids"):
        return out
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    dists = res.get("distances", [[None]*len(ids)])[0]
    for _id, meta, doc, dist in zip(ids, metas, docs, dists):
        # Reconstruct a menu item-like dict from metadata (includes name/price/category/tags)
        item = dict(meta)
        item["_doc"] = doc
        item["_distance"] = dist
        out.append(item)
    return out

def search_menu(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Vector search over menu. Returns list of dicts with keys:
      id, name, price, category, tags, ... plus _doc and _distance
    """
    _ensure()
    assert _coll is not None
    q = query.strip()
    if not q:
        # Popular default: simple generic query to get a broad spread
        q = "popular dishes"
    res = _coll.query(query_texts=[q], n_results=top_k)
    return _pack_results(res)

def find_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Use semantic search to fetch best match for an item name.
    """
    _ensure()
    assert _coll is not None
    res = _coll.query(query_texts=[name], n_results=1)
    out = _pack_results(res)
    return out[0] if out else None