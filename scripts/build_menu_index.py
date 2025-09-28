# scripts/build_menu_index.py
import os
from utils.rag import init_db

if __name__ == "__main__":
    menu_path = os.getenv("MENU_JSON_PATH", "/Users/taherpanbiharwala/Desktop/Win/Flow/menu.json")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./.chroma_menu")
    print(f"Indexing {menu_path} → {persist_dir}")
    init_db(menu_path, persist_dir)
    print("Done.")