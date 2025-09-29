# viz_graph.py
from graph import build_graph

def save_graph_png(path="graph.png"):
    compiled = build_graph()  # your g.compile() result
    png_bytes = compiled.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved {path}")

if __name__ == "__main__":
    save_graph_png("graph.png")