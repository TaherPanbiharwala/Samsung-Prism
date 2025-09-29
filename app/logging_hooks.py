from langsmith import traceable

@traceable(name="NLU.log", tags=["nlu","log"])
def nlu_logger(text: str, stage: str|None, label: str, confidence: float):
    # You can also append to a local CSV if you want:
    # with open("nlu_logs.csv","a") as f: f.write(f"{stage},{confidence:.2f},{label},{text}\n")
    return {"stage": stage, "label": label, "confidence": confidence, "text": text}