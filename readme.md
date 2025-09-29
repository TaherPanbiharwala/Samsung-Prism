# AI Waiter Assistant – Team Biscuit (VIT Vellore)

## 🚀 Project Overview
Restaurants face challenges like order delays, dietary risks, and inconsistent service during peak hours. Customers expect faster, personalized service and accurate orders, but staff often struggle to deliver consistently.

We built an **AI Waiter Assistant** — a conversational agent (voice + chat) that interacts with customers in a restaurant-like manner, takes orders, recommends dishes, and returns structured outputs for downstream use (POS, analytics, etc.).

---

## 🏗️ Architecture
Our system is modular and designed for extensibility:

1. **Frontend (Customer UI)**  
   - Streamlit-based chat & voice interface.  
   - Provides a simple, intuitive demo for Round-2.  

2. **Voice Layer**  
   - **ASR (Speech-to-Text):** Vosk  
   - **TTS (Speech Synthesis):** Piper  

3. **AI Core**  
   - **LLM:** Fine-tuned Gemma-2-2B-IT (served locally via Ollama)  
   - **Flow Orchestration:** LangGraph (manages conversation state & JSON compliance)  

4. **NLU & Retrieval**  
   - LangChain + Sentence-Transformers for embeddings  
   - ChromaDB as vector store for menu retrieval  

5. **Backend**  
   - FastAPI service exposing inference endpoints  

6. **Deployment**  
   - Dockerized setup for reproducibility and ease of submission  

---

## ⚡ Features (Current Implementation)
- **Voice & Chat Waiter Agent** (multilingual support).  
- **Order Taking** with structured JSON output (drinks, appetizers, mains, sides).  
- **Menu-Aware Conversations** powered by ChromaDB retrieval.  
- **Personalized Recommendations** (based on simple preference cues).  
- **Structured Slot-Filling** with guaranteed JSON compliance.  

---

## 📊 Model Fine-Tuning
We fine-tuned **Gemma-2-2B-IT** on a dataset of **17,000 synthetic restaurant conversations** across ~31 customer scenarios (greetings, ordering, dietary restrictions, complaints, etc.).

### Evaluation Results (Test Set)
- **BLEU:** 84.45  
- **chrF:** 96.34  
- **ROUGE-1/2/L F1:** ~0.93  
- **BERTScore F1:** 0.9277  
- **Structured Slot F1:** 1.000 (perfect)  
- **Exact Match:** 0.0 (expected due to natural variations in language)  

👉 Results show **high semantic & structural accuracy**, especially for slot-filling and JSON compliance.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Voice:** Vosk (ASR), Piper (TTS)  
- **LLM:** Fine-tuned Gemma-2-2B-IT (via Ollama)  
- **Orchestration:** LangGraph  
- **Retrieval:** LangChain + ChromaDB  
- **Backend:** FastAPI  
- **Deployment:** streamlit  

---

## 🔮 Future Scope
The system is designed to scale further with:  
- **Seating & Table Management** (automated seat allocation).  
- **Group Ordering & Bill Splitting** (multi-user conversations).  
- **Manager Analytics Dashboard** (sales insights, customer behavior, staff optimization).  
- **POS & Payment Integration** (Odoo POS / ERPNext or similar).  
- **Enhanced Recommendation Engine** (LightFM, Surprise, or hybrid deep models).  

---

## 👥 Team
**Team Biscuit – VIT Vellore**  
- Chimoy Chandan Deb (23BCT0050)  
- Yusuf Bootwala (23BDS0185)  
- Taher Panbiharwala (23BCT0028)  

---

## 📹 Submissions
- **Live App:** [Streamlit Deployment](https://taherpanbiharwala-samsung-prism-streamlit-app-hl7e7o.streamlit.app/)  
- **Demo Video:** https://drive.google.com/drive/folders/1WTXjrST7-KWTxu8nUN0A-owtKpENCHNH?usp=sharing  
- **Model and Dataset Link:** https://drive.google.com/drive/folders/1WTXjrST7-KWTxu8nUN0A-owtKpENCHNH?usp=sharing 
- **Supplementary File:** TeamBiscuit_VITVellore.pdf (in repo root)  

---

## 🔧 Setup Instructions

### Requirements
Python 3.11+  

```bash
git clone https://github.com/TaherPanbiharwala/Samsung-Prism.git
cd restaurant-automator
pip install -r requirements.txt