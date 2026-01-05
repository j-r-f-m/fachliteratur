# normen-rag

Ein lokales **RAG-System (Retrieval-Augmented Generation)** zur Recherche in **Normen und Fachliteratur (PDF)**.  
Das Projekt ermöglicht es, Fragen **ausschließlich auf Basis bereitgestellter PDFs** zu beantworten – ohne externes Wissen, ohne Internetzugriff und reproduzierbar.

---

## Projektziel

Ziel dieses Projekts ist ein **kontrollierter KI-Assistent** für technische Dokumente, insbesondere:

- DIN-Normen
- Eurocodes
- Richtlinien
- Fachliteratur (PDF)

Der Assistent:
- durchsucht die Dokumente **semantisch**
- gibt Antworten **nur aus den hinterlegten PDFs**
- eignet sich für **ingenieurmäßige Recherche**, nicht für freie Interpretation

---

## Funktionsprinzip (RAG)

Das System folgt dem RAG-Ansatz:

1. **Retrieval**
   - PDFs werden in Text zerlegt
   - Textabschnitte werden in Vektoren (Embeddings) umgewandelt
   - relevante Abschnitte werden über eine Vector Database gesucht

2. **Augmented**
   - die gefundenen Textstellen werden dem Sprachmodell als Kontext übergeben

3. **Generation**
   - das Sprachmodell formuliert eine Antwort
   - **ohne externes Wissen**
   - **nur aus dem gelieferten Kontext**

---

## Technologiestack

- Python 3.10 / 3.11
- LangChain
- ChromaDB (lokale Vector Database)
- Sentence-Transformers (Embeddings)
- HuggingFace LLM (CPU-tauglich)
- VS Code (empfohlen)

---

## Projektstruktur
normen-rag/
├─ rag.py # Hauptskript
├─ requirements.txt # Python-Abhängigkeiten
├─ data/ # PDFs (nicht versioniert)
├─ chroma_db/ # Vector DB (lokal, nicht versioniert)
├─ venv/ # virtuelle Umgebung
└─ README.md