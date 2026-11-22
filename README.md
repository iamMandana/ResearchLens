# ResearchLens – AI Research Paper Assistant

## Overview

ResearchLens is an AI-powered application that helps researchers explore
academic papers. It reads PDF papers, retrieves the passages most relevant to a
question, and (in later stages) generates answers supported by page citations
using Retrieval-Augmented Generation (RAG).

## Objectives

- Upload and process academic PDF documents.
- Extract and organise research paper content.
- Generate text embeddings.
- Retrieve relevant information using semantic search.
- Generate answers using Large Language Models.
- Provide answers supported by document citations.

## Current Status

Implemented:

- PDF text extraction with page tracking (PyMuPDF)
- Sliding-window text chunking with overlap and metadata
- Embedding generation (Sentence Transformers, `all-MiniLM-L6-v2`)
- Semantic search using cosine similarity (NumPy)
- Command-line interface
- Unit tests (pytest)

Planned:

- Cached embeddings so each paper is embedded only once
- Local LLM answer generation (Ollama) with page citations
- Retrieval evaluation on questions with known answers
- Multiple PDF uploads and a FastAPI backend
- Docker packaging

## Architecture

```
PDF -> text extraction -> chunking -> embeddings -> similarity search -> top-k chunks
        (pdf_reader)    (text_chunker)  (Sentence      (NumPy cosine)
                                         Transformers)
```

## Project Structure

```
src/
  pdf_reader.py       Extract text from each PDF page
  text_chunker.py     Split pages into overlapping word chunks
  semantic_search.py  Rank chunks by cosine similarity to a query
  main.py             Command-line application
tests/                pytest tests for each module
```

## Design Decisions

### Why NumPy instead of FAISS for vector search

The original plan used FAISS as the vector index. I chose exact NumPy search
instead, for two reasons.

1. **FAISS would not run on my development machine.** `faiss-cpu` installed
   correctly, but importing it failed with `DLL load failed while importing
   _swigfaiss: An Application Control policy has blocked this file.` Windows
   Application Control blocks the compiled FAISS binary. This is a machine
   policy issue, not a code or installation issue, and I did not want to
   weaken my system security to work around it.
2. **At this project's scale, FAISS is not needed.** Embeddings are normalised,
   so cosine similarity is a single matrix-vector product
   (`embeddings @ query_embedding`). This is exact, simple to understand and
   fast for the number of chunks that a few dozen papers produce. FAISS's
   advantages (approximate indexes, very large collections) matter at far
   larger scale.

The real performance problem in the first version was re-embedding every chunk
on every query, and that does not require FAISS. The fix is to compute
embeddings once and store them, which is the next step.

The search code is kept behind a small interface, so a FAISS index can replace
the NumPy search later (for example inside a Linux Docker container) by changing
one module rather than the whole application.

### Chunking by words

`chunk_size=200` counts words, not tokens, with a 40-word overlap so context is
preserved across chunk boundaries. Each chunk keeps its page number so answers
can be cited.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python -m src.main
```

Enter the path to a PDF, then type a question. The application prints the most
relevant sections with their page numbers and similarity scores.

## Running Tests

```powershell
python -m pytest
```

## Technologies

- Python
- PyMuPDF
- Sentence Transformers
- NumPy
- pytest
- Git and GitHub
- Planned: Ollama, FastAPI, Docker (FAISS optional later)

## Author

Independent AI Engineering Project.
