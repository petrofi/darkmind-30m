# RAG Plan For DarkMind

## Summary

RAG means retrieval-augmented generation. It is not training. Instead of changing model weights, a RAG system retrieves relevant local documents and provides them as context before the model answers.

## Why RAG Helps

DarkMind is a small model. Retraining every time project documentation changes would be slow and risky. RAG can help the model look up local project docs, experiment notes, config explanations, and dataset policies without pretending that the model memorized everything.

## Useful Scope

- Project documentation lookup
- Training config explanations
- Dataset policy reminders
- Experiment registry summaries
- Model card and roadmap lookup
- Local notes and approved documents

## Possible Future Script

`scripts/local_doc_search_demo.py`

A simple dependency-free first version could:

1. Read `.md` and `.txt` files from selected local folders.
2. Split documents into short chunks.
3. Score chunks with basic keyword overlap.
4. Print the top matching chunks for a user query.
5. Let the user paste those chunks into a model prompt.

## Limitations With A Tiny Model

- Retrieval can provide context, but the model may still misunderstand it.
- Keyword search is not semantic search.
- Long retrieved context can exceed block size.
- RAG does not fix weak reasoning or code generation by itself.
- Source quality still matters.

## Safety Note

RAG is safer for factual project updates than constantly retraining on new text, but retrieved documents must still be trusted and reviewed.
