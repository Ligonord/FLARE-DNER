based model: 5 DNER models.
BioBERT
ensemble_gpt: 5 DNER models' output as reference, GPT choose the best answer or extract the entities by itself.
ensemble_voting: 5 DNER model hard voting.
GPT-5-mini: GPT extract the entities without any reference.
m1: Transition-based.
m2: Span-based.
m3: Mac.
m4: W2NER.
m5: TOE.
qbc: FLARE-DNER.
rag: RAG-GPT + hard voting.
reevaluator: re-evaluate models performence after data quality evaluation.
union: rag union ensemble_voting.