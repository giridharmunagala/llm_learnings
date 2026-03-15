To clean data effectively for Pre-training I followed the below steps
- UTF-8 normalization
- NFKC normalization
- Emoji normalization
- Whitespace Canonicalization
- Length filtering 
- HTML clean up
- Repetition filtering
- Removing documents with excessive special character
- Perplexity filtering
- Deduplication
- 

## Implemented pipeline
- A staged JSONL pipeline now lives in [src/data/pretraining_pipeline.py](/home/giridhar/llm_learnings/src/data/pretraining_pipeline.py).
- The document-level cleaning functions live in [src/utils/pretrain_cleaning.py](/home/giridhar/llm_learnings/src/utils/pretrain_cleaning.py).
- The MinHash/LSH utilities live in [src/utils/dedup_utils.py](/home/giridhar/llm_learnings/src/utils/dedup_utils.py).

### Stages
- `ingest`
- `source_normalize`
- `document_clean`
- `quality_filter`
- `dedup_exact`
- `dedup_near`
- `tokenize_pack_export`

### Run locally
```bash
python3 -m src.data.run_pretraining_pipeline \
  --input-path data/inputs/pile_uncopyrighted/pile_1.jsonl \
  --output-root data/processed \
  --run-name pile_demo \
  --source-name pile_uncopyrighted \
  --source-type plain_text
```

The pipeline writes immutable stage outputs under `data/processed/<run_name>/stage=<stage>/source=<source>/`.










