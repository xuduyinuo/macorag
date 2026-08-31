# Resumable Teacher E5 Index Build Design

## Goal

Make the full teacher-corpus E5/FAISS build observable and safely resumable, while keeping the existing generic retrieval launcher compatible.

## Interface

- Add `scripts/build_teacher_retrieval.sh` with default config `config/build_retrieval_trajectory_train_e5.yml`.
- Preserve `CONFIG_PATH` as an optional override; users normally run `bash scripts/build_teacher_retrieval.sh`.
- Keep `scripts/build_retrieval.sh` and its evaluation-oriented default unchanged.
- Print dataset start/skip/complete messages and passage-encoding progress with elapsed time and ETA.

## Resume Contract

Each dataset writes temporary artifacts under its target directory:

- `.e5_embeddings.npy`: an open-format float32 NumPy memmap with shape `(corpus_count, dimension)`.
- `.e5_build_state.json`: schema version, dataset, source corpus SHA256, model name, max length, corpus count, dimension, and encoded row count.

The state is updated atomically after every encoded batch. A restart resumes only when every contract field matches. A mismatch discards no user data: the build raises a clear error instructing the caller to remove the two temporary artifacts or use a different output directory. Completed `corpus.jsonl`, `e5_Flat.index`, and `index_metadata.json` assets are validated and skipped.

After all vectors are encoded, the FAISS index and metadata are written atomically. Temporary embedding and state files are then removed. The source corpus is copied atomically only when finalizing so an interrupted build cannot masquerade as a complete index.

## Runtime Structure

- `E5Encoder.encode_passage_batches()` yields contiguous normalized float32 batches and reports progress through `tqdm` when enabled.
- `build_e5_faiss_index()` owns the resume manifest and memmap lifecycle.
- The CLI creates one `E5Encoder` and passes it to every dataset build, avoiding three model loads.
- Query encoding remains unchanged and does not display progress.

## Failure Handling

- Interruptions preserve the last completed batch.
- An invalid or stale resume state fails closed instead of mixing embeddings from different corpora/models.
- A complete but corrupt index is not skipped; validation fails with the existing contract error.
- No full teacher build is launched during verification; tests use small fake encoders and fake FAISS objects.

