# Data sources and reproduction boundaries

| Source | Use in this release | Source / terms |
| --- | --- | --- |
| DRKG (2020) | Derived drug/protein/disease molecular graph; recorded indication labels | https://github.com/gnn4dr/DRKG; Apache-2.0 repository, per-source data terms |
| DrugBank and Hetionet records in DRKG | Historical candidate identifiers and mapped positive labels | DRKG's preserved source terms; see THIRD_PARTY_NOTICES.md |
| MONDO, 2026-09-01 release | Exact/equivalent disease mappings | https://mondo.monarchinitiative.org/; CC BY 4.0 |
| NCBI human gene information | Gene names for concept inputs | https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/ |
| Archived literature extractions | Qualified molecular relations used in derived literal inputs | Raw abstracts and record-level review materials are not redistributed |
| DeepSeek and Qwen services | Cached generated plausibility scores, descriptions, and embeddings | Prompts and settings are recorded in code/protocol; model weights are not included |

The release contains filtered graph edges, normalized candidate records, cached features and predictions, support splits, and numerical summaries. It supports exact cached evaluation and retraining from the supplied derived features. Full reconstruction from the private historical extraction archive is not provided. Hosted model aliases may change, and new API calls need not reproduce the cached responses. All query-level labels are database-reference indicators, not adjudicated efficacy or safety outcomes.

Source identifiers in graph relations are retained. Each source's applicable conditions continue to govern its records. The project does not claim ownership of third-party data or offer a blanket commercial data license.
