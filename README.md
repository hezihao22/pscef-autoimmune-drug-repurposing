# PSCEF for autoimmune drug repurposing

Positive Support Calibrated Evidence Fusion combines three experts: source-disease Qwen literal-text logistic regression, source-disease TF-IDF logistic regression, and cached DeepSeek plausibility scores. Three known positive target-disease examples calibrate expert weights. Query labels never enter calibration.

## Research scope

The benchmark contains 4,477 archived approved-flag drugs for each of six autoimmune diseases: 26,862 pairs, 144 mapped known indications. Unobserved pairs are unlabelled rather than clinically ineffective. These six domains informed method development. The full-universe extension and repeated supports are exploratory; they are not an independent prospective validation.

The archive includes common-graph adaptations of TxGNN, TxGNN with semantic node features, REDDA, R-GCN, DistMult, ComplEx, RotatE, K-Paths, and LLaDR. Consult the Supporting Information for all changed inputs, language backbones, resource caps, and source-validation rules. The scores do not reproduce the original papers’ original datasets or establish superiority over every fully tuned system.

## Core usage

```python
from experiments_v3.pscef import fuse_scores
result = fuse_scores(expert_scores, positive_support_indices)
rank_scores = result['scores']
expert_weights = result['weights']
```

Rows are candidates, columns are experts, and larger expert scores mean stronger ranking preference. Positive supports are unique row indices. The core has no query-label parameter. Outputs are not clinical probabilities.

## Reproduce the cached comparison

Use Python 3.10 with the recorded environment. The local benchmark used an NVIDIA RTX 3060 with CUDA-enabled PyTorch. The cached evaluation itself does not require API keys. Install the required packages in an isolated environment, then run:

```text
python experiments_v3/evaluate.py
python experiments_v3/extra_analyses.py
python experiments_v3/verify.py
python experiments_v3/figures.py
python experiments_v3/figure1.py
```

The included base-score caches and model prediction arrays reproduce the reported evaluation without retraining or paid services. Read `experiments_v3/results/verification.json` for the independent metric check. `primary_summary.csv` uses three matched support draws for every method. Thirty-draw results apply to the cached experts and fusion controls; graph models have three support/initialization repetitions.

## Retrain the published architectures

Run `python fetch_upstream.py` to retrieve the recorded source commits. Vendor code is not redistributed here. Respect upstream licenses. Native Windows uses DGL 1.1.2; later DGL versions may require a compatible GraphBolt/PyTorch build. The tested relational encoder can use algebraically equivalent CUDA sparse aggregation, whose output and gradient checks are recorded in the protocol directory.

```text
python experiments_v3/graph_bench.py TxGNN --cuda-encoder
python experiments_v3/graph_bench.py TxGNN --text-features --cuda-encoder
python experiments_v3/graph_bench.py RGCN --cuda-encoder
python experiments_v3/graph_bench.py REDDA
python experiments_v3/kge_bench.py DistMult
python experiments_v3/kge_bench.py ComplEx
python experiments_v3/kge_bench.py RotatE
python experiments_v3/kge_bench.py LLaDR
```

Existing result files are resumable caches. To perform a genuinely fresh retraining, preserve the supplied results elsewhere and use a new output directory or rename the relevant prediction files. Source validation selects training length and the model is refitted on permitted source labels and target supports. Target-query labels are excluded from fitting and graph construction.

## Regenerate external representations

This step is optional and can incur service charges. Supply `DEEPSEEK_API_KEY` and `EMBEDDING_API_KEY` as environment variables; never put them in files committed to a repository. The generation scripts retain prompts, caches and usage records and impose conservative budget guards. Hosted aliases can change; new calls may not reproduce cached scores exactly. For raw graph reconstruction, download DRKG from its original project and obtain the retained normalized augmentation input under its original terms. The full private extraction archive is not redistributed.

## Files and interpretation

- `experiments_v3/pscef.py`: standalone ranking/calibration rule.
- `experiments_v3/protocol/`: frozen design, implementation commits and acceleration equivalence check.
- `experiments_v3/work/`: derived candidate and entity inputs, cached features and scores.
- `experiments_v3/results/`: all splits, training traces, predictions, metrics and verification.
- `figures/`: vector/raster figures and the source-data contract.
- `data/processed/`: exact mappings, reference labels, candidate coverage.

The data sources include DRKG, MONDO, the NCBI human gene table, and public model implementations listed in the protocol. Literature-derived graph relations remain noisy and are not clinical evidence. Source verification is described in the manuscript as author-led manual review with AI-assisted comparison; it is qualitative and is not a numerical accuracy estimate.

## Public release and data access

Repository: https://github.com/hezihao22/pscef-autoimmune-drug-repurposing

Download `PSCEF-reproducibility-v1.0.0.zip` from the [v1.0.0 release](https://github.com/hezihao22/pscef-autoimmune-drug-repurposing/releases/tag/v1.0.0), then extract it into a separate working directory. The archive includes the code plus the cached arrays required by the commands above. A repository clone alone provides code, configuration, and numerical summaries; it does not contain the large cached arrays. Check the archive SHA-256 against `RELEASE_SHA256.txt` in the release assets.

The versioned release supports the revised manuscript; it does not imply journal acceptance. Record-level source-review materials, full abstracts, private manuscript correspondence, and credentials are excluded.

## Results at a glance

| Method | Macro AP, three matched support draws |
| --- | ---: |
| PSCEF | 0.552 |
| Reciprocal rank fusion, same experts | 0.463 |
| Indication frequency | 0.441 |
| REDDA adaptation | 0.409 |
| Qwen regression with supports | 0.395 |
| RotatE adaptation | 0.376 |
| TxGNN adaptation | 0.300 |
| LLaDR adaptation | 0.155 |

The 30-support-draw PSCEF AP is 0.541. The disease-level PSCEF-minus-RRF AP interval includes zero and RRF has higher AUROC. These are heterogeneous exploratory results, not uniform superiority. Full metrics, disease-level results, matched split definitions, and intervals are supplied under `experiments_v3/results/`.

## License and attribution

New project code is MIT-licensed. Dataset records and derived data retain their source-specific terms, including noncommercial terms; the MIT license does not relicense them. See `DATA_SOURCES.md` and `THIRD_PARTY_NOTICES.md` before reusing data or external implementations. The archived DRKG license table is included for source attribution.

## Citation

Use `CITATION.cff` to cite the software. No journal acceptance or article DOI is claimed.
