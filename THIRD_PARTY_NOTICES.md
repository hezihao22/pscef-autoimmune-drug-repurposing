# Third-party notices

The MIT license applies only to new PSCEF project code and documentation. It does not relicense original databases, graph records, model weights, or external source implementations.

DRKG is by Vassilis N. Ioannidis, Xiang Song, Saurav Manchanda, Mufei Li, Xiaoqin Pan, Da Zheng, Xia Ning, Xiangxiang Zeng, and George Karypis (2020). Its repository uses Apache-2.0, while integrated records retain source-specific terms. DRKG's source table identifies DrugBank and GNBR records as CC BY-NC 4.0 and Hetionet as mixed-license. We preserve relation/source identifiers in the derived molecular graph and provide the upstream notices under `licenses/`. Derived materials are filtered subsets, normalized mappings, generated representations, or experimental outputs; they are not a complete or current release of the original databases. Noncommercial source terms continue to apply to relevant derived data.

Official DRKG: https://github.com/gnn4dr/DRKG
Source terms: https://github.com/gnn4dr/DRKG/blob/master/licenses/Readme.md
Hetionet source terms: https://github.com/hetio/hetionet#license
CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

TxGNN and other comparator implementations are obtained from their official repositories at the commits in `experiments_v3/protocol/upstream_sources.json`. They are not bundled. Availability of a repository without an explicit license is not a redistribution grant. `fetch_upstream.py` records provenance without assigning a new license to those sources.
