# BUSI Duplicate and Label-Noise Audit

This audit is descriptive only. It does not modify labels, splits, or reported main metrics.

- BUSI VOC root: `<LOCAL_THYROID_ROOT>\busi\busi_voc_v3_square_consistent`
- Images audited: 647
- Trainval IDs: 518
- Fixed test IDs: 129
- Exact duplicate pairs by SHA-256: 1
- Near-duplicate pairs by dHash/aHash threshold <= 5: 141
- Cross-split duplicate/near-duplicate pairs: 41
- Label-conflict duplicate/near-duplicate pairs: 9
- Diagnostic label source: `<LOCAL_THYROID_ROOT>\busi\busi_voc_v3_square_consistent\manifests\label_manifest.csv`
- Note: VOC XML `object/name` is not used as the BUSI diagnostic label in this audit when `label_manifest.csv` is present.

Outputs:
- `busi_image_hash_manifest.csv`
- `busi_exact_duplicate_pairs.csv`
- `busi_near_duplicate_pairs.csv`
- `busi_cross_split_near_duplicate_pairs.csv`
- `busi_label_conflict_duplicate_pairs.csv`

First cross-split candidates:
- test_benign_0004 (test, y=0) vs trainval_benign_0321 (trainval, y=0), dHash=1, aHash=0
- test_benign_0005 (test, y=0) vs trainval_benign_0322 (trainval, y=0), dHash=1, aHash=1
- test_benign_0009 (test, y=0) vs trainval_benign_0326 (trainval, y=0), dHash=2, aHash=2
- test_benign_0081 (test, y=0) vs trainval_benign_0197 (trainval, y=0), dHash=5, aHash=1
- test_benign_0132 (test, y=0) vs trainval_benign_0038 (trainval, y=0), dHash=0, aHash=0
- test_benign_0133 (test, y=0) vs trainval_benign_0051 (trainval, y=0), dHash=4, aHash=0
- test_benign_0136 (test, y=0) vs trainval_benign_0050 (trainval, y=0), dHash=1, aHash=1
- test_benign_0207 (test, y=0) vs trainval_benign_0256 (trainval, y=0), dHash=5, aHash=0
- test_benign_0215 (test, y=0) vs trainval_benign_0270 (trainval, y=0), dHash=1, aHash=2
- test_benign_0228 (test, y=0) vs trainval_benign_0306 (trainval, y=0), dHash=2, aHash=0
