# Data placement

This repository does not include raw datasets.

Expected local layout:

```text
data/
  camels_us/
    camels_attributes_v2.0/
    basin_dataset_public_v1p2/
    nldas/
    ...
  traffic_datasets/
    real/
      metr_la_data.npy
      metr_la_adj.npy
```

CAMELS-US and METR-LA should be obtained from their official sources and placed in the paths above. The scripts intentionally fail fast if files are missing instead of silently downloading unofficial mirrors.
