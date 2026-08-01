---
license: cc-by-sa-4.0
task_categories:
  - tabular-classification
tags:
  - biology
  - neuroscience
  - drosophila
  - olfaction
  - chemoinformatics
source_datasets:
  - ropensci/DoOR.data
size_categories:
  - n<1K
---

# DoOR Olfactory Receptor Responses (Processed)

A processed odor × olfactory receptor matrix derived from the
[Database of Odorant Responses (DoOR)](https://github.com/ropensci/DoOR.data) v2.0,
prepared for training the
[FlyWire Olfactory SNN](https://huggingface.co/{{HF_USER}}/flywire-olfactory-snn).

## Dataset description

Each row is an **odorant** (identified by InChIKey or name). Each column is an
**olfactory receptor** gene (Or10a, Or13a, … Or9a — 52 receptors total). Cell values
represent the **median response magnitude** across published studies compiled by DoOR.

| Feature | Value |
|---------|-------|
| Rows (odors) | ~500 (varies with DoOR version) |
| Columns (receptors) | 52 Or genes |
| Missing value handling | Column-median imputation, remaining NaN → 0 |
| Format | CSV (`odor_key` + 52 receptor columns) |

## Processing steps applied

This dataset is a **derivative work** of DoOR.data with the following transformations:

1. **Selected** 52 Or receptor CSV files from the full DoOR repository
2. **Extracted median response** across all published studies for each odor–receptor pair
3. **Identified odors** by InChIKey (falling back to chemical name when InChIKey is missing)
4. **Imputed missing values** with per-receptor column medians
5. **Filled remaining NaNs** with 0.0
6. **Merged** into a single odor × receptor matrix (CSV format)

Source code for this processing: [`src/flywire_snn/data/door.py`](https://github.com/YOUR_GITHUB/flywire-olfactory-snn/blob/main/src/flywire_snn/data/door.py)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{{HF_USER}}/door-olfactory-responses")
# or load the CSV directly:
import pandas as pd
df = pd.read_csv("door_or_merged.csv")
odor_keys = df["odor_key"]
receptor_matrix = df.drop(columns=["odor_key"]).to_numpy()
```

## License

This dataset is licensed under **CC BY-SA 4.0**, inheriting from the original
DoOR.data license.

### Attribution (required by CC BY-SA 4.0)

This dataset is derived from the **Database of Odorant Responses (DoOR)** v2.0:

- **Authors:** Daniel Münch, C. Giovanni Galizia, Shouwen Ma, Martin Strauch, Anja Nissler
- **Source:** https://github.com/ropensci/DoOR.data
- **Publications:**
  - Münch & Galizia (2016). DoOR 2.0. *Scientific Reports*, 6, 21841.
    https://doi.org/10.1038/srep21841
  - Galizia et al. (2010). *Chemical Senses*, 35(7), 551–563.
    https://doi.org/10.1093/chemse/bjq042

### ShareAlike

Any redistribution or derivative of this dataset must also be licensed under
CC BY-SA 4.0 or a compatible license.

## Citation

If you use this dataset, please cite both DoOR and this processed version:

```bibtex
@article{munch2016door,
  title={DoOR 2.0 -- Comprehensive mapping of Drosophila melanogaster odorant responses},
  author={M{\"u}nch, Daniel and Galizia, C. Giovanni},
  journal={Scientific Reports},
  volume={6},
  pages={21841},
  year={2016},
  doi={10.1038/srep21841}
}
```
