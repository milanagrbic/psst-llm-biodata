# LLM-generated biomedical tail-risk

Supplementary materials for the paper *"Prompting Shapes the Statistical Tails of LLM-Generated Biomedical Data"* by Andrej Novak, Milana Grbić, Matej Ivaniček, and Dragan Matić.

This repository includes code for LLM-based biomedical data generation, statistical tail analysis, and experimental results.


## Files

- `main_llm.ipynb` — main notebook executed once for checking.
- `tail_recompute_lib.py` — reusable analysis functions used by the notebook.
- `rezultati.csv` — input dataset with columns `i`, `j`, `k`, `t`, `p`, `r`, and `data` where `data` is a Python-style list in square brackets.
- `GenerateData.ipynb` — notebook for generating LLM-based biomedical datasets.

## Run
```bash
pip install -r requirements.txt
jupyter notebook llm_tail_recompute.ipynb
```
