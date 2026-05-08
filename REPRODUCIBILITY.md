# Reproducibility and expected variation

This project is designed for transparent pattern-level reproduction, not bitwise-identical reproduction.

## Sources of variation

- PyTorch CPU/CUDA/MPS kernels may be nondeterministic.
- CAMELS preprocessing can differ depending on source files and missing-value handling.
- METR-LA arrays may differ across public mirrors.
- Small changes in seed, backend, or thread count can move NSE/MAE by small amounts.
- The position-paper claims are mechanism-oriented; use multi-seed summaries before making performance claims.

## Minimum evidence checklist

Before reporting a claim, save:

- command line and seed;
- `results.json` from training;
- router intervention JSON/Markdown;
- structure diagnosis JSON/Markdown;
- hardware backend and PyTorch version.

## Recommended interpretation

Do not claim that routing universally improves every domain. The intended interpretation is:

- weak-prior domains such as hydrology test whether differentiated learned structures make routing meaningful;
- strong-prior domains such as traffic test whether fixed physical topology dominates and learned structure acts as a supplement.
