# uncertain-ics

Code and data for *Uncertainty-Aware Cyber-Physical Threat Prioritisation for
Industrial Control Systems* (Kumar, Singh, Kumar).

`reproduce_script.py` regenerates every number, table value and data-driven
figure in the manuscript and its supplementary material.

## Run

```bash
pip install -r requirements.txt
python3 reproduce_script.py                    # primary run, Eq. (4) boundary rule
python3 reproduce_script.py --boundary renorm  # boundary-treatment robustness check
```

The defaults are N = 100,000 draws, seed 42, p_m = 0.6, w_opr = 0.4 and w_saf = 0.6.
A full run takes about 15 s.

## Inputs (in the script)

| Object | Paper location |
|---|---|
| `RATERS`: pre-consensus scores of the three assessors (84 judgements) | Supplement, Table S2 |
| `PANEL_CONSENSUS`: panel consensus, used for the agreement statistics | Supplement, Tables S2, S4 and S5 |
| `SCENARIOS`: consensus with L re-assessed (4, 3, 5, 4), used for all results | Main, Table 10 |
| `BASELINE_RM_IMPACT`, `CVSS_VECTORS` | Main, Section 4.5; Supplement, Tables S6 and S7 |

## Outputs

| File | Paper location |
|---|---|
| `outputs/results.json` | Tables 11 to 14, S4 and S5, and all statistics quoted in Section 5 |
| `outputs/results_boundary_renorm.json` | Section 5.4, boundary treatment |
| `figures/fig_sensitivity.pdf` | Fig. 4 (tornado and p_m sweep) |
| `figures/fig_rankprob.pdf` | Fig. 5 (rank-probability matrix) |
| `figures/fig_method_comparison.pdf` | Fig. 6 (method comparison) |
| `figures/fig_convergence.pdf` | Fig. S1 |
| `figures/fig_weights.pdf` | Fig. S2 |

The framework, architecture and attack-tree diagrams (Figs. 1 to 3) were
drawn separately and are not produced by the script.
