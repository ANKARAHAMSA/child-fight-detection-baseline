# Alert-level check (what the tool actually raises)

Computed from the per-clip JSON files of the same run (small checkpoint, stride 1.0, threshold 0.6, default event logic).
A clip counts as "alerted" if the tool reported at least one event. Terminal output:

```
SURV n 300 median len 2.0s | alerts: caught 2 / 150 false alarms 0 / 150 | clips where score and alert disagree: 160
AIRTLab n 350 median len 5.3s | alerts: caught 151 / 230 false alarms 41 / 120 | clips where score and alert disagree: 20
```

| Dataset | Median clip | Accuracy | Fights alerted | False alarms | Precision |
|---|---|---|---|---|---|
| SURV | 2.0 s | 50.7% | 2/150 (1.3%) | 0/150 | 100% (2 of 2) |
| AIRTLab | 5.3 s | 65.7% | 151/230 (65.7%) | 41/120 (34.2%) | 78.6% |

All disagreements are clips with score >= 0.6 and no alert (SURV: 136 fights + 24 non-fights; AIRTLab: 15 + 5).
The per-clip JSON files are not included; the numbers above come from the printed totals.
