# Run report: main

- generated: 2026-10-13T09:50:55
- git commit: HEAD
- feature set: leakage_free
- split scheme: held_out_transition
- seconds per row (derived): 3.594

## Label audit
```json
{
  "n_rows": 11519,
  "n_transitions": 1,
  "transition_rows_1based": [
    8943
  ],
  "n_healthy": 8942,
  "n_clogged": 2577,
  "clogged_fraction": 0.2237,
  "validation_score_range": [
    0,
    13
  ],
  "min_score_when_clogged": 11,
  "max_score_when_healthy": 10,
  "score_separates_label": true,
  "empirical_clogged_threshold": 11
}
```

## Split composition
```json
{
  "scheme": "held_out_transition",
  "segments_1based": {
    "train_a": [
      1,
      7140
    ],
    "val_a": [
      7201,
      7800
    ],
    "test": [
      7861,
      9600
    ],
    "val_b": [
      9661,
      10260
    ],
    "train_b": [
      10321,
      11519
    ]
  },
  "train": {
    "n": 8339,
    "n_pos": 1199,
    "n_neg": 7140,
    "pos_frac": 0.1438
  },
  "val": {
    "n": 1200,
    "n_pos": 600,
    "n_neg": 600,
    "pos_frac": 0.5
  },
  "test": {
    "n": 1740,
    "n_pos": 658,
    "n_neg": 1082,
    "pos_frac": 0.3782
  },
  "n_purged": 240
}
```

## Horizon h0

### XGBoost (primary) — test metrics
```json
{
  "threshold": 0.9333256940020761,
  "balanced_accuracy": 0.7381495485676081,
  "precision": 0.5376432078559739,
  "recall": 0.9984802431610942,
  "f1": 0.698936170212766,
  "mcc": 0.5051331291413838,
  "brier": 0.3324131890232667,
  "ece": 0.3375981643418608,
  "confusion_matrix": [
    [
      517,
      565
    ],
    [
      1,
      657
    ]
  ],
  "pr_auc": 0.9327213267119943,
  "roc_auc": 0.975388928529291
}
```
### Bootstrap 95% CIs (moving-block)
```json
{
  "f1": {
    "ci_low": 0.486359464123308,
    "ci_high": 0.834799956246126,
    "n_valid_resamples": 1000
  },
  "balanced_accuracy": {
    "ci_low": 0.6332179512868465,
    "ci_high": 0.8309971533219835,
    "n_valid_resamples": 1000
  },
  "mcc": {
    "ci_low": 0.3299011289609554,
    "ci_high": 0.6620050563538731,
    "n_valid_resamples": 1000
  },
  "pr_auc": {
    "ci_low": 0.8437594208485227,
    "ci_high": 0.9725738039265617,
    "n_valid_resamples": 1000
  },
  "roc_auc": {
    "ci_low": 0.9571116116072128,
    "ci_high": 0.9877291466893777,
    "n_valid_resamples": 1000
  },
  "_config": {
    "block_len": 50,
    "n_resamples": 1000,
    "seed": 42
  }
}
```
### Detection delay
```json
{
  "onset_row_in_window": 1082,
  "alarm_decision_row_in_window": 484,
  "delay_rows": -598,
  "consecutive_rule": 3,
  "alarm_convention": "delay measured at the k-th confirmation row",
  "cleared_alarm_episodes_before_onset": 4,
  "n_events": 1,
  "note": "single onset event: no CI on delay; see sensitivity table",
  "delay_minutes": -35.82
}
```
### RandomForest baseline — test metrics
```json
{
  "threshold": 0.962581617734156,
  "balanced_accuracy": 0.9475683890577508,
  "precision": 1.0,
  "recall": 0.8951367781155015,
  "f1": 0.9446672012830793,
  "mcc": 0.9173195918768757,
  "brier": 0.2871187297141409,
  "ece": 0.30378368068896183,
  "confusion_matrix": [
    [
      1082,
      0
    ],
    [
      69,
      589
    ]
  ],
  "pr_auc": 0.965615803488036,
  "roc_auc": 0.9711737804021597
}
```
### XGB vs RF comparison
```json
{
  "paired_block_bootstrap_f1_diff": {
    "metric": "f1_diff (A - B)",
    "mean": -0.2773195509264989,
    "ci_low": -0.483980833199206,
    "ci_high": -0.12060032604601224,
    "n_valid_resamples": 1000
  },
  "mcnemar": {
    "b_A_wrong_B_right": 565,
    "c_A_right_B_wrong": 68,
    "p_value": 1.9224760285777082e-98,
    "caveat": "rows are autocorrelated within one lifecycle; effective sample size is inflated, so this p-value is optimistic"
  }
}
```

## Six-level risk (secondary target)
```json
{
  "n_classes": 6,
  "quadratic_weighted_kappa": 0.422490129183118,
  "level_mae": 0.5971264367816091,
  "within_one_accuracy": 1.0,
  "per_level_recall": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0397,
    1.0
  ],
  "confusion_matrix": [
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      481,
      43,
      558
    ],
    [
      0,
      0,
      0,
      0,
      0,
      658
    ]
  ],
  "test_level_counts": [
    0,
    0,
    0,
    0,
    1082,
    658
  ],
  "caveats": [
    "Risk levels are bands of the labeling rule's own score: this measures rule emulation from causal features, not independently observed risk.",
    "Score levels are nearly monotone in time within the single lifecycle; part of the measured skill is trend-following.",
    "Thesis text places the clogged decision at score >= 12; the stored data uses >= 11. Bands align with the data; the discrepancy is documented."
  ],
  "bands_score_ranges": [
    [
      0,
      1
    ],
    [
      2,
      3
    ],
    [
      4,
      5
    ],
    [
      6,
      8
    ],
    [
      9,
      10
    ],
    [
      11,
      20
    ]
  ]
}
```

## Caveats (mandatory in any publication)
- Single filter lifecycle: all results describe one clogging episode on one rig; cross-filter generalization is untested and unclaimed.
- Labels are generated by the thesis's deterministic multi-criteria rule, not by independent ground truth (e.g. visual inspection); supervised results therefore measure how well the model reproduces/anticipates the rule from causal sensor features.
- Held-out-transition split (S2): the clogged training segment post-dates the test window, so S2 evaluates interpolation across a held-out transition, not pure forecasting.
- Detection delay is estimated from a single onset event; no confidence interval is possible (sensitivity across alarm rules reported instead).
- Block-bootstrap CIs quantify within-lifecycle sampling noise only.
- Risk levels are bands of the labeling rule's own score: this measures rule emulation from causal features, not independently observed risk.
- Score levels are nearly monotone in time within the single lifecycle; part of the measured skill is trend-following.
- Thesis text places the clogged decision at score >= 12; the stored data uses >= 11. Bands align with the data; the discrepancy is documented.