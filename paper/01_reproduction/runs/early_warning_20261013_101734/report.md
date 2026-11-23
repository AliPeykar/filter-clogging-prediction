# Run report: early_warning

- generated: 2026-10-13T10:19:19
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

## Horizon h15

### XGBoost (primary) — test metrics
```json
{
  "threshold": 0.993839538683579,
  "balanced_accuracy": 0.7572633552014996,
  "precision": 0.5650713685978169,
  "recall": 1.0,
  "f1": 0.7221030042918455,
  "mcc": 0.5392071145928579,
  "brier": 0.3095428647396879,
  "ece": 0.3161699477914541,
  "confusion_matrix": [
    [
      549,
      518
    ],
    [
      0,
      673
    ]
  ],
  "pr_auc": 0.9833998092831744,
  "roc_auc": 0.9897415508619382
}
```
### Bootstrap 95% CIs (moving-block)
```json
{
  "f1": {
    "ci_low": 0.5170433579677016,
    "ci_high": 0.8480503020159998,
    "n_valid_resamples": 1000
  },
  "balanced_accuracy": {
    "ci_low": 0.657561981746341,
    "ci_high": 0.8481459245238377,
    "n_valid_resamples": 1000
  },
  "mcc": {
    "ci_low": 0.3596163587332897,
    "ci_high": 0.6971727892379673,
    "n_valid_resamples": 1000
  },
  "pr_auc": {
    "ci_low": 0.9338434622182636,
    "ci_high": 1.0,
    "n_valid_resamples": 1000
  },
  "roc_auc": {
    "ci_low": 0.9620521666898106,
    "ci_high": 1.0,
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
  "onset_row_in_window": 1067,
  "alarm_decision_row_in_window": 489,
  "delay_rows": -578,
  "consecutive_rule": 3,
  "alarm_convention": "delay measured at the k-th confirmation row",
  "cleared_alarm_episodes_before_onset": 3,
  "n_events": 1,
  "note": "single onset event: no CI on delay; see sensitivity table",
  "delay_minutes": -34.62
}
```
### RandomForest baseline — test metrics
```json
{
  "threshold": 0.984541059596134,
  "balanced_accuracy": 0.9383358098068351,
  "precision": 1.0,
  "recall": 0.8766716196136701,
  "f1": 0.934283452098179,
  "mcc": 0.9018862446960746,
  "brier": 0.3133792743119388,
  "ece": 0.3191026488096187,
  "confusion_matrix": [
    [
      1067,
      0
    ],
    [
      83,
      590
    ]
  ],
  "pr_auc": 0.9573725684445515,
  "roc_auc": 0.9696312862854429
}
```
### XGB vs RF comparison
```json
{
  "paired_block_bootstrap_f1_diff": {
    "metric": "f1_diff (A - B)",
    "mean": -0.24225609456920866,
    "ci_low": -0.44313344184888925,
    "ci_high": -0.09084569006115323,
    "n_valid_resamples": 1000
  },
  "mcnemar": {
    "b_A_wrong_B_right": 518,
    "c_A_right_B_wrong": 83,
    "p_value": 8.470785342312122e-78,
    "caveat": "rows are autocorrelated within one lifecycle; effective sample size is inflated, so this p-value is optimistic"
  }
}
```

## Horizon h30

### XGBoost (primary) — test metrics
```json
{
  "threshold": 0.9815945579290593,
  "balanced_accuracy": 0.766858807144752,
  "precision": 0.5841836734693877,
  "recall": 0.998546511627907,
  "f1": 0.7371244635193133,
  "mcc": 0.5575327870934639,
  "brier": 0.3069027067620177,
  "ece": 0.3124085057900304,
  "confusion_matrix": [
    [
      563,
      489
    ],
    [
      1,
      687
    ]
  ],
  "pr_auc": 0.9372539705054499,
  "roc_auc": 0.9720327836236626
}
```
### Bootstrap 95% CIs (moving-block)
```json
{
  "f1": {
    "ci_low": 0.5403391429921602,
    "ci_high": 0.8564114828042363,
    "n_valid_resamples": 1000
  },
  "balanced_accuracy": {
    "ci_low": 0.6649963768115943,
    "ci_high": 0.8555237499794861,
    "n_valid_resamples": 1000
  },
  "mcc": {
    "ci_low": 0.38170200126969644,
    "ci_high": 0.709146287391341,
    "n_valid_resamples": 1000
  },
  "pr_auc": {
    "ci_low": 0.8559138123424416,
    "ci_high": 0.9760847750999764,
    "n_valid_resamples": 1000
  },
  "roc_auc": {
    "ci_low": 0.9425704752608546,
    "ci_high": 0.9892928281161391,
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
  "onset_row_in_window": 1052,
  "alarm_decision_row_in_window": 489,
  "delay_rows": -563,
  "consecutive_rule": 3,
  "alarm_convention": "delay measured at the k-th confirmation row",
  "cleared_alarm_episodes_before_onset": 4,
  "n_events": 1,
  "note": "single onset event: no CI on delay; see sensitivity table",
  "delay_minutes": -33.72
}
```
### RandomForest baseline — test metrics
```json
{
  "threshold": 0.9757541667029428,
  "balanced_accuracy": 0.9222383720930232,
  "precision": 1.0,
  "recall": 0.8444767441860465,
  "f1": 0.9156816390858944,
  "mcc": 0.8755077584478814,
  "brier": 0.2812807493542284,
  "ece": 0.2928933244199287,
  "confusion_matrix": [
    [
      1052,
      0
    ],
    [
      107,
      581
    ]
  ],
  "pr_auc": 0.9489667871294735,
  "roc_auc": 0.9596318474224069
}
```
### XGB vs RF comparison
```json
{
  "paired_block_bootstrap_f1_diff": {
    "metric": "f1_diff (A - B)",
    "mean": -0.20616547202722224,
    "ci_low": -0.413466571580822,
    "ci_high": -0.056450269714356925,
    "n_valid_resamples": 1000
  },
  "mcnemar": {
    "b_A_wrong_B_right": 489,
    "c_A_right_B_wrong": 106,
    "p_value": 1.0169963873968638e-59,
    "caveat": "rows are autocorrelated within one lifecycle; effective sample size is inflated, so this p-value is optimistic"
  }
}
```

## Horizon h60

### XGBoost (primary) — test metrics
```json
{
  "threshold": 0.9675849980342632,
  "balanced_accuracy": 0.7735583186607722,
  "precision": 0.608843537414966,
  "recall": 0.9972144846796658,
  "f1": 0.7560718057022175,
  "mcc": 0.5754722399263497,
  "brier": 0.27773535387333503,
  "ece": 0.28372614625256964,
  "confusion_matrix": [
    [
      562,
      460
    ],
    [
      2,
      716
    ]
  ],
  "pr_auc": 0.9711915187027002,
  "roc_auc": 0.9757459838974321
}
```
### Bootstrap 95% CIs (moving-block)
```json
{
  "f1": {
    "ci_low": 0.5610811604490035,
    "ci_high": 0.8753808613778316,
    "n_valid_resamples": 1000
  },
  "balanced_accuracy": {
    "ci_low": 0.6683690340167753,
    "ci_high": 0.8590571384760756,
    "n_valid_resamples": 1000
  },
  "mcc": {
    "ci_low": 0.3943229976722436,
    "ci_high": 0.7310925257126941,
    "n_valid_resamples": 1000
  },
  "pr_auc": {
    "ci_low": 0.9011381328608765,
    "ci_high": 1.0,
    "n_valid_resamples": 1000
  },
  "roc_auc": {
    "ci_low": 0.9274173870603455,
    "ci_high": 1.0,
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
  "onset_row_in_window": 1022,
  "alarm_decision_row_in_window": 489,
  "delay_rows": -533,
  "consecutive_rule": 3,
  "alarm_convention": "delay measured at the k-th confirmation row",
  "cleared_alarm_episodes_before_onset": 4,
  "n_events": 1,
  "note": "single onset event: no CI on delay; see sensitivity table",
  "delay_minutes": -31.93
}
```
### RandomForest baseline — test metrics
```json
{
  "threshold": 0.960631808006845,
  "balanced_accuracy": 0.9052924791086351,
  "precision": 1.0,
  "recall": 0.8105849582172702,
  "f1": 0.8953846153846153,
  "mcc": 0.8458053705093309,
  "brier": 0.249796353430554,
  "ece": 0.26172804381841364,
  "confusion_matrix": [
    [
      1022,
      0
    ],
    [
      136,
      582
    ]
  ],
  "pr_auc": 0.9378684045000791,
  "roc_auc": 0.9455666152445639
}
```
### XGB vs RF comparison
```json
{
  "paired_block_bootstrap_f1_diff": {
    "metric": "f1_diff (A - B)",
    "mean": -0.16419842309543953,
    "ci_low": -0.3625254387676742,
    "ci_high": -0.0027478790197569894,
    "n_valid_resamples": 1000
  },
  "mcnemar": {
    "b_A_wrong_B_right": 460,
    "c_A_right_B_wrong": 134,
    "p_value": 9.094128427893094e-43,
    "caveat": "rows are autocorrelated within one lifecycle; effective sample size is inflated, so this p-value is optimistic"
  }
}
```

## Caveats (mandatory in any publication)
- Single filter lifecycle: all results describe one clogging episode on one rig; cross-filter generalization is untested and unclaimed.
- Labels are generated by the thesis's deterministic multi-criteria rule, not by independent ground truth (e.g. visual inspection); supervised results therefore measure how well the model reproduces/anticipates the rule from causal sensor features.
- Held-out-transition split (S2): the clogged training segment post-dates the test window, so S2 evaluates interpolation across a held-out transition, not pure forecasting.
- Detection delay is estimated from a single onset event; no confidence interval is possible (sensitivity across alarm rules reported instead).
- Block-bootstrap CIs quantify within-lifecycle sampling noise only.