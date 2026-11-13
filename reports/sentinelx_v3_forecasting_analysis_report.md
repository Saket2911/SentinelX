# SentinelX V3 — Predictive World Model Forecasting Analysis

## 1. Executive summary

The SentinelX V3 LSTM World Model was evaluated for numerical future network-state prediction using recursive forecasts at 1, 2, 5, 10 and 30 seconds.
The model was compared with a persistence baseline, which predicts the future state by repeating the most recent observed state.
The feature-wise analysis covered 209 model features derived from 71 original network features, grouped into six feature families.
All 20 aggregate comparisons between the feature-wise experiment and the original multi-step forecasting results passed verification.

## 2. Model and evaluation configuration

- Model: SentinelX V3 LSTM World Model
- Checkpoint: `best_model.pt`
- Best checkpoint epoch: 11
- Input sequence length: 10
- Input features: 209
- Hidden size: 128
- LSTM layers: 1
- Dropout: 0.2
- Output features: 209
- Training loss: Huber
- Forecasting method: recursive prediction
- Forecast horizons: 1, 2, 5, 10 and 30 seconds
- Evaluation device for this analysis: CPU
- Common forecast origins: 31,377
- Test sequence count: 31,406

The test partition had previously been used for exploratory evaluation. These results are therefore not an untouched independent test evaluation.

## 3. Recursive multi-step aggregate metrics

Source: `multi_step_forecasting_v3_fixed/multi_step_metrics.csv`

```text
 Horizon_Seconds Target_Label  Valid_Origins  LSTM_Huber  Persistence_Huber  LSTM_MAE  Persistence_MAE  MAE_Improvement_Percent  LSTM_RMSE  Persistence_RMSE  RMSE_Improvement_Percent
               1          ALL          31377     0.12070            0.17708   0.32065          0.35624                  9.99002    0.53138           0.67908                  21.74929
               1       Benign          11189     0.12579            0.18694   0.32456          0.36515                 11.11523    0.54627           0.70378                  22.37945
               1          Bot          20187     0.11781            0.17153   0.31841          0.35121                  9.33838    0.52272           0.66470                  21.35969
               1     NO_FLOWS              1     1.51136            2.03307   1.95784          2.42884                 19.39195    2.24289           2.88947                  22.37714
               2          ALL          31376     0.12751            0.19026   0.32902          0.37584                 12.45555    0.54702           0.70423                  22.32487
               2       Benign          11188     0.13309            0.20219   0.33347          0.38827                 14.11375    0.56323           0.73217                  23.07380
               2          Bot          20187     0.12436            0.18360   0.32649          0.36890                 11.49460    0.53765           0.68811                  21.86559
               2     NO_FLOWS              1     1.28719            1.16634   1.71939          1.42712                -20.47922    1.98307           2.18842                   9.38348
               5          ALL          31373     0.14384            0.20796   0.35180          0.40455                 13.03863    0.58297           0.73597                  20.78905
               5       Benign          11185     0.14850            0.21990   0.35428          0.41520                 14.67223    0.59680           0.76399                  21.88384
               5          Bot          20187     0.14123            0.20123   0.35040          0.39854                 12.07846    0.57511           0.71960                  20.07966
               5     NO_FLOWS              1     0.61641            2.36929   1.02323          2.76717                 63.02241    1.28686           3.39091                  62.04968
              10          ALL          31368     0.15427            0.22108   0.36618          0.42498                 13.83731    0.60518           0.75901                  20.26768
              10       Benign          11180     0.15891            0.23242   0.36826          0.43493                 15.32898    0.61846           0.78608                  21.32351
              10          Bot          20187     0.15168            0.21469   0.36501          0.41936                 12.95915    0.59765           0.74319                  19.58376
              10     NO_FLOWS              1     0.37728            2.42715   0.63715          2.83776                 77.54730    1.18894           3.54576                  66.46871
              30          ALL          31348     0.16938            0.23935   0.38446          0.45352                 15.22775    0.63773           0.78951                  19.22371
              30       Benign          11160     0.17494            0.25123   0.38815          0.46350                 16.25683    0.65174           0.81826                  20.35075
              30          Bot          20187     0.16621            0.23268   0.38231          0.44791                 14.64381    0.62956           0.77291                  18.54659
              30     NO_FLOWS              1     2.04103            2.04810   2.49603          2.43510                 -2.50216    2.82488           2.86318                   1.33757
```

These metrics are measured in preprocessed model feature space, not raw network units.

## 4. Feature-wise forecasting

- Feature-wise records: 4,180
- Unique model features: 209
- Forecast horizons: 5
- Label groups in the source experiment: ALL, Benign, Bot, NO_FLOWS

For three CWE Flag Count features, persistence MAE is exactly zero. Their percentage improvement is undefined, although their absolute errors remain available. They must not be assigned an artificial percentage.

## 5. Feature-family organization

                       Feature_Family  Feature_Count
               Packet and flow counts             22
           Packet lengths and headers             63
Protocol and network-state indicators             16
                 Rates and throughput             12
                            TCP flags             27
                               Timing             69

The six families are:
1. Timing
2. Packet lengths and headers
3. Packet and flow counts
4. TCP flags
5. Rates and throughput
6. Protocol and network-state indicators

## 6. Overall feature-family forecasting results

 Horizon_Seconds                        Feature_Family  Feature_Count  LSTM_Better_Count  Persistence_Better_Count  LSTM_Mean_MAE  Persistence_Mean_MAE  MAE_Improvement_Percent
               1                Packet and flow counts             22                 22                         0        0.29122               0.36333                 19.84504
               1            Packet lengths and headers             63                 42                        21        0.26345               0.27368                  3.73711
               1 Protocol and network-state indicators             16                 13                         3        0.25729               0.27763                  7.32720
               1                  Rates and throughput             12                 12                         0        0.43013               0.49204                 12.58203
               1                             TCP flags             27                  7                        20        0.37478               0.34798                 -7.70310
               1                                Timing             69                 69                         0        0.35674               0.42722                 16.49647
               2                Packet and flow counts             22                 21                         1        0.29733               0.37912                 21.57219
               2            Packet lengths and headers             63                 48                        15        0.27144               0.29076                  6.64346
               2 Protocol and network-state indicators             16                 13                         3        0.26472               0.29557                 10.43532
               2                  Rates and throughput             12                 12                         0        0.43898               0.51562                 14.86354
               2                             TCP flags             27                 13                        14        0.38934               0.37330                 -4.29715
               2                                Timing             69                 69                         0        0.36389               0.44777                 18.73246
               5                Packet and flow counts             22                 21                         1        0.31764               0.41464                 23.39415
               5            Packet lengths and headers             63                 52                        11        0.29342               0.32243                  8.99834
               5 Protocol and network-state indicators             16                 13                         3        0.28517               0.31933                 10.69961
               5                  Rates and throughput             12                 12                         0        0.45683               0.55014                 16.96229
               5                             TCP flags             27                 13                        14        0.40600               0.41174                  1.39456
               5                                Timing             69                 69                         0        0.39199               0.46794                 16.23187
              10                Packet and flow counts             22                 21                         1        0.33140               0.43910                 24.52570
              10            Packet lengths and headers             63                 53                        10        0.30655               0.33924                  9.63692
              10 Protocol and network-state indicators             16                 13                         3        0.29890               0.33109                  9.72273
              10                  Rates and throughput             12                 12                         0        0.48077               0.57124                 15.83689
              10                             TCP flags             27                 15                        12        0.41723               0.43622                  4.35268
              10                                Timing             69                 69                         0        0.40740               0.49071                 16.97732
              30                Packet and flow counts             22                 21                         1        0.34563               0.47068                 26.56863
              30            Packet lengths and headers             63                 53                        10        0.32119               0.36709                 12.50425
              30 Protocol and network-state indicators             16                  9                         7        0.32713               0.34301                  4.62856
              30                  Rates and throughput             12                  9                         3        0.56125               0.60864                  7.78607
              30                             TCP flags             27                 16                        11        0.41724               0.46970                 11.16792
              30                                Timing             69                 65                         4        0.42433               0.51928                 18.28548

Interpretation:
- Timing features show positive macro MAE improvement at all horizons.
- Packet and flow counts show strong positive improvement at all horizons.
- Rates and throughput improve at all horizons, with a smaller advantage at 30 seconds.
- TCP flags are weaker than persistence at 1 and 2 seconds, then show positive family-level improvement at longer horizons.
- Protocol and network-state indicators improve at all horizons, but their advantage is small at 30 seconds.

## 7. Benign versus Bot feature-family comparison

Source: `feature_family_benign_bot_comparison_v1.csv`

 Horizon_Seconds                        Feature_Family  MAE_Improvement_Percent_Benign  MAE_Improvement_Percent_Bot  Bot_Minus_Benign_Improvement
               1                Packet and flow counts                        21.98491                     18.66517                      -3.31973
               1            Packet lengths and headers                         4.85026                      3.08408                      -1.76619
               1 Protocol and network-state indicators                         9.17313                      6.08963                      -3.08350
               1                  Rates and throughput                        17.18183                      8.34994                      -8.83189
               1                             TCP flags                         2.02047                    -14.68571                     -16.70618
               1                                Timing                        14.32233                     17.53516                       3.21283
               2                Packet and flow counts                        24.22402                     20.10517                      -4.11885
               2            Packet lengths and headers                         8.10938                      5.79713                      -2.31224
               2 Protocol and network-state indicators                        12.49026                      9.06393                      -3.42633
               2                  Rates and throughput                        19.62134                     10.37705                      -9.24428
               2                             TCP flags                         7.33925                    -12.83717                     -20.17642
               2                                Timing                        16.71127                     19.70612                       2.99485
               5                Packet and flow counts                        24.84137                     22.59924                      -2.24213
               5            Packet lengths and headers                         8.47421                      9.26314                       0.78893
               5 Protocol and network-state indicators                        14.09003                      8.42376                      -5.66627
               5                  Rates and throughput                        19.83666                     14.28945                      -5.54721
               5                             TCP flags                        11.22347                     -5.55068                     -16.77414
               5                                Timing                        16.00030                     16.33512                       0.33482
              10                Packet and flow counts                        26.30508                     23.55380                      -2.75128
              10            Packet lengths and headers                         8.72427                     10.11896                       1.39468
              10 Protocol and network-state indicators                        14.26230                      6.64940                      -7.61290
              10                  Rates and throughput                        18.20518                     13.61458                      -4.59060
              10                             TCP flags                        14.15725                     -2.57887                     -16.73613
              10                                Timing                        16.50143                     17.19306                       0.69163
              30                Packet and flow counts                        28.61271                     25.49806                      -3.11466
              30            Packet lengths and headers                         8.80872                     14.55953                       5.75081
              30 Protocol and network-state indicators                        10.98597                      0.36712                     -10.61885
              30                  Rates and throughput                        12.99621                      3.00714                      -9.98908
              30                             TCP flags                        18.56847                      6.02646                     -12.54201
              30                                Timing                        18.66695                     18.10682                      -0.56014

Key observations:
- Packet and flow counts improve over persistence for both labels at every horizon.
- Timing remains a strong family for both labels.
- TCP-flag forecasting is substantially weaker for Bot-labelled samples at shorter horizons; its family-level improvement becomes positive at 30 seconds.
- At 30 seconds, packet-length and header features show a larger relative improvement for Bot than for Benign samples.
- Rates and throughput show a smaller advantage for Bot samples, particularly at 30 seconds.

These are within-label forecasting comparisons. They do not establish that a traffic label caused a prediction difference or that a feature is an attack signature.

## 8. Methodology and verification

- The model architecture and checkpoint were preserved.
- Recursive predictions were fed into subsequent forecast windows.
- A common set of forecast origins was used across horizons.
- Paths crossing segment boundaries or timestamp gaps were excluded.
- Feature order was taken from the trained checkpoint.
- Feature-family assignment covered all 209 features exactly once.
- Family metrics are macro-averages across features; each feature receives equal weight within its family.
- The feature-wise aggregate metrics were checked against the original multi-step results.
- Aggregate verification: 20/20 PASS.

## 9. Limitations

1. Metrics are in preprocessed model feature space, not raw bytes, packets, seconds or network rates.
2. Test data was previously used for exploratory evaluation; results are not independent untouched test evidence.
3. The test partition contains Benign and Bot labels. Comparisons between these groups may reflect traffic composition and capture conditions as well as label differences.
4. Feature-family averages can conceal individual feature weaknesses.
5. Recursive prediction errors can accumulate over longer horizons.
6. Numerical state forecasting does not establish attack-detection accuracy, early-warning performance or attack-stage identification.
7. SHAP explanations, residuals and feature errors describe model behavior; they are not causal evidence or definitive attack indicators.

## 10. Conclusion

The V3 LSTM World Model demonstrates lower aggregate forecasting errors than the persistence baseline across the evaluated recursive forecast horizons.
Feature-level analysis shows that performance varies across network feature families. Timing and packet/flow-count features are consistently favorable, while TCP-flag forecasting is more challenging, particularly for Bot-labelled samples at short horizons.
These findings characterize the current model's future-state prediction behavior. They do not, by themselves, demonstrate attack detection or prevention.

## 11. Reproducibility and artifact locations

- Model checkpoint: `Models/lstm_world_model_v3_gpu_run2/best_model.pt`
- Multi-step metrics: `Models/lstm_world_model_v3_gpu_run2/multi_step_forecasting_v3_fixed/multi_step_metrics.csv`
- Feature-wise metrics: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/feature_wise_metrics.csv`
- Aggregate verification: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/aggregate_verification.csv`
- Feature-family mapping: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/feature_family_mapping_v1.csv`
- Feature-family summary: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/feature_family_summary_v1.csv`
- Label-wise summary: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/feature_family_label_summary_v1.csv`
- Benign/Bot comparison: `Models/lstm_world_model_v3_gpu_run2/multi_step_feature_analysis_v1/feature_family_benign_bot_comparison_v1.csv`

Report generated from existing verified experiment artifacts. No model retraining or architecture modification was performed.