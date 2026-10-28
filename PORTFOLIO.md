# Portfolio: Bitcoin Multi-Horizon LSTM Forecasting

## Project summary

Built a multivariate time-series forecasting project to predict the next 24 hourly Bitcoin closing prices using 24 past observations and six market features. Implemented a direct LSTM baseline, a teacher-forced Seq2Seq encoder-decoder, and a larger bidirectional/stacked Seq2Seq experiment in TensorFlow/Keras. Used chronological data splits, training-only normalization, a custom serializable dense layer, and custom gradient-based training loops. In the notebook's recorded test evaluation, the standard Seq2Seq model achieved an MAE of 325.43 USDT and RMSE of 462.72 USDT, improving on the LSTM baseline by approximately 9.67% and 6.53%, respectively.

## Short project-card copy

**Bitcoin Multi-Horizon Forecasting**

Compared three LSTM architectures for 24-hour Bitcoin price forecasting. Implemented custom TensorFlow training loops and autoregressive Seq2Seq inference; the standard Seq2Seq model reduced recorded test MAE by 9.67% against a direct LSTM baseline.

## Outputs to showcase

| Priority | Output | What to explain |
| --- | --- | --- |
| 1 | [Forecast comparisons](predictions_vs_actual_plot.png) | Show the 24-hour input history and forecast horizon. Explain where each model follows or diverges from the actual prices. The "Best" legend refers to an experiment name, not the strongest result. |
| 2 | [Recorded evaluation table in the README](README.md#recorded-results) | Compare MAE and RMSE in USDT. Lead with the standard Seq2Seq model's improvement over the baseline and include the unsuccessful larger experiment. |
| 3 | [Full notebook](muhammad%20akhtar%20ariq_submission_akhir_dltm.ipynb) | Demonstrate the workflow from chronological splitting and train-only scaling to custom training and autoregressive evaluation. |
| 4 | [Training curves](accuracy_loss_plot.png) | Discuss convergence and why teacher-forced validation can differ from autoregressive test performance. These are loss curves, not accuracy curves. |
| 5 | [Correlation heatmap](correlation_heatmap.png) | Introduce the six inputs and their relationships. Correlation alone does not establish forecasting value or causation. |
| 6 | Saved `.keras` models | Link the standard Seq2Seq artifact as the strongest recorded model, alongside the other two experiments. Explain the custom-layer, scaler, and inference requirements. |

Use the forecast comparison as the main visual, followed by the metrics table. Keep the complete comparison visible so the case study reflects the actual results.

## Technical contributions to describe

- Prepared chronological 80/10/10 splits and fitted feature/target scalers only on the training partition.
- Converted six-variable histories into 24-step input and target windows.
- Implemented and serialized a custom dense layer with trainable weights and biases.
- Built custom training loops with `tf.GradientTape`, graph-compiled training steps, and best-validation-weight restoration.
- Compared direct prediction with teacher-forced training and autoregressive Seq2Seq inference.
- Evaluated forecasts after inverse scaling and visualized results across three selected test windows.

## Technologies used

| Technology | Use in this project |
| --- | --- |
| Python 3.12 | Notebook programming language; metadata records 3.12.4 |
| TensorFlow | Automatic differentiation, `tf.function`, tensor operations, Adam optimization, and MSE loss; saved output reports 2.21.0 |
| Keras / `tf.keras` | Sequential and Functional model APIs, LSTM, Bidirectional, Dropout, TimeDistributed, custom layers, and `.keras` serialization; archives record Keras 3.14.1 |
| NumPy | Sliding windows, array reshaping, batching, and numerical calculations |
| pandas | CSV loading, date parsing, chronological splits, and comparison tables |
| scikit-learn | `MinMaxScaler`, MAE, and MSE calculations used to derive RMSE |
| Matplotlib | Training curves and forecast visualizations |
| Seaborn | Correlation heatmap |
| Jupyter Notebook / IPython | Interactive project narrative, execution, saved outputs, and table display |
| Git / GitHub | Version control and project hosting prepared for this portfolio release |

## Honest framing and follow-up work

Describe this as an experimental forecasting comparison. The notebook provides no evidence of live deployment, trading returns, or production monitoring. The original CSV is absent, metrics are from saved outputs, and seeds are not fixed. The larger experiment's weak autoregressive results are an opportunity to discuss evaluation and debugging: its second decoder layer currently discards recurrent states between forecast steps. Follow-up work could address that inference path, add autoregressive validation and a persistence baseline, report errors by horizon, and export scalers with a complete inference interface.
