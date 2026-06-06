# Experiment Notes

The experiments use a calibration-and-holdout split.

Calibration rows are used to fit or choose the hybrid score.

Holdout rows are used to evaluate the result.

Average Precision is the main metric because anomaly detection is usually imbalanced. F1-style numbers are useful for reading the results, but the main comparison is ranking quality.

The included replay is lightweight. It starts from prepared score tables and does not rerun ADAMM, GAMA, or raw event-log preprocessing.
