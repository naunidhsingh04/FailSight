/** One source for the "?" tooltips and the Glossary page. Plain words first (about grade 8); `more` is the technical detail. */
export interface Entry { term: string; plain: string; more?: string }

export const GLOSSARY: Record<string, Entry> = {
  learning: {
    term: 'Learning what normal looks like',
    plain: 'When it starts, FailSight just watches the shop for about two minutes to see what a normal day looks like. Later, anything very different counts as a sign of trouble.',
    more: 'The first 150 seconds of an app are the learning period: no alerts are raised, and each reading’s normal value is taken from it.',
  },
  recorded: {
    term: 'Recorded benchmark data',
    plain: 'Data that was recorded earlier from two sample applications. It has no exact start or end time for each problem, so it cannot prove that a problem is noticed quickly.',
  },
  window: {
    term: 'Window',
    plain: 'A 30-second slice of recent activity. A new window is made every 10 seconds, so neighbouring windows overlap.',
    more: 'Each window is summarised into 155 numbers: 8 about API requests, 3 about log lines and 144 about system metrics.',
  },
  risk: {
    term: 'Failure risk',
    plain: 'The model’s estimate, from 0% to 100%, that something is going wrong right now. It describes the present, not the future.',
    more: 'It is the model’s predicted probability of the Anomaly class for the current window. The system does not forecast when a failure will happen.',
  },
  alertline: {
    term: 'Alert line',
    plain: 'The risk level at which an alert is raised. It was chosen using practice data only, never the test data.',
    more: 'Saved in thresholds.json per model, tuned on out-of-fold predictions of the training part. For the default model it is a very high 0.99 probability.',
  },
  alert: { term: 'Alert', plain: 'A window whose risk crossed the alert line. Alerts are checked against what the recording says really happened.' },
  correctalert: { term: 'Correct alert', plain: 'An alert raised while a failure was really happening (according to the recording).' },
  falsealarm: { term: 'False alarm', plain: 'An alert raised while everything was normal (according to the recording).' },
  missed: { term: 'Missed failure', plain: 'A window where a failure was really happening but no alert was raised.' },
  nodata: { term: 'Not enough data to judge', plain: 'When more than half of the readings a system normally sends are missing in a window, the system does not guess. It raises no alert for that window and counts it separately.' },
  recording: {
    term: 'Known from the recording',
    plain: 'The data comes from a recorded experiment in which the failures were created on purpose, so we know the right answer for every window. A real system would not know this.',
  },
  replay: {
    term: 'Replay',
    plain: 'We send recorded data through the system as if it were happening now. It is a demonstration, not a connection to a real production system.',
    more: 'scripts/replay_scenario.py reads record-level events of test-split blocks from AnoMod (read-only) and posts them to the ingestion API at 1x, 10x or 60x.',
  },
  ingestion: { term: 'Ingestion', plain: 'The door through which traffic records, log lines and measurements enter the system.', more: 'POST /ingest/api, /ingest/logs and /ingest/metrics, single events or batches, with validation of every event.' },
  collector: { term: 'Collector', plain: 'A small program next to your application that gathers its records and forwards them. In the recorded-data replay, the replay script plays that role.' },
  normalrecall: {
    term: 'Normal windows left alone',
    plain: 'Of all the windows that were really normal, the share that did not trigger an alert. 55.6% means that 44.4% of normal windows caused a false alarm.',
    more: 'Normal recall (recall of the Normal class) on the test set. It is the weak spot of every model here.',
  },
  balancedaccuracy: {
    term: 'Balanced score',
    plain: 'The average of two things: how many failures were caught and how many normal windows were left alone. 0.5 is no better than guessing, 1.0 is perfect.',
    more: 'Balanced accuracy = mean recall over the classes.',
  },
  f1: { term: 'F1 score', plain: 'One number that combines how many alerts were right with how many failures were caught. Higher is better, 1.0 is perfect.', more: 'Macro-F1: the F1 score averaged over the classes.' },
  rocauc: {
    term: 'ROC-AUC',
    plain: 'How well the risk score puts failure windows above normal ones, ignoring where the alert line is. 0.5 is a coin flip.',
    more: 'Area under the receiver operating characteristic curve. On the LO2 test it was high for some models although they detected nothing, because log volume alone separates the classes.',
  },
  smote: { term: 'SMOTE', plain: 'A way to create extra practice examples of rare cases (like normal windows) so the model learns from them. Used on practice data only.', more: 'Synthetic Minority Over-sampling Technique, applied inside each training fold only.' },
  shap: { term: 'SHAP', plain: 'A method that shows how much each signal pushed the risk up or down for one window.', more: 'SHapley Additive exPlanations; for trees, exact Shapley values (TreeExplainer). Base value plus all contributions equals the model output.' },
  lime: { term: 'LIME', plain: 'A second way to explain one window: it tries small changes and sees which signals matter. We compare it with SHAP.', more: 'Local Interpretable Model-agnostic Explanations.' },
  confidenceinterval: { term: 'Confidence interval', plain: 'A range that shows how much a score could change if the test had been slightly different. A wide range means the score is uncertain.', more: '95% intervals from 1000 block-bootstrap resamples of the test windows.' },
  testset: { term: 'Test set', plain: 'Windows kept locked away while the models were built. Each model is scored on them once, at the end.', more: '647 windows (20%), never used for fitting, tuning or choosing thresholds.' },
  trainset: { term: 'Practice data', plain: 'The windows (80%) the models learn from and are tuned on.', more: '2,549 windows.' },
  overfitting: { term: 'Overfitting', plain: 'When a model memorises its practice data and does worse on new data. Several models showed this and were retrained more gently.' },
  leakage: { term: 'Data leakage', plain: 'When a model gets a peek at answers it should not have, which makes results look better than they are. We ran checks for it and report what they found.' },
  block: { term: 'Block', plain: 'Ten neighbouring windows from the same experiment. Blocks, not single windows, were split between practice and test data.' },
  crossvalidation: { term: 'Cross-validation', plain: 'Practice data is cut into 10 parts; the model learns on 9 and is checked on the 10th, ten times over.', more: 'Run-aware: windows of the same block never sit in two folds.' },
  counter: { term: 'Counter rate', plain: 'Some measurements only ever count upwards (like total CPU time). We turn them into “per second” values by comparing two readings, which is why the first windows of a replay are a warm-up.' },
  lo2: { term: 'LO2', plain: 'A different dataset (a login service) used as an outside test. The models did not work on it: see the Limits page.' },
  anomod: { term: 'AnoMod', plain: 'The recorded dataset behind this project: two sample applications (Social Network and Train Ticket) with failures created on purpose.' },
  otel: { term: 'OpenTelemetry', plain: 'A common standard for collecting traces, metrics and logs from applications. A real deployment would use it to feed this system.' },
  prometheus: { term: 'Prometheus', plain: 'A popular tool that measures systems (CPU, memory, disk) every few seconds. Its numbers are what the “metric” signals are.' },
  fluentbit: { term: 'Fluent Bit', plain: 'A light program that reads log files and forwards the lines. A real deployment could use it to send logs to the ingestion API.' },
  calibration: { term: 'Calibration', plain: 'Whether “90% risk” really means failure about 9 times out of 10.' },
  sse: { term: 'Live updates', plain: 'The page receives each new result the moment it is ready, over one open connection (Server-Sent Events), without refreshing.' },
}

export const GLOSSARY_KEYS = Object.keys(GLOSSARY).sort((a, b) => GLOSSARY[a].term.localeCompare(GLOSSARY[b].term))
