import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import ResponsiveTable from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { fmt, int, pct, range } from '../format'
import { Detailed, Simple } from '../mode'
import Term from '../components/Term'
import { usePageTitle } from '../usePageTitle'
import type { DataResponse, Lo2Model, Lo2Response, ModelInfo, ModelsResponse } from '../types'

interface Masking { unmasked: number; api70: number; log70: number; metric70: number; all70: number; normal_recall_unmasked: number; normal_recall_metric70: number }
interface ProofReport { xgb_binary: Masking }
const maskingOf = (r: ProofReport): Masking | null => r.xgb_binary ?? null

export default function Limits() {
  usePageTitle('Limits')
  const models = useApi<ModelsResponse>('/api/models')
  const lo2 = useApi<Lo2Response>('/api/lo2')
  const data = useApi<DataResponse>('/api/data')
  const proofs = useApi<ProofReport>('/api/proofs/masking')
  return (
    <>
      <h1>Limits</h1>
      <p className="lede">What FailSight cannot do. Nothing on this page is softened: the numbers come straight from the result files.</p>
      <Simple><PlainLimits models={models} lo2={lo2} proofs={proofs} /></Simple>
      <Detailed>
      <Async level={2} state={models} rows={5} label="Loading model results">{(m) => <NormalRecall models={m.models.filter((x) => x.label_version === 'label_binary')} all={m.models} data={data.data} />}</Async>
      <Async level={2} state={lo2} rows={6} label="Loading LO2 results">{(l) => <External l={l} />}</Async>
      <Async level={2} state={proofs} rows={2} label="Loading the masking check">{(p) => <Reliance m={maskingOf(p)} />}</Async>
      <Async level={2} state={data} rows={3} label="Loading other limits">{(d) => <Other d={d} />}</Async>
      </Detailed>
    </>
  )
}

function NormalRecall({ models, all, data }: { models: ModelInfo[]; all: ModelInfo[]; data: DataResponse | null }) {
  const ranked = useMemo(() => [...models].sort((a, b) => (b.metrics['official/ALL']?.balanced_accuracy ?? 0) - (a.metrics['official/ALL']?.balanced_accuracy ?? 0)), [models])
  const best = ranked[0]
  const bm = best.metrics['official/ALL']
  const nr = bm.normal_recall ?? 0
  const ba = bm.balanced_accuracy ?? 0
  const support = bm.support_Normal ?? 0
  const ci = best.ci['ALL/normal_recall']
  const reach = models.filter((m) => (m.metrics['official/ALL']?.normal_recall ?? 0) >= 0.7).length
  const tuned = best.metrics['tuned_threshold/ALL']
  const lvBest = [...all.filter((x) => x.label_version === 'label_level')].sort((a, b) => (b.metrics['official/ALL']?.balanced_accuracy ?? 0) - (a.metrics['official/ALL']?.balanced_accuracy ?? 0))[0]
  const err = data?.errors ?? []
  const cnt = (lab: string, v: string) => err.find((e) => e.label_version === lab && e.breakdown === 'counts' && e.value === v)?.n_misclassified_normal
  const pred = (lab: string) => err.filter((e) => e.label_version === lab && e.breakdown === 'predicted_class').sort((a, b) => b.n_misclassified_normal - a.n_misclassified_normal)
  return (
    <section aria-labelledby="nr-h">
      <h2 id="nr-h" style={{ marginTop: 0 }}>1. Normal windows are the weak spot</h2>
      <div className="grid cols-3">
        <div className="stat"><div className="num">{fmt(nr)}</div><div className="cap">Normal recall of the best binary model ({best.name}) on {support} Normal test windows; 95% CI {ci ? range(ci.low, ci.high) : '–'}</div></div>
        <div className="stat"><div className="num">{fmt(ba)}</div><div className="cap">its balanced accuracy, which is carried by anomaly recall of {fmt(2 * ba - nr)} (derived: 2 x balanced accuracy - Normal recall)</div></div>
        <div className="stat"><div className="num">{reach} of {models.length}</div><div className="cap">binary models reach Normal recall of 0.70 or more on the test set</div></div>
      </div>
      <div className="callout" style={{ marginTop: 'var(--space-5)' }}>
        <span className="label">What this means</span>
        <ul>
          <li>The best binary model misses about {pct(1 - nr, 0)} of normal windows, calling them anomalies. As an alert system it would raise many false alarms.</li>
          {best.decision_rule && tuned && <li>The saved decision rule ({best.decision_rule.type === 'threshold' ? `Anomaly when P(Anomaly) ≥ ${fmt(best.decision_rule.threshold, 2)}` : 'class weights'}) reached Normal recall {fmt(best.decision_rule.oof_normal_recall)} on out-of-fold predictions but {fmt(tuned.normal_recall)} on the test set: tuning on out-of-fold data did not carry over.</li>}
          {cnt('label_binary', 'misclassified') !== undefined && <li>Binary: {cnt('label_binary', 'misclassified')} of {cnt('label_binary', 'normal_test_windows')} Normal test windows are misclassified (as Anomaly). Five-class ({lvBest ? 'best single tree model, LGBM_tuned' : ''}): {cnt('label_level', 'misclassified')} of {cnt('label_level', 'normal_test_windows')}, predicted as {pred('label_level').map((p) => `${p.value} ${p.n_misclassified_normal}`).join(', ')}.</li>}
          <li>The Normal test windows come from only two runs (one per system), so even this rate is uncertain; the interval above says so.</li>
        </ul>
      </div>
      <ResponsiveTable<ModelInfo>
        caption="Binary models: Normal recall on the test set (95% block-bootstrap CI)"
        rows={ranked} rowKey={(r) => r.name} initialSort={{ key: 'nr', dir: 'desc' }}
        columns={[
          { key: 'm', label: 'Model', nameCell: true, render: (r) => r.name, sortValue: (r) => r.name },
          { key: 'nr', label: 'Normal recall', numeric: true, render: (r) => fmt(r.metrics['official/ALL']?.normal_recall), sortValue: (r) => r.metrics['official/ALL']?.normal_recall ?? null },
          { key: 'ci', label: '95% CI', numeric: true, render: (r) => range(r.ci['ALL/normal_recall']?.low, r.ci['ALL/normal_recall']?.high) },
          { key: 'tn', label: 'With saved threshold', numeric: true, render: (r) => fmt(r.metrics['tuned_threshold/ALL']?.normal_recall), sortValue: (r) => r.metrics['tuned_threshold/ALL']?.normal_recall ?? null },
        ]}
      />
    </section>
  )
}

function External({ l }: { l: Lo2Response }) {
  const a = l.feature_mapping.A_logs_only
  const b = l.feature_mapping.B_logs_plus_node_metrics
  const maxBA = Math.max(...l.models.map((m) => m.balanced_accuracy))
  const allAnom = l.models.filter((m) => m.share_predicted_anomaly === 1).length
  const allNorm = l.models.filter((m) => m.share_predicted_anomaly === 0).length
  const lv = l.log_volume_check
  const lc = l.distribution_shift.find((r) => r.feature === 'log_line_count' && r.lo2_class === 'Normal')
  const hiAuc = l.models.filter((m) => m.roc_auc >= 0.9)
  const otherSettings = l.all_settings.filter((s) => !(s.feature_set === 'A_logs_only' && s.system_flag === 'TT'))
  return (
    <section aria-labelledby="lo2-h">
      <h2 id="lo2-h">2. External test on LO2: the models did not transfer</h2>
      <p>LO2 is a different system (light-oauth2 micro-services) with mostly logs. The saved models were applied unchanged to windows built from {l.sample.scenarios} whole scenarios of one run ({l.sample.megabytes.toFixed(1)} MB). Nothing was trained or tuned on LO2.</p>
      <div className="grid cols-4">
        <div className="stat"><div className="num">{fmt(maxBA)}</div><div className="cap">balanced accuracy of every one of the {l.models.length} saved models (chance level)</div></div>
        <div className="stat"><div className="num">{a.used_features_built} of {a.used_features_total}</div><div className="cap">model input features that can be built from LO2 logs; the rest are missing</div></div>
        <div className="stat"><div className="num">{l.sample.windows}</div><div className="cap">windows: {l.sample.normal_windows} Normal (one scenario) and {l.sample.anomaly_windows} Anomaly</div></div>
        <div className="stat"><div className="num">{allAnom} + {allNorm}</div><div className="cap">models that label every window Anomaly + models that label every window Normal</div></div>
      </div>
      <div className="callout" style={{ marginTop: 'var(--space-5)' }}>
        <span className="label">The mapping loses most features</span>
        <p>Only {a.features_built} of {a.features_total} AnoMod features can be built from LO2 ({a.used_features_built} of the {a.used_features_total} the models use). All API features are missing, and with logs only every metric feature is missing too. An extra variant that adds the node metrics shared with AnoMod builds {b.used_features_built} of {b.used_features_total} model features ({pct(b.used_features_built / b.used_features_total, 0)}); every model still predicted Anomaly for every window under that variant too, with either system flag.</p>
        <p>{lc && <>The three log features also sit far outside the training range: the median LO2 Normal window has {int(lc.lo2_median)} log lines against a TrainTicket training median of {int(lc.anomod_train_median)}.</>}</p>
      </div>

      <h3>The high ROC-AUCs are not evidence of detection</h3>
      <p>
        Balanced accuracy is exactly {fmt(maxBA)}, yet {hiAuc.length} of {l.models.length} models have a ROC-AUC of 0.90 or higher on these windows. That is not detection: the raw <code>log_line_count</code> alone separates the two classes with ROC-AUC {fmt(lv.raw_log_line_count_auc)}, because the Normal scenario runs 60 s and the failure scenarios 10&ndash;18 s, so it simply produces more log lines.
        Only {lv.n_high_auc_tracking_log_volume} of the {lv.high_auc_models.length} models with ROC-AUC &ge; 0.9 have a rank correlation with log volume above 0.8; for the others the driver is unidentified. All five Normal windows come from one scenario, so any separation cannot be told apart from a scenario-specific artefact. The mean predicted anomaly probability differs between Normal and Anomaly windows by at most {fmt(Math.max(...l.models.map((m) => Math.abs(m.mean_p_gap_anomaly_minus_normal))))}.
      </p>
      <ResponsiveTable<Lo2Model>
        caption={`LO2 results per model (${l.primary_setting}). Intervals resample windows, so they understate the real uncertainty.`}
        rows={l.models} rowKey={(r) => r.model} initialSort={{ key: 'auc', dir: 'desc' }}
        columns={[
          { key: 'm', label: 'Model', nameCell: true, render: (r) => r.model, sortValue: (r) => r.model },
          { key: 'ba', label: 'Balanced acc.', numeric: true, render: (r) => fmt(r.balanced_accuracy), sortValue: (r) => r.balanced_accuracy },
          { key: 'ad', label: 'AnoMod test', numeric: true, render: (r) => fmt(r.anomod_test), sortValue: (r) => r.anomod_test },
          { key: 'nr', label: 'Normal recall', numeric: true, render: (r) => fmt(r.normal_recall), sortValue: (r) => r.normal_recall },
          { key: 'sp', label: 'Predicted Anomaly', numeric: true, render: (r) => pct(r.share_predicted_anomaly, 0), sortValue: (r) => r.share_predicted_anomaly },
          { key: 'auc', label: 'ROC-AUC (95% CI)', numeric: true, render: (r) => <>{fmt(r.roc_auc)} <span className="muted small">({range(r.roc_auc_ci95_low, r.roc_auc_ci95_high)})</span></>, sortValue: (r) => r.roc_auc },
          { key: 'gap', label: 'Mean P(Anomaly): Anomaly - Normal windows', numeric: true, render: (r) => fmt(r.mean_p_gap_anomaly_minus_normal), sortValue: (r) => r.mean_p_gap_anomaly_minus_normal },
        ]}
      />
      <p className="small muted">Other settings (variant B with node metrics; system flag SN): {otherSettings.length} model runs, balanced accuracy {fmt(Math.min(...otherSettings.map((s) => s.balanced_accuracy)))} to {fmt(Math.max(...otherSettings.map((s) => s.balanced_accuracy)))}.</p>
    </section>
  )
}

function Other({ d }: { d: DataResponse }) {
  const ovSN = d.leakage.find((r) => r.check === 'time_overlap' && r.scope === 'SN' && r.item.includes('SAME run'))
  const ovTT = d.leakage.find((r) => r.check === 'time_overlap' && r.scope === 'TT' && r.item.includes('SAME run'))
  const runs = d.leakage.find((r) => r.check === 'run_overlap' && r.scope === 'ALL' && r.item.includes('runs present in both'))
  const perClass = d.leakage.find((r) => r.check === 'run_overlap' && r.item.includes('runs per class'))
  return (
    <section aria-labelledby="oth-h">
      <h2 id="oth-h">4. Other things to keep in mind</h2>
      <ul>
        <li><strong>Known scenarios only.</strong> All {String(runs?.value)} runs appear in both train and test (different blocks), so the test set does not measure generalisation to a new failure type or a new deployment. LO2 is the only check of that, and it failed.</li>
        <li><strong>Overlapping windows.</strong> {String(ovSN?.value)} SN and {String(ovTT?.value)} TT training windows overlap in time with a test window of the same run. Cross-validation scores are optimistic; the test set is the headline.</li>
        <li><strong>Few runs per class.</strong> Each class comes from a small number of runs ({perClass ? String(perClass.value) : 'see the leakage table'}), so the intervals on the Results page cannot cover run-to-run variation.</li>
        <li><strong>The best model was chosen on the test set.</strong> Its lead over the others is optimistic; many models are statistically tied with it.</li>
        <li><strong>Explanations describe the model, not the system.</strong> The most important features are host-level gauges that may identify the run; see <Link to="/how-it-works">How it works</Link>.</li>
        <li><strong>Reliability bands are heuristics.</strong> The data-quality bands (10% and 50% of readings missing) were not validated; the only measured evidence about missing features is the LO2 result above.</li>
        <li><strong>The demo feed is a replay.</strong> The Console replays recorded AnoMod windows through the real ingestion pipeline; it is not a live production system.</li>
        <li><strong>It detects, it does not forecast.</strong> A window is judged on what it contains now; the model gives no time-to-failure.</li>
      </ul>
    </section>
  )
}

const pc = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`

function PlainLimits({ models, lo2, proofs }: { models: ReturnType<typeof useApi<ModelsResponse>>; lo2: ReturnType<typeof useApi<Lo2Response>>; proofs: ReturnType<typeof useApi<ProofReport>> }) {
  return (
    <div className="stack" style={{ maxWidth: 'var(--measure)' }}>
      <Async level={2} state={models} rows={2} label="Loading results">
        {(m) => {
          const bin = m.models.filter((x) => x.label_version === 'label_binary' && x.metrics['official/ALL'])
          const best = [...bin].sort((a, b) => (b.metrics['official/ALL'].balanced_accuracy ?? 0) - (a.metrics['official/ALL'].balanced_accuracy ?? 0))[0]
          const nr = best.metrics['official/ALL'].normal_recall ?? 0
          return (
            <section>
              <h2 style={{ fontSize: '1.2rem', marginTop: 0 }}>It often cries wolf</h2>
              <p>On healthy recordings the best model ({best.name}) was right only {pc(nr, 1)} of the time. That means about {pc(1 - nr, 1)} of healthy moments were flagged as problems. Its best score at catching failures is much higher, but the false alarms are real. There were only {best.metrics['official/ALL'].support_Normal} healthy test windows, so this number is itself uncertain.</p>
            </section>
          )
        }}
      </Async>
      <Async level={2} state={proofs} rows={2} label="Loading the masking check">
        {(p) => {
          const m = maskingOf(p)
          if (!m) return null
          return (
            <section>
              <h2 style={{ fontSize: '1.2rem' }}>It relies on the system readings, not on the request data</h2>
              <p>We hid random parts of the data to see what the best model leans on. Hiding 70% of the request (API) readings, or of the log readings, changed nothing ({m.unmasked.toFixed(3)} before, {m.api70.toFixed(3)} and {m.log70.toFixed(3)} after). Hiding 70% of the machine and container readings took it to {m.metric70.toFixed(3)}, which is guessing. So for this model the API data adds nothing, and its good results come from the machine readings. When those readings are missing it tends to call the window a failure: it recognised only {(m.normal_recall_metric70 * 100).toFixed(1)}% of healthy windows (before: {(m.normal_recall_unmasked * 100).toFixed(1)}%). That is why the Console shows &ldquo;Not enough data to judge&rdquo; instead of an alert when more than half of the readings are missing. The checks that the missing-data markers themselves are not what carries the result are on the Proofs page (P13).</p>
            </section>
          )
        }}
      </Async>
      <Async level={2} state={lo2} rows={2} label="Loading outside test">
        {(l) => {
          const a = l.feature_mapping.A_logs_only
          const maxBA = Math.max(...l.models.map((x) => x.balanced_accuracy))
          return (
            <section>
              <h2 style={{ fontSize: '1.2rem' }}>It did not work on a different system</h2>
              <p>We tried the saved models on <Term k="lo2">LO2</Term>, a different system. They scored {maxBA.toFixed(3)} on balanced accuracy, which is the same as guessing. Only {a.used_features_built} of the {a.used_features_total} readings the models need could be built from that data, and we had just {l.sample.windows} windows to test on. Some models still showed a high ranking score (ROC-AUC), but that is explained by log volume: the normal run was longer and so wrote more log lines. It is not evidence of detection.</p>
            </section>
          )
        }}
      </Async>
      <section>
        <h2 style={{ fontSize: '1.2rem' }}>The live feed is a replay</h2>
        <p>The Console replays recorded AnoMod data through a real ingestion pipeline. It is a demonstration on recorded data, not a live production system. The &ldquo;correct alert&rdquo; and &ldquo;false alarm&rdquo; counts are known only because the recording says what really happened.</p>
      </section>
      <section>
        <h2 style={{ fontSize: '1.2rem' }}>It sees what is happening now, not what will happen</h2>
        <p>The model looks at the last 30 seconds and says whether they look like a failure. It does not predict a future failure or say when one will happen.</p>
      </section>
      <section>
        <h2 style={{ fontSize: '1.2rem' }}>Only failures it has seen before</h2>
        <p>The test recordings come from the same kinds of failure the models were trained on, so the scores do not show how it would handle a brand-new kind of failure. The best model was also picked by looking at the test scores, which makes it look a little better than it really is. The reasons shown with an alert are the signals the model leaned on, not a proven cause.</p>
      </section>
    </div>
  )
}

function Reliance({ m }: { m: Masking | null }) {
  if (!m) return null
  return (
    <section aria-labelledby="rel-h">
      <h2 id="rel-h">3. What the saved XGBoost relies on (missingness check, Proofs P13)</h2>
      <ul>
        <li>Masking 70% of the real values: API {m.api70.toFixed(3)}, log {m.log70.toFixed(3)}, metric {m.metric70.toFixed(3)}, all groups {m.all70.toFixed(3)} balanced accuracy (unmasked {m.unmasked.toFixed(3)}). The result rests on the metric readings.</li>
        <li>With 70% of the metric values missing, Normal recall falls from {m.normal_recall_unmasked.toFixed(3)} to {m.normal_recall_metric70.toFixed(3)}: missing readings push the call toward Anomaly.</li>
        <li>The was_missing_* indicators alone, and a retrained XGBoost without them and without the system flag, do not explain the result (details and intervals in P13).</li>
      </ul>
    </section>
  )
}
