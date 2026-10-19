import { useMemo, useState } from 'react'
import { useApi } from '../api'
import { ConfusionMatrix, LineChart, RangePlot } from '../components/Charts'
import { LabelToggle } from '../components/pickers'
import ResponsiveTable from '../components/ResponsiveTable'
import type { Column } from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { FAMILY_LABEL, fmt, range } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { CorrectionRow, Label, ModelDetail, ModelInfo, ModelsResponse, OverfitRow, PairedRow, ResultsResponse, V1Row } from '../types'

type MetricKey = 'balanced_accuracy' | 'macro_f1' | 'normal_recall' | 'roc_auc'
const METRICS: { key: MetricKey; label: string }[] = [
  { key: 'balanced_accuracy', label: 'Balanced accuracy' },
  { key: 'macro_f1', label: 'Macro-F1' },
  { key: 'normal_recall', label: 'Normal recall' },
  { key: 'roc_auc', label: 'ROC-AUC' },
]

function Cell({ m, k }: { m: ModelInfo; k: MetricKey }) {
  const ci = m.ci[`ALL/${k}`]
  return (
    <>
      <strong>{fmt(m.metrics['official/ALL']?.[k])}</strong>
      <span className="muted small"> ({ci ? range(ci.low, ci.high) : '–'})</span>
    </>
  )
}

export default function Models() {
  usePageTitle('Models')
  const models = useApi<ModelsResponse>('/api/models')
  const results = useApi<ResultsResponse>('/api/results')
  const [label, setLabel] = useState<Label>('label_binary')
  const [metric, setMetric] = useState<MetricKey>('balanced_accuracy')
  const [selected, setSelected] = useState<string | null>(null)
  const list = useMemo(() => (models.data?.models ?? []).filter((m) => m.label_version === label), [models.data, label])
  const best = useMemo(() => [...list].sort((a, b) => (b.metrics['official/ALL']?.balanced_accuracy ?? 0) - (a.metrics['official/ALL']?.balanced_accuracy ?? 0))[0], [list])
  const sel = list.find((m) => m.name === (selected ?? best?.name))

  const columns: Column<ModelInfo>[] = [
    {
      key: 'name', label: 'Model', nameCell: true, sortValue: (m) => m.name,
      render: (m) => (
        <>
          <button type="button" className="btn secondary" style={{ minHeight: '2.25rem', padding: '0.15rem 0.6rem', fontWeight: 650 }} aria-pressed={sel?.name === m.name} onClick={() => setSelected(m.name)}>{m.name}</button>{' '}
          <span className="badge">{FAMILY_LABEL[m.family]}</span> {m.corrected_variant && <span className="badge accent">corrected</span>} {!m.available && <span className="badge">report only</span>}
        </>
      ),
    },
    { key: 'ba', label: 'Balanced acc. (95% CI)', numeric: true, sortValue: (m) => m.metrics['official/ALL']?.balanced_accuracy ?? null, render: (m) => <Cell m={m} k="balanced_accuracy" /> },
    { key: 'f1', label: 'Macro-F1 (95% CI)', numeric: true, sortValue: (m) => m.metrics['official/ALL']?.macro_f1 ?? null, render: (m) => <Cell m={m} k="macro_f1" /> },
    { key: 'nr', label: 'Normal recall (95% CI)', numeric: true, sortValue: (m) => m.metrics['official/ALL']?.normal_recall ?? null, render: (m) => <Cell m={m} k="normal_recall" /> },
    { key: 'auc', label: 'ROC-AUC (95% CI)', numeric: true, sortValue: (m) => m.metrics['official/ALL']?.roc_auc ?? null, render: (m) => <Cell m={m} k="roc_auc" /> },
    { key: 'pr', label: 'PR-AUC', numeric: true, sortValue: (m) => m.metrics['official/ALL']?.pr_auc ?? null, render: (m) => fmt(m.metrics['official/ALL']?.pr_auc) },
  ]

  return (
    <>
      <h1>Models</h1>
      <p className="lede">Every model was scored once on the 647 untouched test windows. Intervals are 95% block-bootstrap ranges, and they are wide: with few Normal windows, small differences between models are not reliable.</p>
      <div className="row" style={{ marginBottom: 'var(--space-5)' }}>
        <LabelToggle value={label} onChange={(l) => { setLabel(l); setSelected(null) }} />
      </div>

      <section aria-labelledby="lb">
        <h2 id="lb" style={{ marginTop: 0 }}>Leaderboard</h2>
        <Async state={models} rows={6} label="Loading leaderboard">
          {(m) => (
            <div className="stack">
              <ResponsiveTable<ModelInfo>
                caption={`Test-set results, ${label === 'label_binary' ? 'binary' : 'five-class'} labels (decision = argmax, all test windows). Click a column to sort, a model name to inspect it.`}
                columns={columns} rows={list} rowKey={(r) => r.name} initialSort={{ key: 'ba', dir: 'desc' }} activeKey={sel?.name ?? null}
              />
              <p className="small muted">{m.ci_note} The best model per label was identified on the test set itself, so its lead is optimistic. &ldquo;Report only&rdquo; = soft voting needs 40 fold models and cannot be served live.</p>
              <div className="card">
                <div className="row" style={{ marginBottom: 'var(--space-3)' }}>
                  <div className="field">
                    <label htmlFor="metric">Interval plot metric</label>
                    <select id="metric" value={metric} onChange={(e) => setMetric(e.target.value as MetricKey)}>{METRICS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}</select>
                  </div>
                </div>
                <RangePlot
                  ariaLabel={`Point estimate and 95% confidence interval of ${METRICS.find((x) => x.key === metric)?.label} for each model`}
                  items={[...list].sort((a, b) => (b.ci[`ALL/${metric}`]?.point ?? 0) - (a.ci[`ALL/${metric}`]?.point ?? 0)).map((x) => ({ label: x.name, point: x.ci[`ALL/${metric}`]?.point ?? null, low: x.ci[`ALL/${metric}`]?.low ?? null, high: x.ci[`ALL/${metric}`]?.high ?? null, emphasis: x.name === sel?.name }))}
                />
              </div>
            </div>
          )}
        </Async>
      </section>

      {sel && <Detail model={sel} label={label} best={best?.name ?? ''} paired={results.data?.paired_bootstrap ?? []} />}

      <Async state={results} rows={5} label="Loading comparison tables">
        {(r) => (
          <>
            <FirstRun rows={r.first_run_vs_retrain.filter((x) => x.label_version === label)} notes={r.notes} />
            <Correction label={label} correction={r.correction.filter((x) => x.label_version === label)} overfit={r.overfit.filter((x) => x.label_version === label)} />
          </>
        )}
      </Async>
    </>
  )
}

function Detail({ model, label, best, paired }: { model: ModelInfo; label: Label; best: string; paired: PairedRow[] }) {
  const detail = useApi<ModelDetail>(`/api/results/model/${label}/${model.name}`)
  const bestDetail = useApi<ModelDetail>(model.name === best ? null : `/api/results/model/${label}/${best}`)
  const [scope, setScope] = useState('ALL')
  const [dec, setDec] = useState<'argmax' | 'tuned'>('argmax')
  const pr = paired.find((p) => p.label_version === label && p.other_model === model.name)
  const rule = model.decision_rule
  return (
    <section aria-labelledby="detail-h">
      <h2 id="detail-h">{model.name}: confusion matrix and curves</h2>
      <p className="muted">
        {FAMILY_LABEL[model.family]} model, imbalance handling <code>{model.imbalance_method}</code>{model.corrected_variant ? ', regularised ("corrected") variant adopted after the overfitting diagnosis' : ''}.
        {rule && <> Saved decision rule: {rule.type === 'threshold' ? <>Anomaly when P(Anomaly) &ge; {fmt(rule.threshold, 2)}</> : 'per-class weights'}; out-of-fold balanced accuracy {fmt(rule.oof_balanced_accuracy)} (argmax: {fmt(rule.oof_balanced_accuracy_default_argmax)}), out-of-fold Normal recall {fmt(rule.oof_normal_recall)}.</>}
      </p>
      {pr && (
        <p>Against the best model ({pr.best_model}), balanced accuracy differs by {fmt(pr.mean_diff, 3)} (95% CI {range(pr.ci95_low, pr.ci95_high, 3)}; Holm-adjusted p = {fmt(pr.p_holm, 3)}) &mdash; {pr['significant_at_0.05_holm'] ? 'a difference the bootstrap supports' : 'not distinguishable from the best model'}.</p>
      )}
      <Async state={detail} rows={5} label="Loading curves">
        {(d) => {
          const cm = d.confusion[`${scope}/${dec}`] ?? d.confusion['ALL/argmax']
          const b = bestDetail.data
          return (
            <div className="grid cols-2">
              <div className="card">
                <div className="row" style={{ marginBottom: 'var(--space-3)' }}>
                  <div className="field"><label htmlFor="cm-scope">Windows</label>
                    <select id="cm-scope" value={scope} onChange={(e) => setScope(e.target.value)}><option value="ALL">All test windows</option><option value="SN">SN only</option><option value="TT">TT only</option></select></div>
                  <div className="field"><span className="label" id="dec-l">Decision</span>
                    <div className="segmented" role="group" aria-labelledby="dec-l">
                      <button type="button" aria-pressed={dec === 'argmax'} onClick={() => setDec('argmax')}>Argmax</button>
                      <button type="button" aria-pressed={dec === 'tuned'} onClick={() => setDec('tuned')}>Saved threshold</button>
                    </div></div>
                </div>
                {cm ? <ConfusionMatrix labels={cm.labels} matrix={cm.matrix} ariaLabel={`Confusion matrix of ${model.name}: rows are true classes, columns are predicted classes`} /> : <p>No confusion matrix file for this selection.</p>}
              </div>
              <div className="card">
                {d.curves ? (
                  <div className="grid cols-2" style={{ gap: 'var(--space-3)' }}>
                    <div>
                      <h3 style={{ marginTop: 0, fontSize: '1rem' }}>ROC</h3>
                      <LineChart ariaLabel={`ROC curve of ${model.name}`} xLabel="false positive rate" yLabel="true positive rate" chance={[[0, 0], [1, 1]]}
                        series={[{ name: model.name, points: d.curves.roc }, ...(b?.curves ? [{ name: `${best} (best)`, points: b.curves.roc }] : [])]} />
                    </div>
                    <div>
                      <h3 style={{ marginTop: 0, fontSize: '1rem' }}>Precision-recall</h3>
                      <LineChart ariaLabel={`Precision-recall curve of ${model.name}`} xLabel="recall" yLabel="precision"
                        series={[{ name: model.name, points: d.curves.pr }, ...(b?.curves ? [{ name: `${best} (best)`, points: b.curves.pr }] : [])]} />
                    </div>
                  </div>
                ) : <p>No curve data for this model.</p>}
                <p className="small muted">{d.curve_note}</p>
              </div>
            </div>
          )
        }}
      </Async>
    </section>
  )
}

function FirstRun({ rows, notes }: { rows: V1Row[]; notes: string[] }) {
  return (
    <section aria-labelledby="v1-h">
      <h2 id="v1-h">First run versus full retrain</h2>
      <p className="muted">Same 647 test windows. The first run capped training at 16 epochs and used row-level folds; the retrain uses the full epoch policy, run-aware folds, SMOTE only inside training folds, tuned thresholds and a correction stage.</p>
      <ResponsiveTable<V1Row>
        caption="Test balanced accuracy and Normal recall, first run (v1) against retrain (v2); only models present in both"
        rows={rows} rowKey={(r) => r.model}
        initialSort={{ key: 'd', dir: 'desc' }}
        columns={[
          { key: 'm', label: 'Model', nameCell: true, render: (r) => r.model, sortValue: (r) => r.model },
          { key: 'b1', label: 'v1 balanced acc.', numeric: true, render: (r) => fmt(r.v1_balanced_accuracy) },
          { key: 'b2', label: 'v2 balanced acc.', numeric: true, render: (r) => fmt(r.v2_balanced_accuracy) },
          { key: 'd', label: 'Change', numeric: true, render: (r) => `${r.delta_balanced_accuracy >= 0 ? '+' : ''}${fmt(r.delta_balanced_accuracy)}`, sortValue: (r) => r.delta_balanced_accuracy },
          { key: 'n1', label: 'v1 Normal recall', numeric: true, render: (r) => fmt(r.v1_normal_recall) },
          { key: 'n2', label: 'v2 Normal recall', numeric: true, render: (r) => fmt(r.v2_normal_recall) },
        ]}
      />
      <ul className="small muted" style={{ marginTop: 'var(--space-3)' }}>{notes.map((n) => <li key={n}>{n}</li>)}</ul>
    </section>
  )
}

function Correction({ label, correction, overfit }: { label: Label; correction: CorrectionRow[]; overfit: OverfitRow[] }) {
  const fin = overfit.filter((o) => o.stage === 'final')
  return (
    <section aria-labelledby="corr-h">
      <h2 id="corr-h">Overfitting: before and after correction</h2>
      <p className="muted">A model was diagnosed as overfit when its train-minus-validation macro-F1 gap exceeded 0.10. It was retrained with stronger regularisation and the corrected version was adopted only if validation balanced accuracy did not drop by more than 0.01 and the gap did not grow.</p>
      <ResponsiveTable<CorrectionRow>
        caption={`Validation scores before and after correction (${label === 'label_binary' ? 'binary' : 'five-class'}; 10-fold, run-aware)`}
        rows={correction} rowKey={(r) => r.model} initialSort={{ key: 'gap0', dir: 'desc' }}
        columns={[
          { key: 'm', label: 'Model', nameCell: true, render: (r) => r.model, sortValue: (r) => r.model },
          { key: 'k', label: 'Kind', render: (r) => r.kind },
          { key: 'b0', label: 'Val. bal. acc. before', numeric: true, render: (r) => fmt(r.val_bal_acc_before), sortValue: (r) => r.val_bal_acc_before },
          { key: 'b1', label: 'after', numeric: true, render: (r) => fmt(r.val_bal_acc_after), sortValue: (r) => r.val_bal_acc_after },
          { key: 'gap0', label: 'Gap before', numeric: true, render: (r) => fmt(r.gap_macro_f1_before), sortValue: (r) => r.gap_macro_f1_before },
          { key: 'gap1', label: 'Gap after', numeric: true, render: (r) => fmt(r.gap_macro_f1_after), sortValue: (r) => r.gap_macro_f1_after },
          { key: 'ad', label: 'Adopted', render: (r) => (r.adopted ? <span className="badge accent">yes</span> : <span className="badge">no</span>), sortValue: (r) => Number(r.adopted) },
        ]}
      />
      <h3>Train, validation and test after the final fit</h3>
      <ResponsiveTable<OverfitRow>
        caption="Macro-F1 on the training part, in cross-validation and on the test set (final models); the last column is the diagnosis"
        rows={fin} rowKey={(r) => r.model} initialSort={{ key: 'gap', dir: 'desc' }}
        columns={[
          { key: 'm', label: 'Model', nameCell: true, render: (r) => r.model, sortValue: (r) => r.model },
          { key: 'tr', label: 'Train (final)', numeric: true, render: (r) => fmt(r.final_train_macro_f1), sortValue: (r) => r.final_train_macro_f1 ?? null },
          { key: 'cv', label: 'CV validation', numeric: true, render: (r) => fmt(r.cv_val_macro_f1), sortValue: (r) => r.cv_val_macro_f1 },
          { key: 'te', label: 'Test', numeric: true, render: (r) => fmt(r.test_macro_f1), sortValue: (r) => r.test_macro_f1 ?? null },
          { key: 'gap', label: 'Train - test', numeric: true, render: (r) => fmt(r.train_test_gap_macro_f1), sortValue: (r) => r.train_test_gap_macro_f1 ?? null },
          { key: 'dx', label: 'Diagnosis', render: (r) => <span className="small">{r.diagnosis}</span> },
        ]}
      />
    </section>
  )
}
