import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, postForm, postJson, useApi } from '../api'
import { parseCsv } from '../csv'
import { FeatureBars } from '../components/Charts'
import MissingPanel from '../components/MissingPanel'
import { LabelToggle, ModelSelect, SampleSelect } from '../components/pickers'
import ResponsiveTable from '../components/ResponsiveTable'
import { Async, ErrorState, SkeletonBlock } from '../components/States'
import { ProbBars } from '../components/Charts'
import { fmt, pct } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { ExplainResponse, Label, Lo2Response, ModelsResponse, PredictResponse, PredictResult, SamplesResponse } from '../types'

type Source = 'sample' | 'csv'
const MAX_BYTES = 5 * 1024 * 1024

export default function Predict() {
  usePageTitle('Predict')
  const models = useApi<ModelsResponse>('/api/models')
  const samples = useApi<SamplesResponse>('/api/samples')
  const lo2 = useApi<Lo2Response>('/api/lo2')
  const [label, setLabel] = useState<Label>('label_binary')
  const [model, setModel] = useState('XGB')
  const [source, setSource] = useState<Source>('sample')
  const [sampleId, setSampleId] = useState('sample-01')
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [csvRows, setCsvRows] = useState<Record<string, string | number | null>[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [rowIdx, setRowIdx] = useState(0)
  const [explain, setExplain] = useState<ExplainResponse | null>(null)
  const [explainErr, setExplainErr] = useState<Error | null>(null)
  const [explainBusy, setExplainBusy] = useState(false)
  const resultRef = useRef<HTMLHeadingElement>(null)

  const forLabel = useMemo(() => (models.data?.models ?? []).filter((m) => m.label_version === label), [models.data, label])
  const current = forLabel.find((m) => m.name === model)
  useEffect(() => {
    if (forLabel.length && !forLabel.some((m) => m.name === model && m.available)) setModel((forLabel.find((m) => m.available) ?? forLabel[0]).name)
  }, [forLabel, model])

  const sample = samples.data?.windows.find((s) => s.id === sampleId)

  async function onFile(f: File | null) {
    setFile(f)
    setFileError(null)
    setCsvRows([])
    if (!f) return
    if (f.size > MAX_BYTES) { setFileError(`This file is ${(f.size / 1048576).toFixed(1)} MB; the limit is 5 MB.`); return }
    if (!/\.(csv|txt)$/i.test(f.name) && !f.type.includes('csv') && !f.type.includes('text')) { setFileError('Please choose a .csv file.'); return }
    const text = await f.text()
    const p = parseCsv(text)
    if (!p.header.length || !p.rows.length) { setFileError('The file has no data rows. It needs a header row and at least one window.'); return }
    setCsvRows(p.rows)
  }

  async function run() {
    setBusy(true); setError(null); setResult(null); setExplain(null); setExplainErr(null); setRowIdx(0)
    try {
      let r: PredictResponse
      if (source === 'sample') {
        r = await postJson<PredictResponse>('/api/predict', { model, label_version: label, sample_id: sampleId })
      } else {
        if (!file) throw new Error('Choose a CSV file first.')
        const fd = new FormData()
        fd.append('file', file)
        fd.append('model', model)
        fd.append('label_version', label)
        r = await postForm<PredictResponse>('/api/predict', fd)
      }
      setResult(r)
      setTimeout(() => resultRef.current?.focus(), 50)
      if (current?.supports_explain && (source === 'sample' || r.n_windows === 1)) void loadExplain(0, r)
    } catch (e) {
      setError(e as Error)
    } finally {
      setBusy(false)
    }
  }

  async function loadExplain(idx: number, r: PredictResponse | null = result) {
    if (!r) return
    setExplainBusy(true); setExplainErr(null)
    try {
      const body: Record<string, unknown> = { model: r.model, label_version: r.label_version, lime: false }
      if (source === 'sample') body.sample_id = sampleId
      else body.window = csvRows[idx]
      setExplain(await postJson<ExplainResponse>('/api/explain', body))
    } catch (e) {
      setExplainErr(e as Error)
    } finally {
      setExplainBusy(false)
    }
  }

  const res: PredictResult | undefined = result?.results[rowIdx]
  const truth = source === 'sample' && sample ? (label === 'label_binary' ? sample.label_binary : sample.label_level) : null

  return (
    <>
      <h1>Predict</h1>
      <p className="lede">Score one real test window, or upload your own windows. The verdict uses the model&apos;s saved decision rule, not just the highest probability.</p>

      <form className="card stack" onSubmit={(e) => { e.preventDefault(); void run() }} aria-label="Prediction inputs">
        <div className="row">
          <LabelToggle value={label} onChange={setLabel} />
          <Async state={models} rows={1}>
            {() => <ModelSelect id="model" models={forLabel} value={model} onChange={setModel} />}
          </Async>
          <div className="field">
            <span className="label" id="src-label">Input</span>
            <div className="segmented" role="group" aria-labelledby="src-label">
              <button type="button" aria-pressed={source === 'sample'} onClick={() => setSource('sample')}>Sample window</button>
              <button type="button" aria-pressed={source === 'csv'} onClick={() => setSource('csv')}>Upload CSV</button>
            </div>
          </div>
        </div>
        {current && !current.available && <p role="alert" className="small"><strong>{current.name} cannot be served:</strong> {current.unavailable_reason}</p>}
        {source === 'sample' ? (
          <Async state={samples} rows={1} label="Loading sample windows">
            {(s) => (
              <div className="stack">
                <SampleSelect id="sample" samples={s.windows} value={sampleId} onChange={setSampleId} />
                <p className="small muted">{s.source}. The true label is shown only to let you check the model.</p>
              </div>
            )}
          </Async>
        ) : (
          <div className="stack">
            <div className="field">
              <label htmlFor="csv">CSV of windows (max 5 MB, up to 2000 rows)</label>
              <input id="csv" type="file" accept=".csv,text/csv,text/plain" onChange={(e) => void onFile(e.target.files?.[0] ?? null)} aria-describedby="csv-help" />
            </div>
            <p id="csv-help" className="small muted">
              One row per 30 s window. Columns: the raw feature names (e.g. <code>log_line_count</code>, <code>api_error_rate</code>, <code>m_node_load5__mean</code>), plus <code>system</code> = <code>SN</code> or <code>TT</code>.
              Optional <code>run_id</code> lets sequence models use the previous windows of the same run. Missing columns are treated as missing features.
            </p>
            {fileError && <p role="alert"><strong>{fileError}</strong></p>}
            {csvRows.length > 0 && <p className="small">{csvRows.length} window(s) read from <code>{file?.name}</code>.</p>}
          </div>
        )}
        <div>
          <button className="btn" type="submit" disabled={busy || !current?.available || (source === 'csv' && (!file || !!fileError))}>{busy ? 'Scoring...' : 'Score'}</button>
        </div>
      </form>

      <div aria-live="polite" style={{ marginTop: 'var(--space-6)' }}>
        {busy && <SkeletonBlock rows={3} label="Scoring" />}
        {error && <ErrorState level={2} error={error} title={error instanceof ApiError && error.status >= 400 && error.status < 500 ? 'The input was not accepted' : 'Prediction failed'} />}
        {result && res && (
          <div className="stack">
            <h2 tabIndex={-1} ref={resultRef} style={{ marginTop: 0 }}>Result</h2>
            {result.n_windows > 1 && (
              <div className="card">
                <h3 style={{ marginTop: 0 }}>{result.n_windows} windows scored</h3>
                <p>{Object.entries(result.class_counts).map(([k, v]) => `${k}: ${v}`).join(' · ')}</p>
                <ResponsiveTable<PredictResult>
                  caption="Predictions per window (select a row to inspect it)"
                  rows={result.results.slice(0, 200)}
                  rowKey={(r) => String(r.index)}
                  activeKey={String(rowIdx)}
                  columns={[
                    { key: 'i', label: '#', numeric: true, render: (r) => <button type="button" className="btn secondary" style={{ minHeight: '2.25rem', padding: '0.2rem 0.7rem' }} onClick={() => { setRowIdx(r.index); setExplain(null) }}>{r.index + 1}</button>, sortValue: (r) => r.index },
                    { key: 'id', label: 'Window', render: (r) => r.window_id ?? '–', nameCell: true },
                    { key: 'sys', label: 'System', render: (r) => r.system },
                    { key: 'p', label: 'Verdict', render: (r) => r.predicted_class, sortValue: (r) => r.predicted_class },
                    { key: 'rel', label: 'Completeness', render: (r) => r.missing.reliability, sortValue: (r) => r.missing.missing_fraction },
                  ]}
                />
                {result.n_windows > 200 && <p className="small muted">Showing the first 200 of {result.n_windows} windows.</p>}
              </div>
            )}
            <div className="grid cols-2">
              <div className="card">
                <p className="eyebrow">Verdict, {result.model}{result.corrected_variant ? ' (corrected variant)' : ''}</p>
                <p style={{ fontFamily: 'var(--font-serif)', fontSize: '2.4rem', lineHeight: 1.1, margin: '0 0 var(--space-3)', fontWeight: 600 }}>{res.predicted_class}</p>
                {truth && <p>True label: <strong>{truth}</strong> &mdash; {truth === res.predicted_class ? 'the model is correct here.' : 'the model is wrong here.'}</p>}
                <ProbBars probs={res.probabilities} predicted={res.predicted_class} />
                <p className="small muted" style={{ marginTop: 'var(--space-4)' }}>
                  Decision rule: {result.decision_rule.type === 'threshold'
                    ? <>call <strong>Anomaly</strong> when P(Anomaly) &ge; <strong>{fmt(result.decision_rule.value, 2)}</strong> (chosen on out-of-fold predictions; plain argmax would say <strong>{res.default_argmax_class}</strong>).</>
                    : <>per-class weights {JSON.stringify((result.decision_rule.weights ?? []).map((w) => Number(w.toFixed(2))))} applied before taking the maximum (plain argmax would say <strong>{res.default_argmax_class}</strong>).</>}
                  {' '}On out-of-fold predictions this rule gave balanced accuracy {fmt(result.decision_rule.oof_balanced_accuracy)} and Normal recall {pct(result.decision_rule.oof_normal_recall, 0)}; see <Link to="/limits">Limits</Link> for test-set Normal recall.
                </p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>What drove it (SHAP, top features)</h3>
                {!current?.supports_explain && <p className="muted">SHAP is available for tree models (RF, ET, XGB, LGBM and their tuned versions). Pick one of those to see feature drivers.</p>}
                {current?.supports_explain && !explain && !explainBusy && !explainErr && (
                  <button className="btn secondary" type="button" onClick={() => void loadExplain(rowIdx)}>Show drivers for {source === 'csv' ? `window ${rowIdx + 1}` : 'this window'}</button>
                )}
                {explainBusy && <SkeletonBlock rows={5} label="Computing SHAP" />}
                {explainErr && <ErrorState error={explainErr} title="Could not compute SHAP" />}
                {explain && (
                  <>
                    <p className="small muted">Explained class: <strong>{explain.explained_class}</strong>. Dark bars push towards it, dashed bars against it. Hover or focus a row for value and group.</p>
                    <FeatureBars valueLabel="SHAP" items={explain.shap.top.slice(0, 8).map((t) => ({ feature: t.feature, group: t.group, value: t.shap, raw: t.raw_value, pre: t.preprocessed_value }))} />
                    <p className="small" style={{ marginTop: 'var(--space-3)' }}><Link to="/explain">Compare with LIME on the Explain page</Link></p>
                  </>
                )}
              </div>
            </div>
            <MissingPanel m={res.missing} lo2={lo2.data} />
            {result.warnings.length > 0 && (
              <div className="callout" role="note">
                <span className="label">Notes on this input</span>
                <ul>{result.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}
