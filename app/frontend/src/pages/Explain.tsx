import { useEffect, useMemo, useState } from 'react'
import { postJson, useApi } from '../api'
import { FeatureBars } from '../components/Charts'
import { LabelToggle, ModelSelect, SampleSelect } from '../components/pickers'
import { Async, ErrorState, SkeletonBlock } from '../components/States'
import { GROUP_LABEL, pct } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { DataResponse, ExplainResponse, Label, ModelsResponse, SamplesResponse } from '../types'

const CLASSES: Record<Label, string[]> = { label_binary: ['Normal', 'Anomaly'], label_level: ['Normal', 'Performance', 'Service', 'Database', 'Code'] }

export default function Explain() {
  usePageTitle('Explain')
  const models = useApi<ModelsResponse>('/api/models')
  const samples = useApi<SamplesResponse>('/api/samples')
  const dataApi = useApi<DataResponse>('/api/data')
  const [label, setLabel] = useState<Label>('label_binary')
  const [model, setModel] = useState('XGB')
  const [sampleId, setSampleId] = useState('sample-12')
  const [cls, setCls] = useState<string>('')
  const [data, setData] = useState<ExplainResponse | null>(null)
  const [err, setErr] = useState<Error | null>(null)
  const [busy, setBusy] = useState(false)

  const explainable = useMemo(() => (models.data?.models ?? []).filter((m) => m.label_version === label && m.supports_explain), [models.data, label])
  useEffect(() => {
    if (explainable.length && !explainable.some((m) => m.name === model)) setModel(explainable[0].name)
  }, [explainable, model])
  useEffect(() => setCls(''), [label, sampleId, model])

  async function run() {
    setBusy(true); setErr(null)
    try {
      setData(await postJson<ExplainResponse>('/api/explain', { model, label_version: label, sample_id: sampleId, explain_class: cls || undefined, lime: true }))
    } catch (e) {
      setErr(e as Error); setData(null)
    } finally {
      setBusy(false)
    }
  }
  const sample = samples.data?.windows.find((s) => s.id === sampleId)
  const truth = sample ? (label === 'label_binary' ? sample.label_binary : sample.label_level) : null

  const offline = useMemo(() => dataApi.data?.explainability.agreement.find((a) => a.label_version === label && a.window_set === '60 random' && a.outcome === 'all') ?? null, [dataApi.data, label])
  const agreement = useMemo(() => {
    if (!data?.lime.available || !data.lime.top) return null
    const s = new Set(data.shap.top.slice(0, 5).map((t) => t.feature))
    const shared = data.lime.top.slice(0, 5).filter((t) => s.has(t.feature)).map((t) => t.feature)
    return shared
  }, [data])

  return (
    <>
      <h1>Explain</h1>
      <p className="lede">SHAP and LIME answer the same question, which features pushed this window towards a class, in different ways. Where they disagree, trust neither blindly.</p>

      <form className="card stack" onSubmit={(e) => { e.preventDefault(); void run() }} aria-label="Explanation inputs">
        <div className="row">
          <LabelToggle value={label} onChange={setLabel} />
          <Async state={models} rows={1}>{() => <ModelSelect id="x-model" models={explainable} value={model} onChange={setModel} onlyExplainable />}</Async>
          <Async state={samples} rows={1}>{(s) => <SampleSelect id="x-sample" samples={s.windows} value={sampleId} onChange={setSampleId} />}</Async>
          <div className="field">
            <label htmlFor="x-class">Explain class</label>
            <select id="x-class" value={cls} onChange={(e) => setCls(e.target.value)}>
              <option value="">{label === 'label_binary' ? 'Anomaly (default)' : 'Predicted class (default)'}</option>
              {CLASSES[label].map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
        <p className="small muted">Only tree models are explained (SHAP TreeExplainer; LIME with a background of 400 preprocessed training windows). Tree boosters report SHAP in log-odds.</p>
        <div><button className="btn" type="submit" disabled={busy}>{busy ? 'Explaining...' : 'Explain this window'}</button></div>
      </form>

      <div aria-live="polite" style={{ marginTop: 'var(--space-6)' }}>
        {busy && <SkeletonBlock rows={6} label="Computing explanations" />}
        {err && <ErrorState level={2} error={err} title="Could not explain this window" />}
        {data && (
          <div className="stack">
            <div className="card">
              <p className="eyebrow">Window {sampleId.replace('sample-', '#')} &middot; {data.prediction.system} &middot; {model}</p>
              <p style={{ margin: 0 }}>
                Verdict <strong>{data.prediction.predicted_class}</strong>
                {truth && <> &middot; true label <strong>{truth}</strong> ({truth === data.prediction.predicted_class ? 'correct' : 'wrong'})</>}
                {' '}&middot; explained class <strong>{data.explained_class}</strong>.
              </p>
            </div>
            <div className="grid cols-2">
              <section className="card" aria-labelledby="shap-h">
                <h2 id="shap-h" style={{ marginTop: 0, fontSize: '1.3rem' }}>SHAP</h2>
                <p className="small muted">Share of total |SHAP| by group: {data.shap.groups.map((g) => `${GROUP_LABEL[g.group] ?? g.group} ${pct(g.share, 0)}`).join(', ')}.</p>
                <FeatureBars valueLabel="SHAP" items={data.shap.top.map((t) => ({ feature: t.feature, group: t.group, value: t.shap, raw: t.raw_value, pre: t.preprocessed_value }))} />
              </section>
              <section className="card" aria-labelledby="lime-h">
                <h2 id="lime-h" style={{ marginTop: 0, fontSize: '1.3rem' }}>LIME</h2>
                {data.lime.available && data.lime.top ? (
                  <>
                    <p className="small muted">Local linear surrogate weights (dark = towards the class, dashed = against).</p>
                    <FeatureBars valueLabel="LIME weight" items={data.lime.top.map((t) => ({ feature: t.feature, group: t.group, value: t.weight, raw: t.raw_value, pre: t.preprocessed_value }))} />
                  </>
                ) : (
                  <div className="state"><h3>LIME is not available</h3><p>{data.lime.reason}</p></div>
                )}
              </section>
            </div>
            {agreement && (
              <div className="callout accent">
                <span className="label">Agreement</span>
                <p>
                  SHAP and LIME share <strong>{agreement.length} of their top 5 features</strong> for this window{agreement.length ? `: ${agreement.join(', ')}` : ''}.{' '}
                  {offline && <>In the offline analysis of {offline.model} on {offline.n_windows} random test windows the two methods shared on average {offline.mean_shared_top5.toFixed(2)} of the top 5 features, and at least 3 in {pct(offline.share_with_ge3_shared, 0)} of windows.</>}
                </p>
              </div>
            )}
            <div className="callout">
              <span className="label">How to read this</span>
              <p>{data.note} Most top features are host-level gauges (disk usage, memory, filesystem space). They may identify which run a window came from rather than what failed, so a large attribution is not evidence of a causal failure signal.</p>
              {data.warnings.length > 0 && <ul>{data.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
            </div>
          </div>
        )}
        {!data && !busy && !err && <div className="state"><h2>Nothing explained yet</h2><p>Choose a window and press <strong>Explain this window</strong>.</p></div>}
      </div>
    </>
  )
}
