import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { ConfusionMatrix, LineChart, RangePlot } from '../components/Charts'
import { LabelToggle } from '../components/pickers'
import ResponsiveTable from '../components/ResponsiveTable'
import type { Column } from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { FAMILY_LABEL, fmt, range } from '../format'
import { Detailed, Simple } from '../mode'
import Term from '../components/Term'
import { usePageTitle } from '../usePageTitle'
import { BothTable } from './Hybrids'
import type { BothResponse, CorrectionRow, HybridResponse, HybridRow, Label, Lo2LeaderboardResponse, ModelDetail, ModelInfo, ModelsResponse, OverfitRow, PairedRow, ResultsResponse, V1Row } from '../types'

type MetricKey = 'balanced_accuracy' | 'macro_f1' | 'normal_recall' | 'roc_auc'
const METRICS: { key: MetricKey; label: string }[] = [
  { key: 'balanced_accuracy', label: 'Balanced accuracy' },
  { key: 'macro_f1', label: 'Macro-F1' },
  { key: 'normal_recall', label: 'Normal recall' },
  { key: 'roc_auc', label: 'ROC-AUC' },
]

const PLAIN_FAMILY: Record<string, string> = { tree: 'Decision-tree model', linear: 'Linear model', deep: 'Neural network', ensemble: 'Combination of models', hybrid: 'Hybrid neural network (two learning parts)' }
const TOP_N = 9
const HYB = 'hybrid' as ModelInfo['family']
const PLAIN_PILLS: [string, string][] = [
  ['Decision-tree model', 'many yes/no questions in a row (Tree)'],
  ['Linear model', 'a weighted sum of the readings (Linear)'],
  ['Neural network', 'layers of small calculators (Deep)'],
  ['Hybrid neural network (two learning parts)', 'two algorithms in a row, the second using the first one&rsquo;s values, checked with 5 folds (Hybrid)'],
  ['Combination of models', 'several models vote together (Ensemble)'],
  ['Retrained to reduce overfitting', 'the first version memorised its practice data, so it was retrained more gently (corrected)'],
  ['Scored for comparison; cannot run live', 'it needs 40 fold models, so it is only in the table (report only)'],
]

/** Column header in two lines (plain name + what it means) with the technical name in small grey text underneath. */
export function Head({ plain, meaning, tech }: { plain: string; meaning: string; tech: string }) {
  return <span className="th-plain"><strong>{plain}</strong><span className="th-mean">{meaning}</span><span className="th-tech">{tech}</span></span>
}
const Help = ({ k }: { k: string }) => <Term k={k}><></></Term>

/** The number, and under it a thin bar from 0 to 1 with the 95% range as a lighter band. The number is always there: the bar is only a help. */
export function Score({ v, low, high }: { v: number | null | undefined; low?: number | null; high?: number | null }) {
  const has = typeof v === 'number'
  const clamp = (x: number) => Math.max(0, Math.min(1, x))
  return (
    <div className="score">
      <div className="score-num"><strong>{fmt(v)}</strong>{low != null && high != null && <span className="muted small"> ({range(low, high)})</span>}</div>
      {has && (
        <div className="sbar" aria-hidden="true">
          {low != null && high != null && <span className="sbar-ci" style={{ left: `${clamp(low) * 100}%`, width: `${Math.max(0, clamp(high) - clamp(low)) * 100}%` }} />}
          <span className="sbar-fill" style={{ width: `${clamp(v as number) * 100}%` }} />
        </div>
      )}
    </div>
  )
}

function Cell({ m, k }: { m: ModelInfo; k: MetricKey }) {
  const ci = m.ci[`ALL/${k}`]
  return <Score v={m.metrics['official/ALL']?.[k]} low={ci?.low} high={ci?.high} />
}

/** A hybrid result row (from /api/hybrid) shaped like a ModelInfo so the same table shows it. Numbers are the saved ones; nothing is computed here. */
function hybridAsModel(r: HybridRow): ModelInfo {
  const ci = (k: string, point?: number | null, low?: number | null, high?: number | null) => ({ [`ALL/${k}`]: { point: point ?? null, low: low ?? null, high: high ?? null, n_resamples: 1000 } })
  return {
    name: `${r.model} (hybrid)`, label_version: r.label_version, family: HYB, kind: 'hybrid', base: null, imbalance_method: r.method ?? null, corrected_variant: r.stage === 'corr', members: [], available: false,
    unavailable_reason: 'hybrid models are scored for comparison and are not served live', supports_explain: false, decision_rule: null,
    metrics: { 'official/ALL': { n: 647, support_Normal: 36, balanced_accuracy: r.test_bal_acc ?? undefined, macro_f1: r.test_macro_f1 ?? undefined, normal_recall: r.test_normal_recall ?? undefined, roc_auc: r.test_roc_auc ?? undefined } },
    ci: { ...ci('balanced_accuracy', r.test_bal_acc, r.test_bal_acc_ci_low, r.test_bal_acc_ci_high), ...ci('macro_f1', r.test_macro_f1, r.macro_f1_ci_low, r.macro_f1_ci_high), ...ci('normal_recall', r.test_normal_recall, r.normal_recall_ci_low, r.normal_recall_ci_high), ...ci('roc_auc', r.test_roc_auc, r.roc_auc_ci_low, r.roc_auc_ci_high) },
  }
}

export default function Results() {
  usePageTitle('Results')
  const models = useApi<ModelsResponse>('/api/models')
  const results = useApi<ResultsResponse>('/api/results')
  const [label, setLabel] = useState<Label>('label_binary')
  const [metric, setMetric] = useState<MetricKey>('balanced_accuracy')
  const [selected, setSelected] = useState<string | null>(null)
  const hybrid = useApi<HybridResponse>('/api/hybrid')
  const [dataset, setDataset] = useState<'both' | 'anomod' | 'lo2'>('both')
  const [showAll, setShowAll] = useState(false)
  const hybridModels = useMemo(() => (hybrid.data?.anomod.models ?? []).filter((r) => r.label_version === label && r.status !== 'not completed' && typeof r.test_bal_acc === 'number').map(hybridAsModel), [hybrid.data, label])
  const list = useMemo(() => [...(models.data?.models ?? []).filter((m) => m.label_version === label), ...hybridModels], [models.data, label, hybridModels])
  const ranked = useMemo(() => [...list].sort((a, b) => (b.metrics['official/ALL']?.balanced_accuracy ?? 0) - (a.metrics['official/ALL']?.balanced_accuracy ?? 0)), [list])
  const shown = showAll ? ranked : ranked.slice(0, TOP_N)
  const best = useMemo(() => [...list].sort((a, b) => (b.metrics['official/ALL']?.balanced_accuracy ?? 0) - (a.metrics['official/ALL']?.balanced_accuracy ?? 0))[0], [list])
  const sel = list.find((m) => m.name === (selected ?? best?.name))

  const support = list[0]?.metrics['official/ALL']?.support_Normal ?? 36
  const columns: Column<ModelInfo>[] = [
    {
      key: 'name', label: 'Model', nameCell: true, sortValue: (m) => m.name,
      render: (m) => (
        <>
          <button type="button" className="btn secondary" style={{ minHeight: '2.25rem', padding: '0.15rem 0.6rem', fontWeight: 650 }} aria-pressed={sel?.name === m.name} onClick={() => setSelected(m.name)}>{m.name}</button>{' '}
          {m.name === best?.name && <span className="badge solid">Best overall score</span>}
          {m.family === HYB && <span className="badge accent">Hybrid</span>}
          <span className="pills"><span className="badge">{PLAIN_FAMILY[m.family] ?? FAMILY_LABEL[m.family]}</span> {m.corrected_variant && <span className="badge accent">Retrained to reduce overfitting</span>} {!m.available && m.family !== HYB && <span className="badge">Scored for comparison; cannot run live</span>}{!m.available && m.family === HYB && <span className="badge">Scored for comparison; 5-fold checking; cannot run live</span>}</span>
        </>
      ),
    },
    { key: 'ba', label: 'Overall score', header: <Head plain="Overall score" meaning="0.5 = coin flip, 1.0 = perfect" tech="Balanced accuracy" />, help: <Help k="lb_ba" />, numeric: true, sortValue: (m) => m.metrics['official/ALL']?.balanced_accuracy ?? null, render: (m) => <Cell m={m} k="balanced_accuracy" /> },
    { key: 'f1', label: 'Fairness across both classes', header: <Head plain="Fairness across both classes" meaning="higher is better" tech="Macro-F1" />, help: <Help k="lb_f1" />, numeric: true, sortValue: (m) => m.metrics['official/ALL']?.macro_f1 ?? null, render: (m) => <Cell m={m} k="macro_f1" /> },
    { key: 'nr', label: 'Healthy windows left alone', header: <Head plain="Healthy windows left alone" meaning={`out of ${support} healthy windows`} tech="Normal recall" />, help: <Help k="lb_nr" />, numeric: true, sortValue: (m) => m.metrics['official/ALL']?.normal_recall ?? null, render: (m) => <Cell m={m} k="normal_recall" /> },
    { key: 'auc', label: 'Ranks risky above safe', header: <Head plain="Ranks risky above safe" meaning="1.0 = perfect ranking" tech="ROC-AUC" />, help: <Help k="lb_roc" />, numeric: true, sortValue: (m) => m.metrics['official/ALL']?.roc_auc ?? null, render: (m) => <Cell m={m} k="roc_auc" /> },
    { key: 'pr', label: 'Catches problems cleanly', header: <Head plain="Catches problems cleanly" meaning="1.0 = perfect" tech="PR-AUC" />, help: <Help k="lb_pr" />, numeric: true, sortValue: (m) => m.metrics['official/ALL']?.pr_auc ?? null, render: (m) => <Score v={m.metrics['official/ALL']?.pr_auc} /> },
  ]

  return (
    <>
      <h1>Results</h1>
      <Simple><SimpleResults models={models} /></Simple>
      <Detailed>
      <p className="lede">Every model was scored once on the 647 untouched test windows. Intervals are 95% block-bootstrap ranges, and they are wide: with few Normal windows, small differences between models are not reliable.</p>
      <div className="row" style={{ marginBottom: 'var(--space-5)' }}>
        <div className="field"><span className="label" id="ds-l">Data set</span>
          <div className="segmented" role="group" aria-labelledby="ds-l">
            <button type="button" aria-pressed={dataset === 'both'} onClick={() => setDataset('both')}>Both data sets (all models)</button>
            <button type="button" aria-pressed={dataset === 'anomod'} onClick={() => setDataset('anomod')}>AnoMod only (earlier models)</button>
            <button type="button" aria-pressed={dataset === 'lo2'} onClick={() => setDataset('lo2')}>LO2 slice (own data)</button>
          </div></div>
        <LabelToggle value={label} onChange={(l) => { setLabel(l); setSelected(null) }} />
      </div>
      {dataset === 'both' && <BothSection />}
      {dataset === 'lo2' && <Lo2Leaderboard label={label} />}
      {dataset === 'anomod' && <>

      <section aria-labelledby="lb">
        <h2 id="lb" style={{ marginTop: 0 }}>Leaderboard</h2>
        <div className="callout accent how-to-read" role="note">
          <span className="label">How to read this table</span>
          <ul>
            <li><strong>Overall score</strong> is the one to look at first: 0.5 means no better than a coin flip, 1.0 means perfect.</li>
            <li><strong>Healthy windows left alone</strong> shows how many healthy moments did not cause a false alarm. A low number means many false alarms (only {support} windows are healthy ones).</li>
            <li><strong>Wide ranges mean differences are not reliable.</strong> The lighter band behind each bar, and the brackets, are the 95% range (95% CI): if two models&rsquo; ranges overlap a lot, one is not clearly better than the other.</li>
            <li><strong>The best model was picked on the test set</strong>, so its lead is optimistic.</li>
          </ul>
        </div>
        <Async state={models} rows={6} label="Loading leaderboard">
          {(m) => (
            <div className="stack">
              <dl className="pill-legend" aria-label="What the labels next to each model mean">
                {PLAIN_PILLS.map(([a, b]) => <div key={a}><dt><span className="badge">{a}</span></dt><dd>{b}</dd></div>)}
              </dl>
              <ResponsiveTable<ModelInfo>
                caption={`Test-set results, ${label === 'label_binary' ? 'binary' : 'five-class'} labels (decision = argmax, all test windows). Click a column to sort, a model name to inspect it.`}
                columns={columns} rows={shown} rowKey={(r) => r.name} initialSort={{ key: 'ba', dir: 'desc' }} activeKey={sel?.name ?? null} highlightKey={best?.name ?? null}
              />
              <div className="show-all">
                <button type="button" className="btn secondary" aria-expanded={showAll} data-testid="show-all" onClick={() => setShowAll((v) => !v)}>{showAll ? `Show only the top ${TOP_N}` : `Show all models (${ranked.length})`}</button>
                <p className="small muted" style={{ margin: 'var(--space-2) 0 0' }} data-testid="top-note">{showAll ? `All ${ranked.length} models are shown.` : `Showing the top ${Math.min(TOP_N, ranked.length)} of ${ranked.length} models by overall score; the others are kept under \u201cShow all models\u201d.`} The ranking uses the test set, so the lead is optimistic. Hybrid models were checked with 5 folds, the original AnoMod models with 10.</p>
              </div>
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

      {sel && sel.family !== HYB && <Detail model={sel} label={label} best={best?.name ?? ''} paired={results.data?.paired_bootstrap ?? []} />}

      <Async state={results} rows={5} label="Loading comparison tables">
        {(r) => (
          <>
            <FirstRun rows={r.first_run_vs_retrain.filter((x) => x.label_version === label)} notes={r.notes} />
            <Correction label={label} correction={r.correction.filter((x) => x.label_version === label)} overfit={r.overfit.filter((x) => x.label_version === label)} />
          </>
        )}
      </Async>
      </>}
      </Detailed>
    </>
  )
}

function BothSection() {
  const api = useApi<BothResponse>('/api/both')
  return (
    <section aria-labelledby="both-h">
      <h2 id="both-h" style={{ marginTop: 0 }}>All models, trained and tested on both data sets</h2>
      <p className="muted">Healthy-or-not question. Each model is trained on the AnoMod and LO2 training windows together and scored on the test windows of both; the last two columns split that test by data set.</p>
      <Async state={api} rows={5} label="Loading results">{(d) => (d.available ? <BothTable rows={d.rows} /> : <div className="state"><h3>No combined results yet</h3><p>{d.note}</p></div>)}</Async>
    </section>
  )
}

/** The LO2 leaderboard: hybrids first, then the other families in priority order; unfinished ones are listed as not completed. Rows carry the number of runs and test windows behind them. */
function Lo2Leaderboard({ label }: { label: Label }) {
  const api = useApi<Lo2LeaderboardResponse>('/api/lo2/leaderboard')
  const hy = useApi<HybridResponse>('/api/hybrid')
  const f = hy.data?.lo2.facts
  const [showAll, setShowAll] = useState(false)
  return (
    <section aria-labelledby="lo2-lb">
      <h2 id="lo2-lb" style={{ marginTop: 0 }}>LO2 slice leaderboard</h2>
      <div className="callout" role="note"><span className="label">Read this first</span>
        <ul>
          <li><strong>Sample.</strong> {f ? `All ${f.runs} downloaded LO2 runs: ${f.train_runs} for training (balanced with SMOTE) and ${f.test_runs} whole runs left untouched for testing (${f.test} windows, ${f.healthy_test_windows} healthy).` : 'All downloaded LO2 runs, 80% for training and 20% untouched for testing.'}</li>
          <li><strong>The binary score is high for a real reason.</strong> Faults in LO2 leave clear error lines and metric changes. Checks show it is not a warm-up or run-length shortcut (see the Proof page, P21). It is still one service with injected HTTP errors.</li>
          <li>These models were trained on LO2&rsquo;s own data. The results say nothing about transfer from AnoMod.</li>
          <li>The ranking uses the test set, so the lead is optimistic.</li>
        </ul>
      </div>
      <Async state={api} rows={5} label="Loading the LO2 leaderboard">
        {(d) => {
          if (!d.available) return <div className="state"><h3>No LO2 leaderboard yet</h3><p>{d.note}</p></div>
          const rows = d.rows.filter((r) => r.label_version === label)
          const done = rows.filter((r) => r.status !== 'not completed').sort((a, b) => (b.test_bal_acc ?? 0) - (a.test_bal_acc ?? 0))
          const nc = rows.filter((r) => r.status === 'not completed')
          const shown = showAll ? done : done.slice(0, TOP_N)
          return (
            <div className="stack">
              <ResponsiveTable<HybridRow>
                caption={`LO2 test results, ${label === 'label_binary' ? 'binary' : 'four-class (Normal, HTTP 400, 401, 404)'} labels, test part = one whole run`}
                rows={shown} rowKey={(r) => r.model} initialSort={{ key: 'ba', dir: 'desc' }} highlightKey={done[0]?.model ?? null}
                columns={[
                  { key: 'name', label: 'Model', nameCell: true, sortValue: (r) => r.model, render: (r) => <><strong>{r.model}</strong> <span className="pills"><span className="badge">{(r.family ?? '').replace(/^\([a-e]\) /, '')}</span></span></> },
                  { key: 'ba', label: 'Overall score', header: <Head plain="Overall score" meaning="0.5 = coin flip, 1.0 = perfect" tech="Balanced accuracy" />, numeric: true, sortValue: (r) => r.test_bal_acc ?? null, render: (r) => <Score v={r.test_bal_acc} low={r.test_bal_acc_ci_low} high={r.test_bal_acc_ci_high} /> },
                  { key: 'f1', label: 'Fairness across both classes', header: <Head plain="Fairness across classes" meaning="higher is better" tech="Macro-F1" />, numeric: true, sortValue: (r) => r.test_macro_f1 ?? null, render: (r) => <Score v={r.test_macro_f1} /> },
                  { key: 'nr', label: 'Healthy windows left alone', header: <Head plain="Healthy windows left alone" meaning={f ? `out of ${f.healthy_test_windows} healthy windows` : 'share of healthy windows'} tech="Normal recall" />, numeric: true, sortValue: (r) => r.test_normal_recall ?? null, render: (r) => <Score v={r.test_normal_recall} /> },
                  { key: 'auc', label: 'Ranks risky above safe', header: <Head plain="Ranks risky above safe" meaning="1.0 = perfect ranking" tech="ROC-AUC" />, numeric: true, sortValue: (r) => r.test_roc_auc ?? null, render: (r) => <Score v={r.test_roc_auc} /> },
                ]}
              />
              {done.length > TOP_N && (
                <div className="show-all">
                  <button type="button" className="btn secondary" aria-expanded={showAll} data-testid="lo2-show-all" onClick={() => setShowAll((v) => !v)}>{showAll ? `Show only the top ${TOP_N}` : `Show all models (${done.length})`}</button>
                </div>
              )}
              {nc.length > 0 && <div className="rel grey" data-testid="lo2-not-completed"><strong>Not completed:</strong> {Array.from(new Set(nc.map((r) => r.model))).join(', ')}.</div>}
              <p className="small muted">{d.note}</p>
            </div>
          )
        }}
      </Async>
    </section>
  )
}

const pc = (x: number) => `${(x * 100).toFixed(1)}%`

/** Three plain cards from test_metrics.csv (binary labels, all 647 test windows); ties are shown, not hidden. */
function SimpleResults({ models }: { models: ReturnType<typeof useApi<ModelsResponse>> }) {
  return (
    <>
      <p className="lede">We tried several kinds of model and scored each one once on recordings it had never seen. Here are the three that matter most, picked by the numbers below. No single model is best at everything.</p>
      <Async state={models} rows={4} label="Loading results" level={2}>
        {(m) => {
          const list = m.models.filter((x) => x.label_version === 'label_binary' && x.metrics['official/ALL'])
          const val = (x: ModelInfo, k: string) => (x.metrics['official/ALL'] as unknown as Record<string, number>)[k] ?? 0
          const pick = (k: string) => {
            const top = Math.max(...list.map((x) => val(x, k)))
            const tied = list.filter((x) => Math.abs(val(x, k) - top) < 1e-9)
            return { top, best: tied[0], others: tied.slice(1) }
          }
          const a = pick('recall_Anomaly'), n = pick('normal_recall'), b = pick('balanced_accuracy')
          const cards = [
            { id: 'catch', title: 'Best at catching failures', p: a, value: pc(a.top), line: (x: ModelInfo) => `It spotted ${pc(val(x, 'recall_Anomaly'))} of the windows that were really failing.`, price: (x: ModelInfo) => `The price: it called only ${pc(val(x, 'normal_recall'))} of the healthy windows healthy, so it raises many false alarms.` },
            { id: 'quiet', title: 'Fewest false alarms', p: n, value: pc(n.top), line: (x: ModelInfo) => `It correctly left ${pc(val(x, 'normal_recall'))} of the healthy windows alone.`, price: (x: ModelInfo) => `That is still low: about ${pc(1 - val(x, 'normal_recall'))} of healthy windows were flagged by mistake. It caught ${pc(val(x, 'recall_Anomaly'))} of the failures.` },
            { id: 'bal', title: 'Most balanced', p: b, value: pc(b.top), line: (x: ModelInfo) => `Average of the two scores above: ${pc(val(x, 'recall_Anomaly'))} of failures caught and ${pc(val(x, 'normal_recall'))} of healthy windows left alone.`, price: () => 'Decent at both jobs, but not perfect at either.' },
          ]
          return (
            <>
              <ul className="grid cols-3" style={{ listStyle: 'none', padding: 0, maxWidth: 'none' }}>
                {cards.map((c) => (
                  <li key={c.id} className="card stack" data-testid={`card-${c.id}`}>
                    <h2 style={{ margin: 0, fontSize: '1.15rem' }}>{c.title}</h2>
                    <div className="num" style={{ fontFamily: 'var(--font-serif)', fontSize: '2.6rem', fontWeight: 650, lineHeight: 1 }}>{c.value}</div>
                    <p style={{ margin: 0 }}><strong>{c.p.best.name}</strong>{c.p.others.length > 0 && <> (tied with {c.p.others.map((o) => o.name).join(', ')})</>}</p>
                    <p className="small" style={{ margin: 0 }}>{c.line(c.p.best)}</p>
                    <p className="small muted" style={{ margin: 0 }}>{c.price(c.p.best)}</p>
                  </li>
                ))}
              </ul>
              <div className="callout" style={{ marginTop: 'var(--space-5)' }}>
                <span className="label">How much to trust this</span>
                <ul>
                  <li>Only {list[0]?.metrics['official/ALL']?.support_Normal} of the {list[0]?.metrics['official/ALL']?.n} test windows are healthy ones, so the false-alarm numbers move a lot with a few windows. Switch to Detailed to see the ranges.</li>
                  <li>We picked these three by looking at the test results, so they look slightly better than they would on brand-new data.</li>
                  <li>On a completely different system (<Term k="lo2">LO2</Term>) none of the models worked. See <Link to="/examiners/limits">Limits</Link>.</li>
                </ul>
              </div>
            </>
          )
        }}
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
