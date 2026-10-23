import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { LabelToggle } from '../components/pickers'
import ResponsiveTable from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { fmt, int, range } from '../format'
import { Detailed } from '../mode'
import { usePageTitle } from '../usePageTitle'
import { Score } from './Results'
import type { HybridResponse, HybridRow, Label, Lo2LeaderboardResponse, ModelsResponse } from '../types'

interface Info { plain: string; tech: string; line: string }

/** Plain-language names and one-line ideas. Numbers never come from here. */
export const HYBRID_INFO: Record<string, Info> = {
  Transformer_BiLSTM: { plain: 'Look-around reader + two-way memory', tech: 'Transformer + BiLSTM', line: 'Compares the recent moments with each other, then reads them in order.' },
  AE_LSTM: { plain: 'Squeeze-and-rebuild + memory', tech: 'Autoencoder + LSTM', line: 'Summarises each moment; unusual moments are hard to rebuild, which is a clue.' },
  GNN_Transformer: { plain: 'Parts-map + look-around reader', tech: 'GNN + Transformer', line: 'Knows which readings belong together, then looks at how they change.' },
  CNN_LSTM: { plain: 'Pattern spotter + memory', tech: 'CNN + LSTM', line: 'Spots sudden jumps and remembers what came before.' },
  VAE_Transformer: { plain: 'Smooth squeezer + look-around reader', tech: 'VAE + Transformer', line: 'Squeezes each moment into a smooth summary, then compares the summaries.' },
  BERT_GNN: { plain: 'Log-message meaning + pattern map', tech: 'BERT-style text encoder + GNN', line: 'Turns kinds of log messages into numbers and learns from which ones appear together.' },
  Transformer_LSTM: { plain: 'Look-around reader + memory', tech: 'Transformer + LSTM', line: 'Compares the last few moments, then reads them in order.' },
  GNN_LSTM: { plain: 'Parts-map + memory', tech: 'GNN + LSTM', line: 'Learns which log sources move together, then reads the map over time.' },
  AE_LSTM_LO2: { plain: 'Squeeze-and-rebuild + memory', tech: 'Autoencoder + LSTM', line: 'Summarises each moment and reads the summaries in order.' },
  Multimodal_Transformer_MLP: { plain: 'Two readers (logs, metrics) + a mixer', tech: 'Multimodal Transformer + MLP', line: 'Reads logs and machine readings separately, then combines them.' },
}
const ANOMOD_ORDER = ['Transformer_BiLSTM', 'AE_LSTM', 'GNN_Transformer', 'CNN_LSTM', 'VAE_Transformer']
const LO2_ORDER = ['BERT_GNN', 'Transformer_LSTM', 'AE_LSTM', 'GNN_LSTM', 'Multimodal_Transformer_MLP']
const infoOf = (ds: 'anomod' | 'lo2', n: string) => (ds === 'lo2' && n === 'AE_LSTM' ? HYBRID_INFO.AE_LSTM_LO2 : HYBRID_INFO[n])

const PLAIN_FAMILY: Record<string, string> = { tree: 'Decision-tree model', linear: 'Linear model', deep: 'Neural network', ensemble: 'Combination of models' }
const LO2_FAMILY = (f: string) => (/classical|Optuna/.test(f) ? 'Decision-tree / linear model' : /stacking|voting/.test(f) ? 'Combination of models' : /deep/.test(f) ? 'Neural network' : 'Hybrid')

interface Row { key: string; name: string; hybrid: boolean; tag: string; ba: number | null; lo: number | null; hi: number | null; nr: number | null; auc: number | null }

function Card({ ds, name, row }: { ds: 'anomod' | 'lo2'; name: string; row?: HybridRow }) {
  const info = infoOf(ds, name)
  const done = row && row.status !== 'not completed' && typeof row.test_bal_acc === 'number'
  return (
    <li className="card hy-card" data-state={done ? 'done' : 'pending'} data-testid={`hybrid-${ds}-${name}`}>
      <h3>{info.plain}</h3>
      <p className="small muted" style={{ margin: 0 }}><Detailed><>{info.tech}. </></Detailed>{info.line}</p>
      {done ? (
        <dl className="hy-facts">
          <dt>Overall score</dt><dd><strong>{fmt(row.test_bal_acc)}</strong> ({range(row.test_bal_acc_ci_low, row.test_bal_acc_ci_high)})</dd>
          <dt>Healthy windows left alone</dt><dd>{fmt(row.test_normal_recall, 2)}</dd>
        </dl>
      ) : <p className="small" style={{ margin: 0 }}>No result available.</p>}
    </li>
  )
}

function Compare({ rows, caption }: { rows: Row[]; caption: string }) {
  const sorted = useMemo(() => [...rows].sort((a, b) => (b.ba ?? 0) - (a.ba ?? 0)), [rows])
  return (
    <ResponsiveTable<Row>
      caption={caption} rows={sorted} rowKey={(r) => r.key} initialSort={{ key: 'ba', dir: 'desc' }}
      columns={[
        { key: 'name', label: 'Model', nameCell: true, sortValue: (r) => r.name, render: (r) => <>{r.name} <span className="pills">{r.hybrid ? <span className="badge accent">Hybrid</span> : <span className="badge">{r.tag}</span>}</span></> },
        { key: 'ba', label: 'Overall score', numeric: true, sortValue: (r) => r.ba, render: (r) => <Score v={r.ba} low={r.lo} high={r.hi} /> },
        { key: 'nr', label: 'Healthy windows left alone', numeric: true, sortValue: (r) => r.nr, render: (r) => <Score v={r.nr} /> },
        { key: 'auc', label: 'Ranks risky above safe', numeric: true, sortValue: (r) => r.auc, render: (r) => <Score v={r.auc} /> },
      ]}
    />
  )
}

function DatasetBlock({ ds, hy }: { ds: 'anomod' | 'lo2'; hy: HybridResponse }) {
  const [label, setLabel] = useState<Label>('label_binary')
  const models = useApi<ModelsResponse>(ds === 'anomod' ? '/api/models' : null)
  const lo2 = useApi<Lo2LeaderboardResponse>(ds === 'lo2' ? '/api/lo2/leaderboard' : null)
  const d = hy[ds]
  const f = d.facts as any
  const hyRows = d.models.filter((m) => m.label_version === label)
  const order = ds === 'anomod' ? ANOMOD_ORDER : LO2_ORDER
  const rows: Row[] = useMemo(() => {
    const out: Row[] = []
    for (const h of d.models.filter((m) => m.label_version === label)) {
      if (h.status === 'not completed' || typeof h.test_bal_acc !== 'number') continue
      out.push({ key: 'h-' + h.model, name: infoOf(ds, h.model)?.plain ?? h.model, hybrid: true, tag: 'Hybrid', ba: h.test_bal_acc, lo: h.test_bal_acc_ci_low ?? null, hi: h.test_bal_acc_ci_high ?? null, nr: h.test_normal_recall ?? null, auc: h.test_roc_auc ?? null })
    }
    if (ds === 'anomod') {
      for (const m of models.data?.models ?? []) {
        if (m.label_version !== label || !m.metrics['official/ALL']) continue
        const x = m.metrics['official/ALL']
        out.push({ key: 'm-' + m.name, name: m.name, hybrid: false, tag: PLAIN_FAMILY[m.family] ?? m.family, ba: x.balanced_accuracy ?? null, lo: m.ci['ALL/balanced_accuracy']?.low ?? null, hi: m.ci['ALL/balanced_accuracy']?.high ?? null, nr: x.normal_recall ?? null, auc: x.roc_auc ?? null })
      }
    } else {
      for (const r of lo2.data?.rows ?? []) {
        if (r.label_version !== label || r.status === 'not completed' || (r.family ?? '').startsWith('(a)')) continue
        out.push({ key: 'm-' + r.model, name: r.model, hybrid: false, tag: LO2_FAMILY(r.family ?? ''), ba: r.test_bal_acc ?? null, lo: r.test_bal_acc_ci_low ?? null, hi: r.test_bal_acc_ci_high ?? null, nr: r.test_normal_recall ?? null, auc: r.test_roc_auc ?? null })
      }
    }
    return out
  }, [d.models, models.data, lo2.data, label, ds])
  const best = [...rows].sort((a, b) => (b.ba ?? 0) - (a.ba ?? 0))
  const bestH = best.find((r) => r.hybrid)
  const bestO = best.find((r) => !r.hybrid)
  const title = ds === 'anomod' ? 'AnoMod data (recorded benchmark)' : 'LO2 data'
  const facts = ds === 'anomod'
    ? `${int(f?.windows)} windows from ${f?.runs} runs; ${int(f?.test)} test windows, ${f?.healthy_test_windows} of them healthy.`
    : `${int(f?.windows)} windows from ${f?.runs} runs; ${int(f?.test)} test windows (one whole run), ${f?.healthy_test_windows} of them healthy. Small sample: read the scores with care.`
  return (
    <section aria-labelledby={`h-${ds}`}>
      <h2 id={`h-${ds}`}>{title}</h2>
      <p className="muted">{facts}</p>
      <div className="row" style={{ marginBottom: 'var(--space-4)' }}><LabelToggle value={label} onChange={setLabel} /></div>
      <ul className="hy-grid" aria-label={`The five hybrid models on ${title}`}>
        {order.map((n) => <Card key={n} ds={ds} name={n} row={hyRows.find((r) => r.model === n)} />)}
      </ul>
      <h3 style={{ marginTop: 'var(--space-5)' }}>All models side by side</h3>
      <Async state={(ds === 'anomod' ? models : lo2) as any} rows={5} label="Loading the comparison">
        {() => <Compare rows={rows} caption={`Test results on ${title}, ${label === 'label_binary' ? 'healthy or not' : 'by kind of problem'}. Hybrid models are tagged; click a column title to sort.`} />}
      </Async>
      {bestH && bestO && (
        <div className="callout" style={{ marginTop: 'var(--space-4)' }} data-testid={`verdict-${ds}`}>
          <span className="label">How do the hybrids compare?</span>
          <p style={{ margin: 0 }}>
            Best hybrid: <strong>{bestH.name}</strong>, overall score {fmt(bestH.ba)} (range {range(bestH.lo, bestH.hi)}). Best earlier model: <strong>{bestO.name}</strong>, {fmt(bestO.ba)} (range {range(bestO.lo, bestO.hi)}).{' '}
            {(bestH.ba ?? 0) > (bestO.ba ?? 0) ? 'The best hybrid scored higher' : (bestH.ba ?? 0) === (bestO.ba ?? 0) ? 'They scored the same' : 'The best earlier model scored higher'}; where the ranges overlap, the difference is not reliable.
          </p>
        </div>
      )}
    </section>
  )
}

export default function Hybrids() {
  usePageTitle('Hybrid models')
  const api = useApi<HybridResponse>('/api/hybrid')
  return (
    <>
      <h1>Hybrid models</h1>
      <p className="lede">Ten two-part neural networks: five trained on the AnoMod data and five on the LO2 data. Each is shown next to the models we built earlier, so you can compare them directly.</p>
      <div className="callout accent how-to-read"><span className="label">How to read this page</span>
        <ul>
          <li><strong>Overall score:</strong> 0.5 is a coin flip, 1.0 is perfect. The range beside it is how much the score could move with different luck.</li>
          <li><strong>Healthy windows left alone:</strong> how many calm moments did not cause a false alarm.</li>
          <li>Rows tagged <span className="badge accent">Hybrid</span> are the new models; the others are the earlier ones.</li>
        </ul>
      </div>
      <Async state={api} rows={6} label="Loading hybrid results">
        {(d) => (
          <div className="stack" style={{ marginTop: 'var(--space-5)' }}>
            <DatasetBlock ds="anomod" hy={d} />
            <DatasetBlock ds="lo2" hy={d} />
            <div className="callout" role="note" data-testid="no-transfer"><span className="label">Good to know</span>
              <ul>
                <li data-testid="fold-note">{d.fold_note}</li>
                {d.notes.slice(1).map((n) => <li key={n}>{n}</li>)}
                <li>The best model in any list was picked by looking at the test results, so its lead looks slightly better than it would on new data.</li>
              </ul>
            </div>
            <p className="small muted">More detail on the <Link to="/examiners/results">Results</Link> page; the checks are on the <Link to="/proofs">Proof</Link> page.</p>
          </div>
        )}
      </Async>
    </>
  )
}
