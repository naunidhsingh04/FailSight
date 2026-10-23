import { Link } from 'react-router-dom'
import { useApi } from '../api'
import ResponsiveTable from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { int } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { BothResponse, BothRow } from '../types'

interface Info { plain: string; tech: string; line: string }

/** Plain-language names and one-line ideas. Numbers never come from here. */
export const HYBRID_INFO: Record<string, Info> = {
  Transformer_BiLSTM: { plain: 'Transformer + BiLSTM', tech: 'Transformer + BiLSTM', line: 'Compares the recent moments with each other, then reads them in order.' },
  AE_LSTM: { plain: 'Autoencoder + LSTM', tech: 'Autoencoder + LSTM', line: 'Summarises each moment; unusual moments are hard to rebuild, which is a clue.' },
  GNN_Transformer: { plain: 'GNN + Transformer', tech: 'GNN + Transformer', line: 'Knows which readings belong together, then looks at how they change.' },
  CNN_LSTM: { plain: 'CNN + LSTM', tech: 'CNN + LSTM', line: 'Spots sudden jumps and remembers what came before.' },
  VAE_Transformer: { plain: 'VAE + Transformer', tech: 'VAE + Transformer', line: 'Squeezes each moment into a smooth summary, then compares the summaries.' },
  BERT_GNN: { plain: 'BERT + GNN', tech: 'BERT + GNN', line: 'Turns kinds of log messages into numbers and learns from which ones appear together.' },
  Transformer_LSTM: { plain: 'Transformer + LSTM', tech: 'Transformer + LSTM', line: 'Compares the last few moments, then reads them in order.' },
  GNN_LSTM: { plain: 'GNN + LSTM', tech: 'GNN + LSTM', line: 'Learns which log sources move together, then reads the map over time.' },
  AE_LSTM_LO2: { plain: 'Autoencoder + LSTM', tech: 'Autoencoder + LSTM', line: 'Summarises each moment and reads the summaries in order.' },
  Multimodal_Transformer_MLP: { plain: 'Multimodal Transformer + MLP', tech: 'Multimodal Transformer + MLP', line: 'Reads logs and machine readings separately, then combines them.' },
}

const pct = (x: number | null | undefined) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : '–')
/** a percentage with a thin bar; the 95% range is shown as text */
function Pct({ v, low, high }: { v?: number | null; low?: number | null; high?: number | null }) {
  const c = (x: number) => Math.max(0, Math.min(1, x)) * 100
  return (
    <div className="score">
      <div className="score-num"><strong>{pct(v)}</strong>{low != null && high != null && <span className="muted small"> ({pct(low)} to {pct(high)})</span>}</div>
      {typeof v === 'number' && <div className="sbar" aria-hidden="true"><span className="sbar-fill" style={{ width: `${c(v)}%` }} /></div>}
    </div>
  )
}

const HYBRID_KEYS = ['Transformer_BiLSTM', 'AE_LSTM', 'GNN_Transformer', 'CNN_LSTM', 'VAE_Transformer', 'BERT_GNN', 'Transformer_LSTM', 'GNN_LSTM', 'Multimodal_Transformer_MLP']

export function BothTable({ rows }: { rows: BothRow[] }) {
  const done = rows.filter((r) => r.status !== 'not completed' && typeof r.test_bal_acc === 'number')
  const nameOf = (r: BothRow) => (r.family === 'Hybrid model' ? HYBRID_INFO[r.model]?.tech ?? r.model : r.model)
  return (
    <ResponsiveTable<BothRow>
      caption="Every model, trained on AnoMod and LO2 together and tested on the test windows of both. Hybrid models are tagged; click a column title to sort."
      rows={done} rowKey={(r) => r.model} initialSort={{ key: 'ba', dir: 'desc' }}
      columns={[
        { key: 'name', label: 'Model', nameCell: true, sortValue: (r) => nameOf(r), render: (r) => <>{nameOf(r)} <span className="pills">{r.family === 'Hybrid model' ? <span className="badge accent">Hybrid</span> : <span className="badge">{r.family}</span>}</span></> },
        { key: 'ba', label: 'Overall score', numeric: true, sortValue: (r) => r.test_bal_acc ?? null, render: (r) => <Pct v={r.test_bal_acc} low={r.test_bal_acc_ci_low} high={r.test_bal_acc_ci_high} /> },
        { key: 'nr', label: 'Healthy windows left alone', numeric: true, sortValue: (r) => r.test_normal_recall ?? null, render: (r) => <Pct v={r.test_normal_recall} /> },
        { key: 'a', label: 'Score on AnoMod windows', numeric: true, sortValue: (r) => r.ba_AnoMod ?? null, render: (r) => <Pct v={r.ba_AnoMod} /> },
        { key: 'l', label: 'Score on LO2 windows', numeric: true, sortValue: (r) => r.ba_LO2 ?? null, render: (r) => <Pct v={r.ba_LO2} /> },
      ]}
    />
  )
}

export default function Hybrids() {
  usePageTitle('Hybrid models')
  const api = useApi<BothResponse>('/api/both')
  return (
    <>
      <h1>Hybrid models</h1>
      <p className="lede">Every model, the hybrid ones and the earlier kinds, is trained on the AnoMod and LO2 data together and tested on the test windows of both. One table, so you can compare them directly.</p>
      <div className="callout accent how-to-read"><span className="label">How to read this page</span>
        <ul>
          <li><strong>Overall score:</strong> 0.5 is a coin flip, 1.0 is perfect. The range beside it is how much the score could move with different luck.</li>
          <li><strong>Healthy windows left alone:</strong> how many calm moments did not cause a false alarm.</li>
          <li>The last two columns split the same test into its AnoMod part and its LO2 part. Rows tagged <span className="badge accent">Hybrid</span> are the new models.</li>
        </ul>
      </div>
      <Async state={api} rows={6} label="Loading results">
        {(d) => (
          <div className="stack" style={{ marginTop: 'var(--space-5)' }}>
            {!d.available && <div className="state"><h3>No combined results yet</h3><p>{d.note}</p></div>}
            {d.available && (() => {
              const done = d.rows.filter((r) => r.status !== 'not completed' && typeof r.test_bal_acc === 'number').sort((a, b) => (b.test_bal_acc ?? 0) - (a.test_bal_acc ?? 0))
              const bh = done.find((r) => r.family === 'Hybrid model'), bo = done.find((r) => r.family !== 'Hybrid model')
              const nm = (r?: BothRow) => (r ? (r.family === 'Hybrid model' ? HYBRID_INFO[r.model]?.tech ?? r.model : r.model) : '')
              return (
                <>
                  {d.facts && <p className="muted">Training windows: {int(d.facts.train_anomod)} from AnoMod + {int(d.facts.train_lo2)} from LO2. Test windows: {int(d.facts.test_anomod)} from AnoMod ({d.facts.healthy_test_anomod} healthy) + {int(d.facts.test_lo2)} from LO2 ({d.facts.healthy_test_lo2} healthy). Healthy-or-not question.</p>}
                  <BothTable rows={d.rows} />
                  {bh && bo && (
                    <div className="callout" data-testid="verdict-both"><span className="label">How do the hybrids compare?</span>
                      <p style={{ margin: 0 }}>Best hybrid: <strong>{nm(bh)}</strong>, overall score {pct(bh.test_bal_acc)} (range {pct(bh.test_bal_acc_ci_low)} to {pct(bh.test_bal_acc_ci_high)}). Best other model: <strong>{nm(bo)}</strong>, {pct(bo.test_bal_acc)} (range {pct(bo.test_bal_acc_ci_low)} to {pct(bo.test_bal_acc_ci_high)}). Where the ranges overlap the difference is not reliable.</p>
                    </div>
                  )}
                </>
              )
            })()}
            <h2>What every number and setting means</h2>
            <dl className="hy-facts" style={{ gridTemplateColumns: 'minmax(10rem, 16rem) 1fr', rowGap: '0.6rem' }} data-testid="glossary">
              <dt>Overall score</dt><dd>Balanced accuracy: the average of two shares, how many risky windows were caught and how many healthy windows were left alone. 50% is a coin flip, 100% is perfect. It is fairer than plain accuracy because healthy windows are rare.</dd>
              <dt>Range in brackets</dt><dd>95% range from 1000 re-draws of the test windows (in blocks of neighbouring windows). A wide range means the score is uncertain; overlapping ranges mean two models are not clearly different.</dd>
              <dt>Healthy windows left alone</dt><dd>Of the healthy test windows, the share the model correctly called healthy. Low means many false alarms.</dd>
              <dt>Score on AnoMod / LO2 windows</dt><dd>The same overall score, computed only on the AnoMod part or only on the LO2 part of the pooled test windows.</dd>
              <dt>Training and test windows</dt><dd>A window is a short slice of recorded system activity (30 seconds for AnoMod, 5 seconds for LO2). 80% of each data set is used to train, 20% to test; the test windows are never used to choose anything and are scored once.</dd>
              <dt>Cross-validation (folds)</dt><dd>The training windows are cut into parts; the model is trained on all but one part and checked on the held-out part, in turn. Used to choose settings. The earlier models used 10 parts, the hybrids 5.</dd>
              <dt>SMOTE / class weights</dt><dd>Two ways to cope with healthy windows being rare: make extra look-alike healthy examples (training windows only) or count mistakes on healthy windows more. Chosen per model by validation.</dd>
              <dt>Overfitting</dt><dd>A model that memorises its training windows scores much higher on them than on new ones. We flag it (gap above 0.10) and retrain more gently.</dd>
              <dt>Hybrid model</dt><dd>One neural network with two parts trained together: the first part prepares the numbers, the second reads them and answers.</dd>
              <dt>Tag next to a model</dt><dd>Decision-tree / linear model, Neural network, Combination of models, or Hybrid: the kind of model it is.</dd>
            </dl>
            <h2>The hybrid models in one line each</h2>
            <ul className="hy-grid">
              {HYBRID_KEYS.map((k) => <li key={k} className="card hy-card"><h3>{HYBRID_INFO[k].tech}</h3><p className="small muted" style={{ margin: 0 }}>{HYBRID_INFO[k].line}</p></li>)}
            </ul>
            <p className="small muted">Ten hybrids were built: five for AnoMod and five for LO2 (Autoencoder + LSTM is in both lists), so nine different kinds appear here. {d.note}</p>
            <div className="callout" role="note" data-testid="no-transfer"><span className="label">Good to know</span>
              <ul>
                <li>The healthy-or-not question is the only one both data sets share, so only that is compared here. The five-kinds-of-problem results are on the Results page for AnoMod.</li>
                <li>Nothing here forecasts failures or the time until a failure; that is future work.</li>
              </ul>
            </div>
            <p className="small muted">More on the <Link to="/examiners/results">Results</Link> page; the checks are on the <Link to="/proofs">Proof</Link> page.</p>
          </div>
        )}
      </Async>
    </>
  )
}
