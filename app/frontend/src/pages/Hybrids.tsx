import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { RangePlot } from '../components/Charts'
import { LabelToggle } from '../components/pickers'
import { Async } from '../components/States'
import { fmt, int, range } from '../format'
import { Detailed, Simple } from '../mode'
import { usePageTitle } from '../usePageTitle'
import type { HybridResponse, HybridRow, Label } from '../types'

interface Info { plain: string; tech: string; first: string; second: string; line: string; note?: string }

/** Plain-language description of each hybrid: what its first part does, what its second part does, and the one-line idea. Numbers never come from here. */
export const HYBRID_INFO: Record<string, Info> = {
  Transformer_BiLSTM: { plain: 'Look-around reader + two-way memory', tech: 'Transformer + BiLSTM', first: 'Looks at six recent windows at once and decides which matter', second: 'Reads them in order, forwards and backwards', line: 'First it lets the windows compare notes, then it reads the story in time order.' },
  AE_LSTM: { plain: 'Squeeze-and-rebuild + memory', tech: 'Autoencoder + LSTM', first: 'Squeezes each window into a short summary and tries to rebuild it', second: 'Reads the six summaries in order', line: 'If a window cannot be rebuilt well, it is probably unusual; that hint is added to the learning.' },
  GNN_Transformer: { plain: 'Parts-map + look-around reader', tech: 'GNN + Transformer', first: 'Treats the system as a map of connected parts and lets neighbours share readings', second: 'Looks across six windows to see how the map changes', line: 'It knows which readings belong together (cpu with cpu, network with network).' },
  CNN_LSTM: { plain: 'Pattern spotter + memory', tech: 'CNN + LSTM', first: 'Spots short patterns such as a sudden jump between windows', second: 'Reads the patterns in time order', line: 'A small pattern detector feeds a model that remembers what came before.' },
  VAE_Transformer: { plain: 'Smooth squeezer + look-around reader', tech: 'VAE + Transformer', first: 'Squeezes each window into a smooth, slightly fuzzy summary', second: 'Lets the six summaries compare notes', line: 'The fuzziness keeps the summary from memorising small details.' },
  BERT_GNN: { plain: 'Meaning of log lines + pattern map', tech: 'BERT-style text encoder (frozen) + GNN', first: 'Turns each kind of log message into a fixed list of numbers (frozen, never trained)', second: 'A map of log messages that appear together; it learns from the map', line: 'The text meaning comes from a simple word-statistics method, because a downloadable BERT model was not available.', note: 'The text encoder is TF-IDF + truncated SVD (frozen), not a real BERT.' },
  Transformer_LSTM: { plain: 'Look-around reader + memory', tech: 'Transformer + LSTM', first: 'Looks at the last three windows at once', second: 'Reads them in order', line: 'The same idea as the AnoMod one, on LO2’s own signals.' },
  GNN_LSTM: { plain: 'Parts-map + memory', tech: 'GNN + LSTM', first: 'A map of the logs of the seven services, the database and the load generator; connected when their activity moves together', second: 'Reads the map summaries in order', line: 'It learns which sources of log lines behave together.' },
  Multimodal_Transformer_MLP: { plain: 'Two readers (logs and metrics) + a mixer', tech: 'Multimodal Transformer + MLP', first: 'One reader for log numbers and one for machine metrics', second: 'A small mixer that joins what both readers found', line: 'Logs and metrics are looked at separately first, then combined.' },
}
const LO2_ORDER = ['BERT_GNN', 'Transformer_LSTM', 'AE_LSTM', 'GNN_LSTM', 'Multimodal_Transformer_MLP']
const ANOMOD_ORDER = ['Transformer_BiLSTM', 'AE_LSTM', 'GNN_Transformer', 'CNN_LSTM', 'VAE_Transformer']
export const AE_LSTM_LO2: Info = { ...HYBRID_INFO.AE_LSTM, line: 'Same idea as on AnoMod, trained separately on LO2’s own windows.' }

function secs(s?: number | null) {
  if (s === null || s === undefined) return '–'
  return s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`
}

function Card({ ds, name, row, facts, reliab }: { ds: 'anomod' | 'lo2'; name: string; row?: HybridRow; facts: string; reliab: string }) {
  const info = ds === 'lo2' && name === 'AE_LSTM' ? AE_LSTM_LO2 : HYBRID_INFO[name]
  const done = row && row.status !== 'not completed' && typeof row.test_bal_acc === 'number'
  const small = ds === 'lo2'
  return (
    <li className="card hy-card" data-state={done ? 'done' : 'pending'} data-testid={`hybrid-${ds}-${name}`}>
      <h3>{info.plain}</h3>
      <p className="small muted" style={{ margin: 0 }}><Detailed><>{info.tech}. {info.note ? `${info.note} ` : ''}</></Detailed>{info.line}</p>
      <div className="wire" role="img" aria-label={`First part: ${info.first}. Second part: ${info.second}. Then the answer.`}>
        <div className="box"><b>First part</b>{info.first}</div>
        <span className="arrow" aria-hidden="true">&rarr;</span>
        <div className="box"><b>Second part</b>{info.second}</div>
        <span className="arrow" aria-hidden="true">&rarr;</span>
        <div className="box ans"><b>Answer</b>healthy or not</div>
      </div>
      {done ? (
        <dl className="hy-facts">
          <dt>Test score</dt><dd><strong>{fmt(row.test_bal_acc)}</strong> (95% range {range(row.test_bal_acc_ci_low, row.test_bal_acc_ci_high)})</dd>
          <dt>Healthy windows left alone</dt><dd>{fmt(row.test_normal_recall, 2)}{row.normal_recall_ci_low != null && <> ({range(row.normal_recall_ci_low, row.normal_recall_ci_high)})</>}</dd>
          <Detailed><><dt>ROC-AUC</dt><dd>{fmt(row.test_roc_auc)}</dd><dt>Macro-F1</dt><dd>{fmt(row.test_macro_f1)}</dd></></Detailed>
          <dt>Ran on</dt><dd>{facts}</dd>
          <dt>Training time</dt><dd>{secs(row.train_seconds_total ?? row.train_secs)}</dd>
          <dt>Folds used</dt><dd>{row.folds ?? 5}-fold</dd>
        </dl>
      ) : <p className="small" style={{ margin: 0 }}>Not completed because of the time limit.</p>}
      {done && row.status === 'did not learn' && <div className="rel amber">This model did not learn (its validation score stayed near chance). Reported as it is.</div>}
      {!done && <div className="rel grey">Not completed: no result is shown because none was produced in the time available.</div>}
      {done && small && <div className="rel amber">Unreliable: very small sample ({reliab}).</div>}
      {done && !small && <div className="rel ok">{reliab}, so the range is wide.</div>}
    </li>
  )
}

function DatasetBlock({ ds, data }: { ds: 'anomod' | 'lo2'; data: HybridResponse }) {
  const [label, setLabel] = useState<Label>('label_binary')
  const d = data[ds]
  const order = ds === 'anomod' ? ANOMOD_ORDER : LO2_ORDER
  const rows = d.models.filter((m) => m.label_version === label)
  const f = d.facts as any
  const factsText = ds === 'anomod'
    ? `${int(f?.windows)} windows (30 s) from ${f?.runs} runs: ${int(f?.train)} to train, ${int(f?.test)} to test`
    : `${int(f?.windows)} windows (5 s) from ${f?.runs} whole runs, ${fmt(f?.megabytes, 1)} MB: ${int(f?.train)} to train, ${int(f?.test)} to test`
  const done = rows.filter((r) => r.status !== 'not completed' && typeof r.test_bal_acc === 'number')
  const best = [...done].sort((a, b) => (b.test_bal_acc ?? 0) - (a.test_bal_acc ?? 0))[0]
  const cmp = (data.anomod.comparison ?? {})[label]
  const reliab = ds === 'anomod' ? `Only ${f?.healthy_test_windows} healthy test windows` : `${f?.runs} runs, only ${f?.healthy_test_windows} healthy test windows`
  return (
    <section aria-labelledby={`h-${ds}`}>
      <h2 id={`h-${ds}`}>{ds === 'anomod' ? 'On the recorded benchmark data (AnoMod)' : 'On a slice of a second data set (LO2)'}</h2>
      <p className="muted">{factsText}. {ds === 'lo2' && <>Trained completely separately from AnoMod: its own windows, its own preprocessing, its own models. These results say nothing about transfer from AnoMod.</>}</p>
      <div className="row" style={{ marginBottom: 'var(--space-4)' }}><LabelToggle value={label} onChange={setLabel} /></div>
      {!d.available && <div className="state"><h3>No hybrid results yet</h3><p>The result files for this data set are not on this computer.</p></div>}
      <ul className="hy-grid" aria-label={`Five hybrid models on ${ds === 'anomod' ? 'AnoMod' : 'LO2'}`}>
        {order.map((n) => <Card key={n} ds={ds} name={n} row={rows.find((r) => r.model === n)} facts={factsText} reliab={reliab} />)}
      </ul>
      {done.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--space-4)' }}>
          <RangePlot ariaLabel={`Test balanced accuracy with 95% range for the hybrid models on ${ds === 'anomod' ? 'AnoMod' : 'LO2'}`} title="Overall score on the test windows (dot) with its 95% range (line). 0.5 is a coin flip, 1.0 is perfect."
            items={[...done].sort((a, b) => (b.test_bal_acc ?? 0) - (a.test_bal_acc ?? 0)).map((r) => ({ label: r.model.replace(/_/g, ' + ').replace('Multimodal + Transformer + MLP', 'Multimodal Transformer + MLP'), point: r.test_bal_acc ?? null, low: r.test_bal_acc_ci_low ?? null, high: r.test_bal_acc_ci_high ?? null }))} />
        </div>
      )}
      {best && (
        <div className="callout" style={{ marginTop: 'var(--space-4)' }} data-testid={`verdict-${ds}`}>
          <span className="label">Which one was best?</span>
          <p style={{ marginTop: 0 }}>
            On the {label === 'label_binary' ? 'healthy-or-not' : 'five-class'} question the best hybrid was <strong>{HYBRID_INFO[best.model]?.tech ?? best.model}</strong> with an overall score of {fmt(best.test_bal_acc)} (95% range {range(best.test_bal_acc_ci_low, best.test_bal_acc_ci_high)}).
            {ds === 'anomod' && cmp && (
              <> The best of the 36 earlier models, {cmp.best_existing}, scored {fmt(cmp.best_existing_ba)}{cmp.best_existing_low != null && <> ({range(cmp.best_existing_low, cmp.best_existing_high)})</>}. {cmp.hybrid_beats_best_existing ? 'The best hybrid scored higher' : 'No hybrid beat the best earlier model'}, and the two ranges {cmp.ranges_overlap ? 'overlap, so we cannot say one is really better' : 'do not overlap'}. Small, table-like data with only 36 healthy test windows is where tree models are hard to beat.</>
            )}
            {ds === 'lo2' && <> LO2 is a very small sample, so this ranking is not reliable.</>}
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
      <p className="lede">A hybrid model is one model made of two parts that learn together: the first part prepares what it sees, the second part reads that and gives the answer. We built five on the recorded benchmark data and five on a slice of a second data set.</p>
      <Simple>
        <div className="callout accent how-to-read"><span className="label">How to read this page</span>
          <ul>
            <li>Each card is one model. The three boxes show how it works: first part, second part, answer.</li>
            <li><strong>Test score</strong>: 0.5 is a coin flip, 1.0 is perfect. The range beside it is how much the score could move with different luck (95% range).</li>
            <li><strong>Healthy windows left alone</strong>: how many calm moments did not cause a false alarm.</li>
            <li>Grey cards are models that were not finished. Amber notes mean a very small sample.</li>
          </ul>
        </div>
      </Simple>
      <Detailed>
        <div className="callout accent how-to-read"><span className="label">How to read this page</span>
          <ul>
            <li>Test score = balanced accuracy on the untouched test windows, with a 1000-resample block-bootstrap 95% range. Healthy windows left alone = recall of the Normal class.</li>
            <li>Every hybrid is one model class: first part, second part and head are trained end to end (the BERT-style encoder on LO2 is frozen by design). Settings were chosen by validation on the training part only; the test part was scored once per model.</li>
            <li>Training: SMOTE or class weights (chosen by validation), early stopping (patience 6, at most 40 epochs), label smoothing, batch size 32, seed 42.</li>
          </ul>
        </div>
      </Detailed>
      <Async state={api} rows={6} label="Loading hybrid results">
        {(d) => (
          <div className="stack" style={{ marginTop: 'var(--space-5)' }}>
            <div className="callout" role="note"><span className="label">Folds and test windows</span><p style={{ margin: 0 }} data-testid="fold-note">{d.fold_note}</p></div>
            <DatasetBlock ds="anomod" data={d} />
            <DatasetBlock ds="lo2" data={d} />
            <div className="callout" role="note" data-testid="no-transfer"><span className="label">What this does not show</span>
              <ul>
                {d.notes.slice(1).map((n) => <li key={n}>{n}</li>)}
                <li>The hybrids are compared with the 36 earlier models on the same 647 test windows; the lead of the best model on any list is optimistic because it was picked on the test set.</li>
              </ul>
            </div>
            <p className="small muted">The full numbers are on the <Link to="/examiners/results">Results</Link> page and the checks are on the <Link to="/proofs">Proof</Link> page (P20 to P23).</p>
          </div>
        )}
      </Async>
    </>
  )
}


