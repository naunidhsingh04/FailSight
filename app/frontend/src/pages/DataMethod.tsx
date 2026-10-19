import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { BeforeAfterBars } from '../components/Charts'
import ResponsiveTable from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { fmt, int, pct } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { DataResponse, Label } from '../types'

const LEVELS = ['Normal', 'Performance', 'Service', 'Database', 'Code']

function lk(d: DataResponse, check: string, scope: string, item: string) {
  return d.leakage.find((r) => r.check === check && r.scope === scope && r.item.includes(item))
}

export default function DataMethod() {
  usePageTitle('Data & method')
  const data = useApi<DataResponse>('/api/data')
  return (
    <>
      <h1>Data &amp; method</h1>
      <p className="lede">What the models were trained on, how the data was split, and what the leakage checks found. The checks found real problems; they are reported here as measured.</p>
      <Async state={data} rows={8} label="Loading data and method">{(d) => <Body d={d} />}</Async>
    </>
  )
}

function Body({ d }: { d: DataResponse }) {
  const ds = d.dataset
  const splitRows = useMemo(() => LEVELS.map((c) => {
    const f = (s: string, sp: string) => d.split_counts.find((r) => r.system === s && r.split === sp && r.level === c)?.windows ?? 0
    return { cls: c, snTrain: f('SN', 'train'), snTest: f('SN', 'test'), ttTrain: f('TT', 'train'), ttTest: f('TT', 'test') }
  }), [d])
  const foldRows = useMemo(() => Array.from({ length: d.folds.n_folds }, (_, i) => {
    const fold = i + 1
    const row: Record<string, number> = { fold }
    LEVELS.forEach((c) => { row[c] = d.fold_validation_counts.find((r) => r.fold === fold && r.level === c)?.validation_windows ?? 0 })
    row.total = d.folds.rows_per_fold.find((r) => r.fold === fold)?.windows ?? 0
    return row
  }), [d])
  const smote = (lab: string) => d.smote_final_train.filter((r) => r.label_version === lab)
  const dup = lk(d, 'duplicates', 'raw features+system', 'identical to a train row')
  const dupPre = lk(d, 'duplicates', 'preprocessed vectors', 'identical')
  const ovSN = lk(d, 'time_overlap', 'SN', 'SAME run')
  const ovTT = lk(d, 'time_overlap', 'TT', 'SAME run')
  const runs = lk(d, 'run_overlap', 'ALL', 'runs present in both')
  const blocks = lk(d, 'run_overlap', 'ALL', 'blocks present in both')
  const forb = lk(d, 'forbidden_columns', 'model input names', 'whose name contains')
  const sfBin = d.single_feature_max.find((r) => r.scope === 'label_binary')
  const sfLev = d.single_feature_max.find((r) => r.scope === 'label_level')
  const ex = d.explainability

  return (
    <>
      <section aria-labelledby="ds-h">
        <h2 id="ds-h" style={{ marginTop: 0 }}>The dataset</h2>
        <div className="grid cols-4">
          <div className="stat"><div className="num">{int(ds.windows)}</div><div className="cap">windows ({ds.window_seconds} s long, {ds.step_seconds} s step); SN {int(ds.systems.SN)}, TT {int(ds.systems.TT)}</div></div>
          <div className="stat"><div className="num">{ds.runs}</div><div className="cap">scenario runs (one run = one failure type of one system, or the normal baseline)</div></div>
          <div className="stat"><div className="num">{ds.raw_features}</div><div className="cap">raw features: API, log and metric groups; {ds.dropped_constant} are constant in training and dropped</div></div>
          <div className="stat"><div className="num">{ds.model_inputs}</div><div className="cap">model inputs after preprocessing (adds 3 was-missing flags and the system flag)</div></div>
        </div>
        <p style={{ marginTop: 'var(--space-4)' }}>
          Features kept per group: {Object.entries(ds.feature_groups).map(([k, v]) => `${k} ${v}`).join(', ')}. {ds.log1p_features} heavy-tailed features get a log1p transform. Statistics for imputation, clipping (0.5 / 99.5 percentiles) and scaling are fitted per system on training rows only.
        </p>
      </section>

      <section aria-labelledby="split-h">
        <h2 id="split-h">The 80:20 split</h2>
        <p>Windows of each run are cut into blocks of {ds.block_windows} consecutive windows; blocks are shuffled per system (seed 42) and {pct(ds.train_share, 2)} of the windows go to training ({int(ds.train)}), the rest to the untouched test set ({int(ds.test)}). Every run contributes blocks to both sides.</p>
        <ResponsiveTable
          caption="Windows per class, system and split"
          rows={splitRows} rowKey={(r) => r.cls}
          columns={[
            { key: 'c', label: 'Class', nameCell: true, render: (r) => r.cls },
            { key: 'a', label: 'SN train', numeric: true, render: (r) => int(r.snTrain) },
            { key: 'b', label: 'SN test', numeric: true, render: (r) => int(r.snTest) },
            { key: 'c2', label: 'TT train', numeric: true, render: (r) => int(r.ttTrain) },
            { key: 'd', label: 'TT test', numeric: true, render: (r) => int(r.ttTest) },
          ]}
        />
      </section>

      <section aria-labelledby="smote-h">
        <h2 id="smote-h">Class counts before and after SMOTE</h2>
        <p>Normal windows are rare. SMOTE adds synthetic minority windows to the <em>training part only</em> (inside each fold when cross-validating); the test set keeps its natural imbalance. Counts below are for the full training part.</p>
        <div className="grid cols-2">
          {(['binary', 'level'] as const).map((lab) => (
            <div className="card" key={lab}>
              <h3 style={{ marginTop: 0, fontSize: '1.05rem' }}>{lab === 'binary' ? 'Binary labels' : 'Five-class labels'}</h3>
              <BeforeAfterBars ariaLabel={`Training windows per class before and after SMOTE, ${lab} labels`} aName="before SMOTE (real windows)" bName="after SMOTE"
                groups={smote(lab).map((r) => ({ label: r.class, a: r.train_raw, b: r.train_smote }))} />
            </div>
          ))}
        </div>
      </section>

      <section aria-labelledby="folds-h">
        <h2 id="folds-h">Run-aware cross-validation</h2>
        <p>{d.folds.n_folds}-fold CV on the training part with windows grouped by <code>{d.folds.group_type}</code>: a validation block never shares a block with training. Plain run-grouping was impossible because every class has fewer than ten runs, so blocks were used as groups; windows of neighbouring blocks still overlap by 20 s, so CV scores remain somewhat optimistic and the test set is the main result.</p>
        <ResponsiveTable
          caption="Validation windows per fold and class (class of each window derived from its scenario name)"
          rows={foldRows} rowKey={(r) => String(r.fold)}
          columns={[
            { key: 'f', label: 'Fold', nameCell: true, render: (r) => String(r.fold) },
            ...LEVELS.map((c) => ({ key: c, label: c, numeric: true, render: (r: Record<string, number>) => String(r[c]) })),
            { key: 't', label: 'Total', numeric: true, render: (r) => String(r.total) },
          ]}
        />
      </section>

      <section aria-labelledby="leak-h">
        <h2 id="leak-h">Leakage checks and what they found</h2>
        <div className="callout">
          <span className="label">Findings</span>
          <ul>
            {runs && blocks && <li><strong>All {String(runs.value)} runs appear in both train and test</strong>, by design of the block split; no block is on both sides ({String(blocks.value)} shared blocks). The test set therefore measures generalisation to unseen <em>time blocks of known scenarios</em>, not to unseen failure types.</li>}
            {ovSN && ovTT && <li><strong>Time overlap:</strong> {String(ovSN.value)} SN and {String(ovTT.value)} TT training windows overlap in time (up to 20 s) with a test window of the same run, because adjacent windows share data at every block boundary. A robustness variant that drops them is reported on the Models page of the first run only.</li>}
            {dup && <li><strong>Duplicates:</strong> {String(dup.value)} test windows are identical to a training window (raw features), all with the same label{dupPre ? `; ${String(dupPre.value)} after preprocessing` : ''}.</li>}
            {sfBin && sfLev && <li><strong>No single feature does the job alone:</strong> the best one-feature model reaches balanced accuracy {fmt(Number(sfBin.value))} (binary) and {fmt(Number(sfLev.value))} (five-class), below the 0.90 alarm level.</li>}
            {forb && <li><strong>No label, scenario, run, split or id columns are model inputs</strong> ({String(forb.value)} offending column names). The system flag is an input on purpose, so a model can tell SN from TT.</li>}
          </ul>
        </div>
        <ResponsiveTable
          caption="All leakage checks except the per-feature scan (flagged = the check found something)"
          rows={d.leakage.map((r, i) => ({ ...r, id: String(i) }))} rowKey={(r) => r.id}
          columns={[
            { key: 'c', label: 'Check', nameCell: true, render: (r) => r.check.replace(/_/g, ' ') },
            { key: 's', label: 'Scope', render: (r) => r.scope },
            { key: 'i', label: 'Item', render: (r) => <span className="small">{r.item}</span> },
            { key: 'v', label: 'Value', numeric: true, render: (r) => String(r.value).slice(0, 60) },
            { key: 'f', label: 'Flagged', render: (r) => (r.flag ? <span className="badge solid">yes</span> : <span className="badge">no</span>) },
          ]}
        />
      </section>

      <section aria-labelledby="rely-h">
        <h2 id="rely-h">What the tree models rely on</h2>
        <div className="grid cols-2">
          {(['label_binary', 'label_level'] as Label[]).map((l) => (
            <div className="card" key={l}>
              <h3 style={{ marginTop: 0, fontSize: '1.05rem' }}>{l === 'label_binary' ? 'Binary' : 'Five-class'}: {ex.global_top[l][0]?.model}</h3>
              <p className="small">Share of total mean |SHAP| by group: {ex.group_share[l].map((g) => `${g.group} ${pct(g.share_of_total_mean_abs_shap, 1)}`).join(', ')}.</p>
              <ol className="small">{ex.global_top[l].slice(0, 5).map((t) => <li key={t.feature}><code>{t.feature}</code> ({t.group}, {fmt(t.mean_abs_shap)})</li>)}</ol>
            </div>
          ))}
        </div>
        <p className="muted">Host-level gauges dominate. Because each class comes from a few runs, such gauges can identify <em>which run</em> a window came from. See <Link to="/limits">Limits</Link>.</p>
      </section>
    </>
  )
}
