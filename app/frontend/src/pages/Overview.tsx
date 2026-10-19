import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { Async } from '../components/States'
import { fmt, int, pct } from '../format'
import { usePageTitle } from '../usePageTitle'
import type { DataResponse, Lo2Response, ModelsResponse, ResultsResponse } from '../types'

function Stat({ num, cap }: { num: string; cap: string }) {
  return (
    <div className="stat">
      <div className="num">{num}</div>
      <div className="cap">{cap}</div>
    </div>
  )
}

export default function Overview() {
  usePageTitle('Overview')
  const data = useApi<DataResponse>('/api/data')
  const results = useApi<ResultsResponse>('/api/results')
  const lo2 = useApi<Lo2Response>('/api/lo2')
  const models = useApi<ModelsResponse>('/api/models')

  return (
    <>
      <p className="eyebrow">Research prototype</p>
      <h1>Can a 30-second window of telemetry tell us an API is failing?</h1>
      <p className="lede">
        FailSight classifies windows of API, log and metric data from two microservice systems, SocialNetwork (SN) and TrainTicket (TT), as normal or failing, and shows which signals drove each verdict.
        It also reports, without softening, where the models fall short.
      </p>

      <section aria-labelledby="numbers" style={{ marginTop: 'var(--space-6)' }}>
        <h2 id="numbers" style={{ marginTop: 0 }}>The data in numbers</h2>
        <Async state={data} rows={3} label="Loading dataset numbers">
          {(d) => (
            <div className="grid cols-4">
              <Stat num={int(d.dataset.windows)} cap={`windows of ${d.dataset.window_seconds} s, step ${d.dataset.step_seconds} s (SN ${int(d.dataset.systems.SN)}, TT ${int(d.dataset.systems.TT)})`} />
              <Stat num={`${int(d.dataset.train)} / ${int(d.dataset.test)}`} cap={`train / untouched test windows (${pct(d.dataset.train_share, 2)} train)`} />
              <Stat num={int(d.dataset.runs)} cap={`scenario runs in ${int(d.dataset.blocks)} blocks of ${d.dataset.block_windows} windows`} />
              <Stat num={`${d.dataset.raw_features - d.dataset.dropped_constant} of ${d.dataset.raw_features}`} cap="raw window features the models use (the rest are constant in training and were dropped)" />
            </div>
          )}
        </Async>
      </section>

      <section aria-labelledby="read-first" style={{ marginTop: 'var(--space-6)' }}>
        <h2 id="read-first" style={{ marginTop: 0 }}>Read this before trusting a verdict</h2>
        <div className="callout">
          <span className="label">Honest summary</span>
          <Async state={results} rows={2} label="Loading headline results">
            {(r) => {
              const top = r.leaderboards.label_binary[0]
              const top5 = r.leaderboards.label_level[0]
              return (
                <ul>
                  <li>
                    On the untouched test windows the best binary model ({top.model}) has balanced accuracy <strong>{fmt(top.test_bal_acc)}</strong> but finds only <strong>{pct(top.test_normal_recall, 1)}</strong> of the Normal windows.
                    The best five-class model ({top5.model}) reaches balanced accuracy {fmt(top5.test_bal_acc)}.
                  </li>
                  <li>
                    <Async state={lo2} rows={1}>
                      {(l) => (
                        <>On a second, unrelated system (LO2), all {l.models.length} saved models score balanced accuracy <strong>{fmt(Math.max(...l.models.map((m) => m.balanced_accuracy)))}</strong>: only {l.feature_mapping.A_logs_only.used_features_built} of the {l.feature_mapping.A_logs_only.used_features_total} features can be built there, from {l.sample.windows} windows.</>
                      )}
                    </Async>
                  </li>
                </ul>
              )
            }}
          </Async>
          <p className="small" style={{ marginBottom: 0 }}><Link to="/limits">Read the full limits page</Link></p>
        </div>
      </section>

      <section aria-labelledby="pipeline">
        <h2 id="pipeline">How the results were produced</h2>
        <p className="muted">The real steps of the pipeline, in order. Test data is touched once, at step 8.</p>
        <Async state={data} rows={3} label="Loading pipeline">
          {(d) => (
            <ol className="pipeline" aria-label="Pipeline steps">
              <li>
                <h3>Telemetry to windows</h3>
                <p>API responses, logs and metrics of each scenario run are cut into overlapping windows.</p>
                <span className="fact">{int(d.dataset.windows)} windows x {d.dataset.raw_features} features</span>
              </li>
              <li>
                <h3>Block split 80:20</h3>
                <p>Blocks of {d.dataset.block_windows} consecutive windows are shuffled per system (seed 42); API windows stay on both sides.</p>
                <span className="fact">{int(d.dataset.train)} train, {int(d.dataset.test)} test</span>
              </li>
              <li>
                <h3>Leakage checks</h3>
                <p>Duplicates, single-feature power, forbidden columns, time overlap and run overlap are measured and reported.</p>
                <span className="fact"><Link to="/data">see findings</Link></span>
              </li>
              <li>
                <h3>Preprocessing</h3>
                <p>Fitted on training rows only, per system: log1p, median imputation with missing flags, clipping, scaling.</p>
                <span className="fact">{d.dataset.model_inputs} model inputs</span>
              </li>
              <li>
                <h3>Run-aware 10-fold CV</h3>
                <p>Folds group windows by block so validation never shares a block with training; SMOTE only inside training folds.</p>
                <span className="fact">{d.folds.n_folds} folds, groups = {d.folds.group_type}</span>
              </li>
              <li>
                <h3>Models</h3>
                <p>Five baselines, Optuna on the top three tree models, four deep networks, XGBoost heads, stacking and soft voting.</p>
                <span className="fact"><Async state={models} rows={1}>{(m) => <>{m.models.length / 2} models per label version</>}</Async></span>
              </li>
              <li>
                <h3>Thresholds and correction</h3>
                <p>Decision thresholds come from out-of-fold predictions only; models diagnosed as overfit are retrained with stronger regularisation.</p>
                <span className="fact"><Link to="/models">see before / after</Link></span>
              </li>
              <li>
                <h3>One test evaluation</h3>
                <p>Final models are fitted on the full training part and scored once on the test windows, with block-bootstrap intervals, SHAP and LIME.</p>
                <span className="fact">{int(d.dataset.test)} windows</span>
              </li>
              <li className="external">
                <h3>External check on LO2</h3>
                <p>The unchanged models are applied to windows built from a different system&apos;s logs.</p>
                <span className="fact"><Async state={lo2} rows={1}>{(l) => <>{l.sample.windows} windows</>}</Async></span>
              </li>
            </ol>
          )}
        </Async>
      </section>

      <section aria-labelledby="explore">
        <h2 id="explore">Explore</h2>
        <div className="grid cols-3">
          <Link to="/predict" className="link-card"><h3>Predict</h3><p>Try a real test window or upload a CSV, pick a model, see the verdict and what drove it.</p><span className="go">Open Predict</span></Link>
          <Link to="/models" className="link-card"><h3>Models</h3><p>Sortable leaderboard with 95% intervals, confusion matrices, curves, first run versus retrain.</p><span className="go">Open Models</span></Link>
          <Link to="/explain" className="link-card"><h3>Explain</h3><p>SHAP and LIME side by side for one window, with values on hover.</p><span className="go">Open Explain</span></Link>
          <Link to="/data" className="link-card"><h3>Data &amp; method</h3><p>Class counts before and after SMOTE, the split, run-aware folds and leakage findings.</p><span className="go">Open Data &amp; method</span></Link>
          <Link to="/limits" className="link-card"><h3>Limits</h3><p>The Normal-recall weakness and the LO2 external test, stated plainly.</p><span className="go">Open Limits</span></Link>
        </div>
      </section>
    </>
  )
}
