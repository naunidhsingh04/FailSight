import { Link } from 'react-router-dom'
import Term from '../components/Term'
import { HowStrip } from './Console'
import { Detailed } from '../mode'
import { usePageTitle } from '../usePageTitle'

function Flow() {
  const boxes: [string, string][] = [['Collector', 'gathers requests, logs and measurements'], ['Ingestion API', 'checks and stores each event'], ['Window builder', 'every 10 s: sums up the last 30 s'], ['Model', 'estimates failure risk'], ['Alerts', 'shows status and the reasons']]
  return (
    <ol className="steps" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(10.5rem, 1fr))' }} aria-label="Architecture: Collector, Ingestion API, Window builder, Model, Alerts">
      {boxes.map(([h, p], i) => (
        <li key={h} style={{ flexDirection: 'column', gap: 'var(--space-1)', position: 'relative' }}>
          <h3>{h}</h3><p>{p}</p>
          {i < boxes.length - 1 && <span aria-hidden="true" style={{ position: 'absolute', right: '-0.9rem', top: '50%', fontSize: '1.4rem', zIndex: 1 }} className="arrow">&rarr;</span>}
        </li>
      ))}
    </ol>
  )
}

export default function HowItWorks() {
  usePageTitle('How it works')
  return (
    <>
      <h1>How it works</h1>
      <p className="lede">FailSight looks at the last 30 seconds of an application’s activity, every 10 seconds, and says how likely it is that something is going wrong <em>right now</em>.</p>
      <h2>In four steps</h2>
      <HowStrip />

      <h2>The pipeline</h2>
      <Flow />
      <p className="muted" style={{ marginTop: 'var(--space-4)' }}>Everything on the Console page goes through these five stages. The browser receives each new result the moment it is ready (<Term k="sse">live updates</Term>).</p>

      <h2>What is real and what is a replay</h2>
      <div className="callout">
        <span className="label">Be clear about the source</span>
        <p><strong>Replay of recorded AnoMod data.</strong> The demonstration sends records from the <em>test part</em> of a recorded dataset (<Term k="anomod" />) through the real ingestion, window-building, model and explanation code. A <Term k="replay" /> script plays the part of the collector. There is no live production system behind this page.</p>
        <p>Because the experiment was recorded, the correct answer for every window is <Term k="recording">known from the recording</Term>, which is how the page can count correct alerts, false alarms and missed failures. A real deployment would not have those answers.</p>
      </div>

      <h2>What the system does and does not do</h2>
      <ul>
        <li>It <strong>detects</strong> whether the last 30 seconds look like a failure. It does <strong>not</strong> predict when a failure will happen in the future.</li>
        <li>It was trained on two sample applications. It did not work on a third, different system (see <Link to="/limits">Limits</Link>).</li>
        <li>Every number shown comes from files, saved models or recomputation (see <Link to="/proofs">Proofs</Link>).</li>
      </ul>

      <h2>How a real deployment connects</h2>
      <p className="muted">The ingestion API accepts three kinds of events as JSON. Common tools can feed it:</p>
      <ul>
        <li><strong>Requests and responses</strong> (status, response time, size) from an <Term k="otel">OpenTelemetry</Term> collector or from web-server access logs, posted to <code>/ingest/api</code>.</li>
        <li><strong>Log lines</strong> (time, level, service) forwarded by <Term k="fluentbit">Fluent Bit</Term> to <code>/ingest/logs</code>.</li>
        <li><strong>Measurements</strong> (CPU, memory, disk and so on) read from <Term k="prometheus">Prometheus</Term> and posted to <code>/ingest/metrics</code>.</li>
      </ul>
      <p className="muted">Nothing here is connected to such tools yet; this describes how it would be done. The models were trained on the signals of the two recorded applications, so a new application needs its own recordings and training first.</p>

      <Detailed>
        <h2>Technical details</h2>
        <div className="table-wrap">
          <table className="rt">
            <caption className="visually-hidden">Pipeline parameters</caption>
            <thead><tr><th scope="col">Item</th><th scope="col">Value</th></tr></thead>
            <tbody>
              <tr><td data-label="Item" className="name-cell">Window</td><td data-label="Value">30 s long, a new one every 10 s (event time); built when the stream has advanced 1 s past its end</td></tr>
              <tr><td data-label="Item" className="name-cell">Features</td><td data-label="Value">155 raw: 8 API, 3 log, 144 metric; counters (*_total) are turned into per-second rates; 128 model inputs after preprocessing</td></tr>
              <tr><td data-label="Item" className="name-cell">Shared code</td><td data-label="Value"><code>scripts/window_features.py</code> (verified against the training tables by tests), <code>scripts/preprocess.py</code></td></tr>
              <tr><td data-label="Item" className="name-cell">Warm-up</td><td data-label="Value">first 2 windows of a session: counter rates incomplete, no alerts, not scored</td></tr>
              <tr><td data-label="Item" className="name-cell">Decision</td><td data-label="Value">the model’s saved rule (<code>thresholds.json</code>), tuned on out-of-fold predictions</td></tr>
              <tr><td data-label="Item" className="name-cell">Explanation</td><td data-label="Value">SHAP (top 10 features) turned into sentences with the measured value and the median of normal training windows</td></tr>
              <tr><td data-label="Item" className="name-cell">Endpoints</td><td data-label="Value"><code>POST /ingest/api|logs|metrics|reset|flush</code>, <code>GET /api/stream</code> (SSE), <code>GET /api/monitor/state</code></td></tr>
            </tbody>
          </table>
        </div>
      </Detailed>
    </>
  )
}
