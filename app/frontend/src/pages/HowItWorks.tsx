import { Link } from 'react-router-dom'
import { StepIcon } from '../components/Icons'
import { Detailed } from '../mode'
import { usePageTitle } from '../usePageTitle'

const STEPS: [1 | 2 | 3 | 4, string, string][] = [
  [1, 'It watches', 'A small helper sits next to the website and passes on what happens: which pages were opened, how long they took, which ones failed.'],
  [2, 'It learns what normal looks like', 'For the first two minutes it just watches. That is the “normal” it compares everything else with.'],
  [3, 'It checks every 10 seconds', 'Every 10 seconds it looks at the last 30 seconds and asks a trained program: does this look like a website in trouble?'],
  [4, 'It says why', 'If two checks out of the last three say yes, it raises an alert and says which numbers went wrong, like “response time went up 4 times”.'],
]

export default function HowItWorks() {
  usePageTitle('How it works')
  return (
    <>
      <h1>How it works</h1>
      <p className="lede">FailSight is an alarm for websites. It notices when a site starts to go wrong and tells you in plain words what changed.</p>
      <ol className="steps" aria-label="How it works in four steps">
        {STEPS.map(([n, h, p]) => <li key={n}><StepIcon n={n} /><div><h2>{n}. {h}</h2><p>{p}</p></div></li>)}
      </ol>

      <h2>What you see in the live monitor</h2>
      <p>The <Link to="/">Live monitor</Link> watches a fictional ticket shop that runs on this computer with pretend visitors. On a separate presenter page you break the shop on purpose. FailSight is <strong>not told</strong> what you broke. It only sees the numbers coming from the shop.</p>

      <h2>Newer, more complex models</h2>
      <p>Besides the models that run live, we tried ten &ldquo;hybrid&rdquo; neural networks made of two learning parts, five on the benchmark data and five on a slice of a second data set. They are scored on the <Link to="/examiners/hybrids">Hybrid models</Link> page; on this small data they did not beat the best of the simpler models, and the page shows the numbers.</p>

      <h2>What FailSight does not do</h2>
      <ul>
        <li>It says that something is wrong <strong>right now</strong>. It does not predict the future.</li>
        <li>It has only been proven on one small shop. It did not work on a different kind of system (see <Link to="/proofs">Proof</Link>).</li>
        <li>It guesses what changed. That is a good clue, not a proven cause.</li>
      </ul>

      <h2>Plugging in a real website</h2>
      <p>The helper that watches a site can be added to a Python or Node website, or read a web server’s log file. The steps are on the <Link to="/examiners/connect">Connect page</Link>. A new site needs its own two minutes of learning and its own check, and FailSight marks it “not yet proven”.</p>

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
