import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Detailed, Simple } from '../mode'
import { TESTBED_LIMITS } from '../testbed'
import type { TbState } from '../testbed'
import { usePageTitle } from '../usePageTitle'

const PY = `# copy the failsight_agent folder next to your app (or: pip install ./agent); pip install psutil for process metrics
from failsight_agent import FailSight

fs = FailSight.from_config(dict(app_id="my-app", monitor_url="http://127.0.0.1:8000"))   # add api_key="..." if the monitor needs one
fs.instrument(app)          # app = your FastAPI / Starlette application
# with a database pool whose acquire() returns a context manager: fs.instrument(app, pool=my_pool)`

const NODE = `// copy agent/node/failsight-express.js (no dependencies, Node 18+)
const failsight = require('./failsight-express');
app.use(failsight({ monitorUrl: 'http://127.0.0.1:8000', appId: 'my-app', apiKey: process.env.FAILSIGHT_KEY }));`

const TAIL = `python -m failsight_agent.tail --file /var/log/nginx/access.log --monitor http://127.0.0.1:8000 --app-id my-app
# optional: --api-key KEY   --from-start (also send the lines already in the file)
# nginx: add '$request_time' at the end of the log_format to get request latencies`

function Snippet({ title, code, id }: { title: string; code: string; id: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); window.setTimeout(() => setCopied(false), 2000) } catch { setCopied(false) }
  }
  return (
    <section className="card stack" aria-labelledby={id} style={{ marginBottom: 'var(--space-4)' }}>
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 id={id} style={{ margin: 0 }}>{title}</h3>
        <button type="button" className="btn secondary" onClick={copy} aria-label={`Copy the ${title} snippet`}>Copy</button>
      </div>
      <pre className="console-pre" tabIndex={0} aria-label={`${title} snippet`} style={{ maxHeight: '16rem' }}>{code}</pre>
      <span className="small muted" role="status" aria-live="polite">{copied ? 'Copied to the clipboard.' : ''}</span>
    </section>
  )
}

function useState3s() {
  const [st, setSt] = useState<TbState | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const alive = useRef(true)
  const load = useCallback(async () => {
    try {
      const r = await fetch('/api/testbed/state')
      const j = await r.json()
      if (!r.ok) { setErr(j.error ?? `HTTP ${r.status}`); setSt(null) } else { setErr(null); setSt(j) }
    } catch (e: any) { setErr(String(e?.message ?? e)) }
    if (alive.current) timer.current = window.setTimeout(load, 3000)
  }, [])
  useEffect(() => { alive.current = true; load(); return () => { alive.current = false; if (timer.current) window.clearTimeout(timer.current) } }, [load])
  return { st, err, reload: load }
}

export default function ConnectApp() {
  usePageTitle('Connect your app')
  const { st, err, reload } = useState3s()
  const [appId, setAppId] = useState('my-app')
  const [hook, setHook] = useState('')
  const [hookMsg, setHookMsg] = useState('')
  const apps = st ? Object.keys(st.apps) : []
  const a = st?.apps[appId]
  const c = a?.counters
  const ago = a?.last_event_at ? Math.max(0, Math.round(Date.now() / 1000 - a.last_event_at)) : null
  const state = !a ? 'not connected' : a.learning.done ? 'live' : 'learning'

  const saveHook = async (clear: boolean) => {
    const r = await fetch('/api/testbed/webhook', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: clear ? '' : hook }) })
    const j = await r.json().catch(() => ({}))
    setHookMsg(r.ok ? (j.webhook_configured ? 'Webhook saved.' : 'Webhook removed.') : (j.error ?? `HTTP ${r.status}`))
    reload()
  }

  return (
    <>
      <h1>Connect your app</h1>
      <p className="lede">
        <Simple>Add a few lines to your app so it reports how it is doing. FailSight watches it for 2.5 minutes to learn what normal looks like, then tells you when it starts to look like a failure.</Simple>
        <Detailed>An agent sends request, log and process-metric events to the monitor (<code>POST /ingest/api|logs|metrics</code>, batched every second). After a 150 s learning period the monitor builds 30 s windows every 10 s, scores them with the frozen detector and raises an alert when 2 of the last 3 windows are flagged.</Detailed>
      </p>
      <div className="callout" role="note">
        <span className="label">Read this first</span>
        <p><strong>The detector was validated only on our own test shop.</strong> Alerts for any other app are marked <em>unvalidated</em>. Another app needs its own training data, or at least the learning period with the per-app detector, and you should break something on purpose in a test setting before you trust it.</p>
        <ul>{TESTBED_LIMITS.map((t) => <li key={t}>{t}</li>)}</ul>
      </div>

      <h2>1. Is your app connected?</h2>
      <div className="field" style={{ maxWidth: '22rem' }}>
        <label htmlFor="appid">App id (the name you gave the agent)</label>
        <input id="appid" list="known-apps" value={appId} onChange={(e) => setAppId(e.target.value.trim())} />
        <datalist id="known-apps">{apps.map((x) => <option key={x} value={x} />)}</datalist>
      </div>
      {err && <div className="state" role="alert"><h3>The monitor is not reachable</h3><p>{err}</p></div>}
      <section className="card stack" aria-labelledby="conn-h" style={{ marginTop: 'var(--space-3)' }}>
        <h3 id="conn-h" style={{ margin: 0 }}>Connection status</h3>
        <p role="status" aria-live="polite" style={{ fontSize: '1.2rem', margin: 0 }}>
          {state === 'not connected' && <strong>Not connected yet: no events received for “{appId}”.</strong>}
          {state === 'learning' && <strong>Learning normal behaviour: {Math.round((a?.learning.progress ?? 0) * 100)}% done</strong>}
          {state === 'live' && <strong>Live: scoring every 10 seconds{a?.idle ? ' (but no data for a while)' : ''}.</strong>}
        </p>
        {state === 'learning' && <progress value={Math.round((a?.learning.progress ?? 0) * 100)} max={100} aria-label="Learning progress" style={{ width: '100%' }} />}
        <ul className="tiles" aria-label="Events received">
          <li><div className="n">{(c?.events_api ?? 0).toLocaleString()}</div><div className="c">Requests received</div></li>
          <li><div className="n">{(c?.events_logs ?? 0).toLocaleString()}</div><div className="c">Log lines received</div></li>
          <li><div className="n">{(c?.events_metrics ?? 0).toLocaleString()}</div><div className="c">Process readings received</div></li>
          <li className={ago !== null && ago > 90 ? 'bad' : ''}><div className="n">{ago === null ? '–' : `${ago} s`}</div><div className="c">Since the last event</div></li>
        </ul>
        {a?.last && <p className="small" style={{ margin: 0 }}>{a.last.validated ? 'This app is the validated testbed shop.' : 'This app is unvalidated: the detector was validated only on the testbed shop.'} Data seen in the last window: requests {a.last.sources.api ? 'yes' : 'no'}, logs {a.last.sources.logs ? 'yes' : 'no'}, process readings {a.last.sources.process_metrics ? 'yes' : 'no'}.</p>}
        <Detailed>
          <p className="small" style={{ margin: 0 }}>
            windows {c?.windows ?? 0}, flagged {c?.flagged ?? 0}, alerts {c?.alerts ?? 0}, rejected events {c?.rejected ?? 0}; model {st?.config.model} on {st?.config.feature_set} features, threshold {st?.config.threshold.toFixed(3)}, frozen {st?.config.frozen_at};
            API key required: {st?.integration?.api_key_required ? `yes (apps: ${st.integration.apps_with_keys.join(', ')})` : 'no (off by default)'}; webhook: {st?.integration?.webhook_configured ? `configured (${st.integration.webhook_host})` : 'not configured'}.
          </p>
        </Detailed>
      </section>

      <h2>2. Add the agent</h2>
      <Snippet id="snip-py" title="Python (ASGI middleware)" code={PY} />
      <Snippet id="snip-node" title="Node (Express middleware)" code={NODE} />
      <Snippet id="snip-tail" title="Any server (tail an nginx or app access log)" code={TAIL} />
      <p className="small muted">Full guide with what to expect, API keys and the Slack webhook: <code>docs/INTEGRATION.md</code>. Replace <code>my-app</code> and the monitor address, then type the same app id above.</p>

      <h2>3. Optional: get alerts in Slack</h2>
      <Simple><p>Paste a Slack (or Slack-compatible) incoming-webhook address. One message is sent for each alert.</p></Simple>
      <Detailed><p>A JSON body <code>{'{"text": "FailSight alert for …"}'}</code> is POSTed to this URL in the background, once per alert episode. Also settable with <code>FAILSIGHT_WEBHOOK_URL</code>.</p></Detailed>
      <div className="row">
        <div className="field" style={{ flex: '1 1 22rem' }}>
          <label htmlFor="hook">Webhook address</label>
          <input id="hook" type="url" value={hook} onChange={(e) => setHook(e.target.value)} placeholder="https://hooks.slack.com/services/…" />
        </div>
        <button type="button" className="btn" onClick={() => saveHook(false)} disabled={!hook}>Save</button>
        <button type="button" className="btn secondary" onClick={() => saveHook(true)}>Remove</button>
      </div>
      <p className="small muted" role="status" aria-live="polite">{hookMsg || (st?.integration?.webhook_configured ? `A webhook is configured (${st.integration.webhook_host}).` : 'No webhook is configured.')}</p>
      <p className="small">See also <Link to="/examiners/testbed">Testbed</Link> and <Link to="/examiners/limits">Limits</Link>.</p>
    </>
  )
}
