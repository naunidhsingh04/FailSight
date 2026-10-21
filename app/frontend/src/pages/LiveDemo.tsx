import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { StatusIcon } from '../components/Icons'
import { Gauge, RiskTimeline } from '../components/RiskCharts'
import Term from '../components/Term'
import { Detailed, Simple } from '../mode'
import { clock, percent } from '../monitor'
import type { HistPoint, Status } from '../monitor'
import { FAULTS, TESTBED_LIMITS, useFaults, useTestbed } from '../testbed'
import type { TbWindow } from '../testbed'

const WORD: Record<string, string> = { learning: 'Learning', healthy: 'Healthy', watch: 'Watch', failing: 'Failing', waiting: 'Waiting for data' }
const SENTENCE: Record<string, string> = {
  waiting: 'No data from the demo shop yet. Start it with start_testbed.bat (or python -m testbed.launcher).',
  learning: 'The monitor is watching the shop to learn what normal looks like. No alerts are raised yet.',
  healthy: 'Everything looks normal.',
  watch: 'One recent window looked unusual. Keeping an eye on it.',
  failing: 'Two of the last three windows look like a failure: an alert was raised.',
}
const asStatus = (s: string): Status => (s === 'healthy' || s === 'watch' || s === 'failing' ? s : 'idle')

export function TestbedLimits() {
  return (
    <div className="callout" role="note">
      <span className="label">What this demo does and does not show</span>
      <ul>{TESTBED_LIMITS.map((t) => <li key={t}>{t}</li>)}</ul>
      <p className="small">Details and numbers: <Link to="/testbed">Testbed</Link> and <Link to="/limits">Limits</Link>.</p>
    </div>
  )
}

export default function LiveDemo() {
  const { state, windows, connected, error, app } = useTestbed('shop')
  const faults = useFaults()
  const [severity, setSeverity] = useState<'mild' | 'medium' | 'severe'>('medium')
  const [selected, setSelected] = useState<TbWindow | null>(null)
  const last = app?.last ?? windows[windows.length - 1] ?? null
  const status = last ? last.status : 'waiting'
  const hist: HistPoint[] = useMemo(() => {
    const src = windows.length ? windows : (app?.history ?? []).map((h) => ({ window_index: h.window_index, window_start: h.window_end - 30, window_end: h.window_end, risk: h.risk, alert: h.alert, learning: h.learning }))
    return src.map((w: any) => ({ window_index: w.window_index, window_start: w.window_end - 30, risk: w.risk, flagged: w.alert, truth: null, warmup: w.learning }))
  }, [windows, app])
  const alerts = useMemo(() => {
    const live = windows.filter((w) => w.new_alert).reverse()
    return live.length ? live : [...(app?.alerts ?? [])].reverse()
  }, [windows, app])
  const progress = last ? last.learning_progress : 0
  const threshold = state?.config.threshold ?? null
  const c = app?.counters
  const now = Date.now() / 1000

  return (
    <>
      <div className="banner" role="note">
        <strong>Live demo of our own test shop</strong>
        <p>A small ticket shop runs on this laptop, reports to the monitor only through the FailSight agent, and gets real faults injected on purpose. It is <em>not</em> a production system and not recorded data.</p>
      </div>
      <TestbedLimits />

      {error && <div className="state" role="alert"><h3>The live monitor is not available</h3><p>{error}</p></div>}

      <h2 style={{ marginTop: 'var(--space-6)' }}>1. Is the shop healthy?</h2>
      <section className={`sys-card ${asStatus(status)}`} aria-labelledby="shop-h" style={{ maxWidth: '34rem' }}>
        <h3 id="shop-h">Demo shop (5 pages: search, book, pay, cancel, health)</h3>
        <div className={`status-word ${asStatus(status)}`} role="status" aria-live="polite" aria-atomic="true">
          <StatusIcon status={asStatus(status)} />
          <span><span className="visually-hidden">Demo shop status: </span>{WORD[status] ?? status}</span>
        </div>
        <p className="small muted" style={{ marginTop: 0 }}>{SENTENCE[status] ?? ''}</p>
        {status === 'learning' && (
          <div>
            <label htmlFor="learn" style={{ fontWeight: 650 }}>Learning normal behaviour: {Math.round(progress * 100)}% done</label>
            <progress id="learn" value={Math.round(progress * 100)} max={100} style={{ width: '100%' }} />
          </div>
        )}
        {status !== 'learning' && <div className="gauge-wrap"><Gauge risk={last ? last.risk : null} threshold={threshold} status={asStatus(status)} label="Demo shop" /></div>}
        {last && <p className="small muted" style={{ marginBottom: 0 }}>Last check ended at {clock(last.window_end)} UTC{app?.idle ? ' (no data for a while: is the shop running?)' : ''}</p>}
        <Detailed>
          <p className="small" style={{ marginBottom: 0 }}>
            risk {last ? last.risk.toFixed(3) : '–'}; alert line {threshold?.toFixed(3) ?? '–'}; model {state?.config.model} on {state?.config.feature_set} features, frozen {state?.config.frozen_at}; rule: {state?.config.alert_rule};
            second opinion (robust z-score, per app, never alerts): {last?.robust_z_score != null ? last.robust_z_score.toFixed(1) : '–'} {last?.robust_z_flag ? '(would flag)' : ''}
          </p>
        </Detailed>
      </section>

      <section className="card" aria-labelledby="tl-live" style={{ marginTop: 'var(--space-5)' }}>
        <h2 id="tl-live" style={{ marginTop: 0, fontSize: '1.3rem' }}>Failure risk over time</h2>
        <div className="tl-wrap"><RiskTimeline points={hist} threshold={threshold} ariaLabel={`Failure risk of the demo shop over the last ${hist.length} windows; ${hist.filter((p) => p.flagged).length} raised an alert`} /></div>
        <p className="takeaway">{hist.length === 0 ? 'The line appears here once data arrives.' : `${hist.filter((p) => p.flagged && !p.warmup).length} of the last ${hist.length} windows are part of an alert. The grey dots are the learning period.`}</p>
      </section>

      <h2 style={{ marginTop: 'var(--space-6)' }}>2. Break something on purpose</h2>
      <p>Every switch changes what the running shop really does (it sleeps, burns processor time, fails, locks its database, leaks memory). The monitor is not told which one you picked.</p>
      <div className="controls">
        <div className="field">
          <span className="label" id="sev-l">How bad</span>
          <div className="segmented" role="group" aria-labelledby="sev-l">
            {(['mild', 'medium', 'severe'] as const).map((s) => <button key={s} type="button" aria-pressed={severity === s} onClick={() => setSeverity(s)}>{s[0].toUpperCase() + s.slice(1)}</button>)}
          </div>
        </div>
        <button type="button" className="btn danger" onClick={() => faults.clear()} disabled={faults.active.length === 0}>Stop all faults</button>
      </div>
      <ul className="pill-group" style={{ listStyle: 'none', padding: 0, maxWidth: 'none', marginTop: 'var(--space-3)' }}>
        {FAULTS.map((f) => (
          <li key={f.id}><button type="button" className="pill" title={f.hint} onClick={() => faults.inject(f.id, severity)} disabled={!!faults.err}>{f.label}</button></li>
        ))}
      </ul>
      <p className="small muted" role="status" aria-live="polite">
        {faults.err ? `The demo shop is not reachable: ${faults.err}` : faults.active.length === 0 ? 'No fault is active.' : 'Active now: ' + faults.active.map((f) => `${FAULTS.find((x) => x.id === f.type)?.label ?? f.type} (${Math.max(0, Math.round(f.until - now))} s left${f.failed ? ', the shop has failed' : ''})`).join('; ')}
      </p>

      <h2 style={{ marginTop: 'var(--space-6)' }}>3. How is it doing?</h2>
      <ul className="tiles" aria-label="Live counters">
        <li><div className="n">{c?.windows ?? 0}</div><div className="c">Windows checked</div></li>
        <li><div className="n">{c?.flagged ?? 0}</div><div className="c">Windows flagged</div></li>
        <li className="bad"><div className="n">{c?.alerts ?? 0}</div><div className="c"><Term k="alert">Alerts raised</Term></div></li>
        <li><div className="n">{(c?.events_api ?? 0).toLocaleString()}</div><div className="c">Requests seen</div></li>
        <li><div className="n">{connected ? 'on' : 'off'}</div><div className="c">Live updates</div></li>
      </ul>
      <p className="takeaway">Here the truth is not known to the monitor: you can see what you injected above. The scripted validation that compares alerts with a recorded fault schedule is on the <Link to="/testbed">Testbed page</Link>.</p>

      <h2 style={{ marginTop: 'var(--space-6)' }}>4. Alerts and why</h2>
      {alerts.length === 0 ? (
        <div className="state"><h3>No alerts yet</h3><p>An alert appears when two of the last three windows look like a failure. Inject a fault above to see one.</p></div>
      ) : (
        <div className="console-grid with-panel" style={{ gridTemplateColumns: 'minmax(0,1fr)' }}>
          <ul className="alerts" aria-label="Alert feed, newest first">
            {alerts.map((a) => (
              <li key={`${a.window_start}`}>
                <button type="button" className="alert-btn" aria-current={selected?.window_start === a.window_start ? 'true' : undefined} onClick={() => setSelected(a)}>
                  <span className="t">Demo shop &middot; {clock(a.window_end)} UTC &middot; risk {percent(a.risk)}</span>
                  <span className="s">{a.summary ?? 'Alert raised.'}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {selected && (
        <aside className="card" aria-labelledby="why-live" style={{ marginTop: 'var(--space-4)' }}>
          <button type="button" className="btn secondary sheet-close" onClick={() => setSelected(null)} aria-label="Close this explanation">Close</button>
          <h3 id="why-live" style={{ marginTop: 0 }}>Why this alert?</h3>
          <p><strong>{selected.summary}</strong></p>
          <Simple><p className="small muted">These are the readings the detector leaned on, compared with what the shop looked like while it was learning. They show what it reacted to, not a proven cause.</p></Simple>
          <Detailed>
            <table className="rt"><caption className="visually-hidden">Top contributions (tree SHAP) of the frozen model</caption>
              <thead><tr><th scope="col">Reading</th><th scope="col">Value</th><th scope="col">Normal</th><th scope="col">Push</th></tr></thead>
              <tbody>{selected.reasons.map((r) => <tr key={r.feature}><td><code>{r.feature}</code></td><td>{r.value.toPrecision(4)}</td><td>{r.normal == null ? '–' : r.normal.toPrecision(4)}</td><td>+{r.shap.toFixed(3)}</td></tr>)}</tbody></table>
          </Detailed>
        </aside>
      )}
    </>
  )
}
