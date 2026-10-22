import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ResponseChart, StatusBar, StatusSentence } from '../components/DemoViz'
import type { Pt } from '../components/DemoViz'
import Term from '../components/Term'
import { Detailed } from '../mode'
import { clock12, levelOf, plainReason, sentence } from '../plain'
import { useFaults, useTestbed } from '../testbed'
import { usePageTitle } from '../usePageTitle'

/** A problem in the shop: found by watching the shop itself (a fault switched on, or the shop stopping answering), so it works no matter where it was caused from. */
interface Problem { at: number; seconds: number; stopped: boolean; alreadyAlerting: boolean }
const KEY = 'failsight.problem'
const readProblem = (): Problem | null => { try { const v = JSON.parse(window.sessionStorage.getItem(KEY) ?? 'null'); return v && typeof v.at === 'number' && Date.now() / 1000 - v.at < 1800 ? v : null } catch { return null } }

export function MonitorLimits() {
  return (
    <aside className="callout limits-box" aria-labelledby="lim-h">
      <h2 id="lim-h" className="label">What this monitor does not prove</h2>
      <ul>
        <li>It was tested on <strong>one app</strong>: our own fictional test shop.</li>
        <li>The test was <strong>small</strong>: 12 test runs with 15 problems, so the scores are rough.</li>
        <li><strong>Other apps are unproven.</strong> A new app needs its own check.</li>
      </ul>
      <p className="small">All the limits, in full: <Link to="/examiners/limits">Limits</Link>.</p>
    </aside>
  )
}

const shopAddress = (path = '') => `${window.location.protocol}//${window.location.hostname}:5000${path}`

export default function LiveMonitor() {
  usePageTitle('Live monitor')
  const { state, windows, connected, error, app } = useTestbed('shop')
  const faults = useFaults()
  const [problem, setProblemState] = useState<Problem | null>(readProblem)
  const [starting, setStarting] = useState(false)
  const [startErr, setStartErr] = useState<string | null>(null)
  const [stopNote, setStopNote] = useState<string | null>(null)
  const wasUp = useRef(false)
  const stopping = useRef(false)
  const seenFault = useRef<string | null>(null)
  const setProblem = (p: Problem | null) => { setProblemState(p); try { if (p) window.sessionStorage.setItem(KEY, JSON.stringify(p)); else window.sessionStorage.removeItem(KEY) } catch { /* optional */ } }

  const shopUp = faults.ready && !faults.err
  const checking = !faults.ready
  const last = app?.last ?? windows[windows.length - 1] ?? null
  const pct = Math.round(((last?.learning_progress ?? app?.learning.progress) ?? 0) * 100)
  const learned = shopUp && !!app?.learning.done
  const status = !shopUp ? null : (last?.status ?? 'learning')
  const downNow = !!problem && problem.stopped && !shopUp && Date.now() / 1000 < problem.at + 150     // the shop stopped and restarts by itself
  const level = downNow ? 'problem' : levelOf(status, shopUp)

  // notice a problem from the shop's own state
  useEffect(() => {
    if (!faults.ready) return
    const f = faults.active[0]
    if (f && seenFault.current !== f.id) {
      seenFault.current = f.id
      setProblem({ at: f.start, seconds: Math.max(30, Math.round(f.until - f.start)), stopped: false, alreadyAlerting: !!last?.alert })
    }
    if (shopUp) wasUp.current = true
    else if (wasUp.current && !stopping.current) {            // it answered before and now it does not: it stopped
      wasUp.current = false
      setProblem({ at: Date.now() / 1000 - 2, seconds: 60, stopped: true, alreadyAlerting: !!last?.alert })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [faults.active, faults.ready, shopUp])

  const notice = useMemo(() => {
    if (!problem || problem.alreadyAlerting) return null        // an earlier alarm was still on: the delay cannot be measured
    return windows.find((w) => w.alert && !w.learning && (w.emitted_at ?? w.window_end) >= problem.at) ?? null
  }, [windows, problem])
  const noticedAt = notice ? (notice.emitted_at ?? notice.window_end) : null
  const series: Pt[] = useMemo(() => windows.filter((w) => !w.learning && w.features?.api_latency_p95_ms != null).map((w) => ({ t: w.window_end, ms: w.features.api_latency_p95_ms as number })), [windows])
  const now = Date.now() / 1000
  const faultOn = faults.active.length > 0
  const missed = !!problem && !problem.alreadyAlerting && !notice && !faultOn && shopUp && now > problem.at + problem.seconds + 45
  const why = plainReason((notice ?? (level === 'watch' || level === 'problem' ? last : null))?.reasons?.[0])

  async function start() {
    setStarting(true); setStartErr(null); stopping.current = false
    try {
      const r = await fetch('/api/testbed/start', { method: 'POST' })
      const j = await r.json().catch(() => ({}))
      if (!r.ok) { setStartErr(j.error ?? `The shop could not be started (HTTP ${r.status}).`); setStarting(false) }
      else { faults.refresh(); window.setTimeout(() => setStarting(false), 90000) }
    } catch (e: any) { setStartErr(String(e?.message ?? e)); setStarting(false) }
  }
  async function stop() {
    stopping.current = true; wasUp.current = false
    const r = await fetch('/api/testbed/stop', { method: 'POST' })
    const j = await r.json().catch(() => ({}))
    setStopNote(j.stopped ? 'The shop was stopped.' : j.note ?? 'Nothing to stop.')
    setProblem(null)
    faults.refresh()
  }

  return (
    <>
      <h1>Live monitor</h1>
      <p className="lede">FailSight watches the shop and tells you in plain words when something goes wrong.</p>

      <div className={`shopstrip ${shopUp ? 'up' : 'down'}`} role="status" aria-live="polite">
        <span><strong>{checking ? 'Checking the shop…' : downNow ? 'The shop has stopped. It restarts by itself.' : shopUp ? 'The shop is running.' : 'The shop is not running.'}</strong></span>
        <a className="btn secondary small-btn" href={shopAddress('/')} target="_blank" rel="noopener noreferrer">Open the shop</a>
      </div>

      <ol className="demo-steps">
        <li className="step" aria-labelledby="s1">
          <h2 id="s1"><span className="num" aria-hidden="true">1</span> Start the shop</h2>
          {error && <p role="alert" className="bad-text">FailSight itself is not answering: {error}. Reload the page, or restart it with start_testbed.bat.</p>}
          {checking ? <p>Checking whether the shop is running…</p>
            : downNow ? <p><strong>The shop has stopped.</strong> It starts again by itself in about 30 seconds. Nothing to do but wait.</p>
              : !shopUp ? (
                <>
                  <button type="button" className="btn big" onClick={start} disabled={starting}>{starting ? 'Starting… about a minute' : 'Start the shop'}</button>
                  <p className="hint">{starting ? 'The shop is starting. This page updates by itself.' : 'This starts the fictional ticket shop on this computer, with pretend visitors.'}</p>
                  {startErr && <p role="alert" className="bad-text">{startErr}</p>}
                  <p className="small muted">Does nothing happen? Double-click <code>start_testbed.bat</code> in the project folder instead.</p>
                </>
              ) : !learned ? (
                <>
                  <p className="ok-text"><strong>✓ The shop is running.</strong></p>
                  <label htmlFor="learn" className="progress-label">FailSight is <Term k="learning">learning what normal looks like</Term>: {pct}% done</label>
                  <progress id="learn" value={pct} max={100} />
                  <p className="hint">This takes about 2 minutes. Then step 2 is ready. Nothing to do but wait.</p>
                </>
              ) : (
                <>
                  <p className="ok-text"><strong>✓ The shop is running and FailSight knows what normal looks like.</strong></p>
                  <p className="hint">Go on to step 2.</p>
                  <button type="button" className="btn secondary small-btn" onClick={stop}>Stop the shop</button>
                  {stopNote && <p className="small muted" role="status">{stopNote}</p>}
                </>
              )}
        </li>

        <li className={`step ${learned ? '' : 'locked'}`} aria-labelledby="s2">
          <h2 id="s2"><span className="num" aria-hidden="true">2</span> Break something</h2>
          <p className="hint">{learned ? 'Breaking the shop is done on the presenter page, which only works on this computer. Come back here to watch.' : shopUp ? 'Wait for step 1 to finish first.' : 'Start the shop in step 1 first.'}</p>
          <p><a className="btn big" href={shopAddress('/ops')} target="_blank" rel="noopener noreferrer" aria-disabled={!learned}>Open the presenter controls</a></p>
        </li>

        <li className="step" aria-labelledby="s3">
          <h2 id="s3"><span className="num" aria-hidden="true">3</span> What does FailSight say?</h2>
          <StatusSentence level={level} text={downNow ? 'Something is wrong: the whole shop is down' : sentence(level, notice ?? last, pct)} />
          <StatusBar level={level} />
          {level === 'idle' && <p className="hint">Press <strong>Start the shop</strong> in step 1. The answer shows up here.</p>}
          {level === 'learning' && <p className="hint">Nothing can be judged yet. In about 2 minutes this turns into a real answer.</p>}
          {(level === 'ok' || level === 'watch' || level === 'problem') && !problem && <p className="hint">{level === 'ok' ? 'Now break something on the presenter page and watch this change.' : 'Nothing was broken while this page was open. Either it was broken earlier, or the shop had a short bad moment.'}</p>}
          {problem && (
            <ul className="timeline" aria-label="What happened, in order">
              <li>{problem.stopped ? 'The shop stopped answering' : 'A problem started'} at <strong>{clock12(problem.at)}</strong>.</li>
              {notice && noticedAt ? <li className="found">FailSight noticed at <strong>{clock12(noticedAt)}</strong> ({Math.max(0, Math.round(noticedAt - problem.at))} seconds later).</li>
                : problem.alreadyAlerting ? <li>FailSight was still on alert from the earlier problem, so we cannot time it. Wait for “Everything looks normal”, then try again.</li>
                  : missed ? <li>FailSight did not raise an alert this time. This can happen: see the limits below.</li>
                    : <li>FailSight has not noticed yet. It looks every 10 seconds and needs about 30 seconds to be sure.</li>}
            </ul>
          )}
          {(notice || level === 'watch' || level === 'problem') && !downNow && <p className="reason"><strong>Why:</strong> {why.reason} <span className="muted">FailSight is not told what went wrong. It only sees numbers, so this is what it reacted to, not a proven cause.</span></p>}

          <h3 className="chart-title">How long pages take to open</h3>
          <ResponseChart points={series} brokeAt={problem?.at ?? null} noticedAt={noticedAt} />
          <p className="small muted">{series.length < 2 ? (shopUp ? 'The line starts after FailSight has finished learning and made two checks (about 3 minutes after step 1).' : 'No line yet. Press Start in step 1 and it appears here.') : 'Lower is better. The red dashed line is the moment the problem started.'}</p>

          <Detailed>
            <div className="numbers">
              <h3>Numbers</h3>
              <ul>
                <li>Risk score: {last ? last.risk.toFixed(3) : '–'} (alert level {state?.config.threshold?.toFixed(3) ?? '–'}); an alert needs 2 of the last 3 checks above it.</li>
                <li>Latest check ended {last ? clock12(last.window_end) : '–'}; checks so far: {app?.counters.windows ?? 0}; alerts: {app?.counters.alerts ?? 0}; requests seen: {(app?.counters.events_api ?? 0).toLocaleString()}.</li>
                <li>Model: {state?.config.model} on {state?.config.feature_set} features, frozen {state?.config.frozen_at}. Live updates: {connected ? 'on' : 'off'}.</li>
              </ul>
              {(notice ?? last)?.reasons?.length ? (
                <table className="rt"><caption className="visually-hidden">Strongest readings behind the latest check</caption>
                  <thead><tr><th scope="col">Reading</th><th scope="col">Now</th><th scope="col">Normal</th></tr></thead>
                  <tbody>{(notice ?? last)!.reasons.map((r) => <tr key={r.feature}><td data-label="Reading">{r.plain_name}</td><td data-label="Now">{r.value.toPrecision(3)}</td><td data-label="Normal">{r.normal == null ? '–' : r.normal.toPrecision(3)}</td></tr>)}</tbody></table>
              ) : null}
            </div>
          </Detailed>
        </li>
      </ol>

      <MonitorLimits />
    </>
  )
}
