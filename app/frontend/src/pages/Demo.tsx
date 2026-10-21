import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ResponseChart, StatusBar, StatusSentence } from '../components/DemoViz'
import type { Pt } from '../components/DemoViz'
import Term from '../components/Term'
import { Detailed } from '../mode'
import { BREAKS, clock12, levelOf, plainReason, sentence } from '../plain'
import { useFaults, useTestbed } from '../testbed'
import { usePageTitle } from '../usePageTitle'

interface Broke { at: number; label: string; what: string; type: string; seconds: number; alreadyAlerting: boolean }

export function DemoLimits() {
  return (
    <aside className="callout limits-box" aria-labelledby="lim-h">
      <h2 id="lim-h" className="label">What this demo does not prove</h2>
      <ul>
        <li>It was tested on <strong>one app</strong>: this small practice shop.</li>
        <li>The test was <strong>small</strong>: 12 test runs with 15 problems, so the scores are rough.</li>
        <li><strong>Other apps are unproven.</strong> A new app needs its own check.</li>
      </ul>
      <p className="small">All the limits, in full: <Link to="/examiners/limits">Limits</Link>.</p>
    </aside>
  )
}

export default function Demo() {
  usePageTitle('Demo')
  const { state, windows, connected, error, app } = useTestbed('shop')
  const faults = useFaults()
  const [broke, setBroke] = useState<Broke | null>(null)
  const [starting, setStarting] = useState(false)
  const [startErr, setStartErr] = useState<string | null>(null)
  const [stopNote, setStopNote] = useState<string | null>(null)

  const shopUp = faults.ready && !faults.err
  const checking = !faults.ready
  const last = app?.last ?? windows[windows.length - 1] ?? null
  const pct = Math.round(((last?.learning_progress ?? app?.learning.progress) ?? 0) * 100)
  const learned = shopUp && !!app?.learning.done
  const status = !shopUp ? null : (last?.status ?? 'learning')
  const crashedNow = !!broke && broke.type === 'crash' && !shopUp && Date.now() / 1000 < broke.at + 150     // the shop is down on purpose and restarts by itself
  const level = crashedNow ? 'problem' : levelOf(status, shopUp)

  const notice = useMemo(() => {
    if (!broke) return null
    if (broke.alreadyAlerting) return null                       // the earlier alarm was still on: the delay cannot be measured
    return windows.find((w) => w.alert && !w.learning && (w.emitted_at ?? w.window_end) >= broke.at) ?? null
  }, [windows, broke])
  const noticedAt = notice ? (notice.emitted_at ?? notice.window_end) : null
  const series: Pt[] = useMemo(() => windows.filter((w) => !w.learning && w.features?.api_latency_p95_ms != null).map((w) => ({ t: w.window_end, ms: w.features.api_latency_p95_ms as number })), [windows])
  const now = Date.now() / 1000
  const faultOn = faults.active.length > 0
  const missed = !!broke && !broke.alreadyAlerting && !notice && !faultOn && now > broke.at + broke.seconds + 45

  const why = plainReason((notice ?? (level === 'watch' || level === 'problem' ? last : null))?.reasons?.[0])

  async function start() {
    setStarting(true); setStartErr(null)
    try {
      const r = await fetch('/api/testbed/start', { method: 'POST' })
      const j = await r.json().catch(() => ({}))
      if (!r.ok) { setStartErr(j.error ?? `The shop could not be started (HTTP ${r.status}).`); setStarting(false) }
      else { faults.refresh(); window.setTimeout(() => setStarting(false), 90000) }
    } catch (e: any) { setStartErr(String(e?.message ?? e)); setStarting(false) }
  }
  async function stop() {
    const r = await fetch('/api/testbed/stop', { method: 'POST' })
    const j = await r.json().catch(() => ({}))
    setStopNote(j.stopped ? 'The shop was stopped.' : j.note ?? 'Nothing to stop.')
    setBroke(null)
    faults.refresh()
  }
  async function breakIt(id: string) {
    const b = BREAKS.find((x) => x.id === id)
    if (!b) return
    const at = Date.now() / 1000
    const alreadyAlerting = !!last?.alert
    const ok = await faults.inject(b.type, 'medium', b.duration_s)
    if (ok || b.type === 'crash') setBroke({ at, label: b.label, what: b.what, type: b.type, seconds: b.duration_s ?? 60, alreadyAlerting })
  }
  const surprise = () => breakIt(BREAKS[Math.floor(Math.random() * BREAKS.length)].id)
  async function fix() { await faults.clear() }

  return (
    <>
      <h1>See FailSight catch a broken website</h1>
      <p className="lede">FailSight watches a website and tells you when something goes wrong. Try it on our practice shop. It takes about three minutes.</p>

      <ol className="demo-steps">
        <li className="step" aria-labelledby="s1">
          <h2 id="s1"><span className="num" aria-hidden="true">1</span> Start the demo shop</h2>
          {error && <p role="alert" className="bad-text">FailSight itself is not answering: {error}. Reload the page, or restart it with start_testbed.bat.</p>}
          {checking ? <p role="status">Checking whether the shop is running…</p>
            : crashedNow ? <p role="status"><strong>The shop is down, because you crashed it.</strong> It starts again by itself in about 30 seconds. Nothing to do but wait.</p>
            : !shopUp ? (
              <>
                <button type="button" className="btn big" onClick={start} disabled={starting}>{starting ? 'Starting… about a minute' : 'Start the demo shop'}</button>
                <p className="hint">{starting ? 'The shop is starting. This page updates by itself.' : 'This starts a small ticket shop on this computer, with pretend visitors.'}</p>
                {startErr && <p role="alert" className="bad-text">{startErr}</p>}
                <p className="small muted">Does nothing happen? Double-click <code>start_testbed.bat</code> in the project folder instead.</p>
              </>
            ) : !learned ? (
              <>
                <p className="ok-text"><strong>✓ The shop is running.</strong></p>
                <label htmlFor="learn" className="progress-label">FailSight is <Term k="learning">learning what normal looks like</Term>: {pct}% done</label>
                <progress id="learn" value={pct} max={100} />
                <p className="hint">This takes about 2 minutes. Then step 2 unlocks. Nothing to do but wait.</p>
              </>
            ) : (
              <>
                <p className="ok-text"><strong>✓ The shop is running and FailSight knows what normal looks like.</strong></p>
                <p className="hint">Go on to step 2.</p>
                <button type="button" className="btn secondary small-btn" onClick={stop}>Stop the demo shop</button>
                {stopNote && <p className="small muted" role="status">{stopNote}</p>}
              </>
            )}
        </li>

        <li className={`step ${learned ? '' : 'locked'}`} aria-labelledby="s2">
          <h2 id="s2"><span className="num" aria-hidden="true">2</span> Break something</h2>
          {!learned && <p className="hint">{shopUp ? 'Wait for step 1 to finish. These buttons unlock by themselves.' : 'Start the shop in step 1 first. These buttons unlock when it is ready.'}</p>}
          {learned && <p className="hint">Pick one. It really breaks the shop for about a minute and a half. FailSight is not told which one you picked.</p>}
          <div className="break-grid">
            {BREAKS.map((b) => <button key={b.id} type="button" className="btn big choice" disabled={!learned} onClick={() => breakIt(b.id)}>{b.label}</button>)}
          </div>
          <div className="row" style={{ marginTop: 'var(--space-3)', alignItems: 'center' }}>
            <button type="button" className="btn secondary" disabled={!learned} onClick={surprise}>Surprise me</button>
            <button type="button" className="btn secondary" disabled={!learned || !faultOn} onClick={fix}>Fix it now</button>
          </div>
          <p className="small" role="status" aria-live="polite" style={{ marginTop: 'var(--space-3)' }}>
            {broke && faultOn ? <>You broke it at <strong>{clock12(broke.at)}</strong>: {broke.what}.</> : broke ? 'The problem is over. Pick another one, or look at step 3.' : learned ? 'Nothing is broken yet.' : ''}
          </p>
        </li>

        <li className="step" aria-labelledby="s3">
          <h2 id="s3"><span className="num" aria-hidden="true">3</span> What does FailSight say?</h2>
          <StatusSentence level={level} text={crashedNow ? 'Something is wrong: the whole shop is down' : sentence(level, notice ?? last, pct)} />
          <StatusBar level={level} />
          {level === 'idle' && <p className="hint">Press <strong>Start the demo shop</strong> in step 1. The answer shows up here.</p>}
          {level === 'learning' && <p className="hint">Nothing can be judged yet. In about 2 minutes this turns into a real answer.</p>}
          {(level === 'ok' || level === 'watch' || level === 'problem') && !broke && <p className="hint">{level === 'ok' ? 'Now break something in step 2 and watch this change.' : 'You have not broken anything on this page. Either you broke it earlier, or the shop had a short bad moment.'}</p>}
          {broke && (
            <ul className="timeline" aria-label="What happened, in order">
              <li>You broke it at <strong>{clock12(broke.at)}</strong>.</li>
              {notice && noticedAt ? <li className="found">FailSight noticed at <strong>{clock12(noticedAt)}</strong> ({Math.max(0, Math.round(noticedAt - broke.at))} seconds later).</li>
                : broke.alreadyAlerting ? <li>FailSight was still on alert from the earlier problem, so we cannot time it. Wait for “Everything looks normal”, then try again.</li>
                : missed ? <li>FailSight did not raise an alert this time. This can happen: see the limits below.</li>
                  : <li>FailSight has not noticed yet. It looks every 10 seconds and needs about 30 seconds to be sure.</li>}
            </ul>
          )}
          {(notice || level === 'watch' || level === 'problem') && <p className="reason"><strong>Why:</strong> {why.reason} <span className="muted">FailSight is not told what you broke. It only sees numbers, so this is what it reacted to, not a proven cause.</span></p>}

          <h3 className="chart-title">How long pages take to open</h3>
          <ResponseChart points={series} brokeAt={broke?.at ?? null} noticedAt={noticedAt} />
          <p className="small muted">{series.length < 2 ? (shopUp ? 'The line starts after FailSight has finished learning and made two checks (about 3 minutes after step 1).' : 'No line yet. Press Start in step 1 and it appears here.') : 'Lower is better. The red dashed line is the moment you broke it.'}</p>

          <Detailed>
            <div className="numbers">
              <h3>Numbers</h3>
              <ul>
                <li>Risk score: {last ? last.risk.toFixed(3) : '–'} (alert level {state?.config.threshold?.toFixed(3) ?? '–'}); an alert needs 2 of the last 3 checks above it.</li>
                <li>Latest check ended {last ? clock12(last.window_end) : '–'}; checks so far: {app?.counters.windows ?? 0}; alerts: {app?.counters.alerts ?? 0}; requests seen: {(app?.counters.events_api ?? 0).toLocaleString()}.</li>
                <li>Model: {state?.config.model} on {state?.config.feature_set} features, frozen {state?.config.frozen_at}. Live updates: {connected ? 'on' : 'off'}.</li>
                {broke && <li>Real fault injected: <code>{broke.type}</code>, medium strength, {broke.seconds} s.</li>}
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

      <DemoLimits />
    </>
  )
}
