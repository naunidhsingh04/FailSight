import { useCallback, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { StatusIcon, StepIcon } from '../components/Icons'
import { Gauge, RiskTimeline } from '../components/RiskCharts'
import Term from '../components/Term'
import WhyPanel from '../components/WhyPanel'
import { Detailed, Simple, useMode } from '../mode'
import { SYSTEMS, SYSTEM_NAME, clock, percent, setModel, useMonitor, useReplay } from '../monitor'
import type { Status, Sys, SysState, WinEvent } from '../monitor'
import { usePageTitle } from '../usePageTitle'

export const SCENARIOS: { id: string; label: string; hint: string }[] = [
  { id: 'Normal', label: 'Normal day', hint: 'Nothing is wrong. Good for checking for false alarms.' },
  { id: 'Performance', label: 'Slow performance', hint: 'The system is overloaded or slow.' },
  { id: 'Service', label: 'A service goes down', hint: 'One part of the application stops responding.' },
  { id: 'Database', label: 'Database trouble', hint: 'The data store is limited or stuck.' },
  { id: 'Code', label: 'Code error', hint: 'A bug makes requests fail.' },
]
const WORD: Record<Status, string> = { idle: 'Waiting', healthy: 'Healthy', watch: 'Watch', failing: 'Failing' }
const SENTENCE: Record<Status, string> = {
  idle: 'No data yet. Choose a scenario and press Play.',
  healthy: 'Everything looks normal.',
  watch: 'One recent window looked unusual. Keeping an eye on it.',
  failing: 'Several recent windows look like a failure.',
}

export function HowStrip() {
  const steps: [1 | 2 | 3 | 4, string, string][] = [
    [1, 'Watch the traffic', 'A collector sends requests, log lines and system readings to us.'],
    [2, 'Measure it', 'Every 10 seconds we summarise the last 30 seconds into numbers.'],
    [3, 'Estimate failure risk', 'A trained model turns those numbers into a risk from 0% to 100%.'],
    [4, 'Explain why', 'We show which signals pushed the risk up, in plain words.'],
  ]
  return (
    <ol className="steps" aria-label="How it works in four steps">
      {steps.map(([n, h, p]) => (
        <li key={n}><StepIcon n={n} /><div><h3>{n}. {h}</h3><p>{p}</p></div></li>
      ))}
    </ol>
  )
}

function SystemCard({ sys, st, threshold }: { sys: Sys; st: SysState; threshold: number | null }) {
  const { detailed } = useMode()
  const last = st.last
  return (
    <li>
      <section className={`sys-card ${st.status}`} aria-labelledby={`h-${sys}`}>
        <h3 id={`h-${sys}`}>{SYSTEM_NAME[sys]}</h3>
        <div className={`status-word ${st.status}`} role="status" aria-live="polite" aria-atomic="true">
          <StatusIcon status={st.status} />
          <span><span className="visually-hidden">{SYSTEM_NAME[sys]} status: </span>{WORD[st.status]}</span>
        </div>
        <p className="small muted" style={{ marginTop: 0 }}>{SENTENCE[st.status]}</p>
        <div className="gauge-wrap"><Gauge risk={st.risk} threshold={threshold} status={st.status} label={SYSTEM_NAME[sys]} /></div>
        {last && <p className="small muted" style={{ marginBottom: 0 }}>Last check: recording time {clock(last.window_end)} UTC</p>}
        <Detailed>
          <p className="small" style={{ marginBottom: 0 }}>
            risk {st.risk === null ? '–' : st.risk.toFixed(4)}; alert line {threshold ?? '–'}; windows {st.counters.windows}; completeness: {last ? last.missing.tier : '–'}{last?.warmup ? '; warm-up window' : ''}{detailed ? '' : ''}
          </p>
        </Detailed>
      </section>
    </li>
  )
}

export default function Console() {
  usePageTitle('Console')
  const { systems, alerts, config, models, connected, error, reset } = useMonitor()
  const rep = useReplay()
  const { detailed } = useMode()
  const [scenario, setScenario] = useState('Performance')
  const [speed, setSpeed] = useState(10)
  const [selected, setSelected] = useState<WinEvent | null>(null)
  const [tab, setTab] = useState<Sys>('SN')
  const lastBtn = useRef<HTMLElement | null>(null)
  const threshold = systems.SN.last?.decision_rule.value ?? systems.TT.last?.decision_rule.value ?? 0.99
  const running = rep.status.running
  const paused = rep.status.paused

  const play = async () => {
    if (running && paused) return rep.pause(false)
    reset(); setSelected(null)
    await rep.start(scenario, speed, 2)
  }
  const close = useCallback(() => { setSelected(null); lastBtn.current?.focus() }, [])

  const totals = useMemo(() => {
    const c = SYSTEMS.map((s) => systems[s].counters)
    const sum = (k: keyof typeof c[0]) => c.reduce((a, x) => a + (x[k] as number), 0)
    return { windows: sum('windows'), alerts: sum('alerts'), correct: sum('correct_alerts'), false: sum('false_alarms'), missed: sum('missed'), labelled: sum('labelled'), quiet: sum('correct_quiet') }
  }, [systems])
  const failureWindows = totals.correct + totals.missed
  const hist = systems[tab].history
  const flaggedN = hist.filter((p) => p.flagged).length
  const timelineLabel = `Failure risk over the last ${hist.length} windows of the ${SYSTEM_NAME[tab]}; ${flaggedN} crossed the alert line`

  return (
    <>
      <h1>Is the system healthy right now?</h1>
      <p className="lede"><Simple>Watch two sample applications while recorded data plays back. The page says in plain words whether each one looks healthy, and why.</Simple><Detailed>Live view of the failure-risk model: windows built from replayed events, scored with the saved decision rule, explained with feature contributions.</Detailed></p>

      <div className="banner" role="note">
        <strong>Replay of recorded AnoMod data</strong>
        <p>This is a demonstration. Recorded data from the test part of the dataset is sent through the real pipeline. It is <em>not</em> a live production system, and the “right answers” are <Term k="recording">known from the recording</Term>.</p>
      </div>

      <h2 style={{ marginTop: 'var(--space-6)' }}>How it works</h2>
      <HowStrip />
      <p className="small"><Link to="/how-it-works">Read more on How it works</Link></p>

      <h2 style={{ marginTop: 'var(--space-6)' }}>1. Choose what to replay</h2>
      <div className="controls" role="group" aria-label="Replay controls">
        <div className="field" style={{ flex: '1 1 20rem' }}>
          <span className="label" id="scn-l">Scenario</span>
          <div className="pill-group" role="group" aria-labelledby="scn-l">
            {SCENARIOS.map((s) => <button key={s.id} type="button" className="pill" aria-pressed={scenario === s.id} onClick={() => setScenario(s.id)} title={s.hint}>{s.label}</button>)}
          </div>
        </div>
        <div className="field">
          <span className="label" id="spd-l">Speed</span>
          <div className="segmented" role="group" aria-labelledby="spd-l">
            {[1, 10, 60].map((s) => <button key={s} type="button" aria-pressed={speed === s} onClick={() => { setSpeed(s); if (running) rep.speed(s) }}>{s}&times;</button>)}
          </div>
        </div>
        <div className="field">
          <span className="label">&nbsp;</span>
          <div className="row" style={{ gap: 'var(--space-2)' }}>
            <button type="button" className="btn play" onClick={play} disabled={rep.busy || (running && !paused)}>{running && paused ? 'Resume' : 'Play'}</button>
            <button type="button" className="btn secondary" onClick={() => rep.pause(true)} disabled={!running || paused || rep.busy}>Pause</button>
            <button type="button" className="btn danger" onClick={() => rep.stop()} disabled={!running || rep.busy}>Stop</button>
          </div>
        </div>
        <Detailed>
          <div className="field">
            <label htmlFor="model">Detection model</label>
            <select id="model" value={config.model} onChange={(e) => setModel(e.target.value)}>
              {(models.length ? models : [config.model]).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </Detailed>
      </div>
      <p className="small muted" role="status" aria-live="polite">
        {error ? `Could not reach the server: ${error}` : rep.err ? rep.err : running ? `${paused ? 'Paused' : 'Playing'}: ${SCENARIOS.find((s) => s.id === rep.status.info.scenario)?.label ?? rep.status.info.scenario} at ${rep.status.speed ?? rep.status.info.speed}×.` : 'Not playing.'}
        {' '}{connected ? 'Live updates connected.' : 'Connecting to live updates…'}
        {SCENARIOS.find((s) => s.id === scenario)?.hint ? ` ${SCENARIOS.find((s) => s.id === scenario)?.hint}` : ''}
      </p>

      <h2>2. Is it healthy?</h2>
      <div className={`console-grid ${selected ? 'with-panel' : ''}`}>
        <div className="stack" style={{ minWidth: 0 }}>
          <ul className="board" aria-label="System health">
            {SYSTEMS.map((s) => <SystemCard key={s} sys={s} st={systems[s]} threshold={threshold} />)}
          </ul>
          <p className="takeaway">
            {totals.windows === 0 ? 'Press Play to start. Each system is checked every 10 seconds of recording time.' :
              `So far ${totals.windows} windows were checked and ${totals.alerts} raised an alert.`}
          </p>

          <section className="card" aria-labelledby="tl-h">
            <h2 id="tl-h" style={{ marginTop: 0, fontSize: '1.3rem' }}>Failure risk over time</h2>
            <div className="tabs" role="tablist" aria-label="Choose a system">
              {SYSTEMS.map((s) => <button key={s} role="tab" aria-selected={tab === s} onClick={() => setTab(s)}>{SYSTEM_NAME[s]}</button>)}
            </div>
            <div className="tl-wrap"><RiskTimeline points={hist} threshold={threshold} ariaLabel={timelineLabel} /></div>
            <div className="legend-row" aria-hidden="true">
              <span><span className="k" style={{ background: 'var(--fail)', borderColor: 'var(--fail)', borderRadius: 2 }} />window that raised an alert</span>
              <span><span className="k" style={{ background: 'var(--bg)' }} />window with no alert</span>
              <span><span className="k" style={{ background: 'var(--ink-3)', borderColor: 'var(--ink-3)' }} />warm-up (first 20 s)</span>
            </div>
            <p className="takeaway">
              {hist.length === 0 ? 'The line appears here once data arrives.' :
                flaggedN === 0 ? `None of the last ${hist.length} windows crossed the alert line.` : `${flaggedN} of the last ${hist.length} windows crossed the alert line (the dashed red line).`}
            </p>
            <Detailed>
              <details><summary>Show the numbers</summary>
                <div className="table-wrap"><table className="rt"><caption className="visually-hidden">Recent windows</caption>
                  <thead><tr><th scope="col">Window</th><th scope="col">Time (UTC)</th><th scope="col">Risk</th><th scope="col">Alert</th><th scope="col">Recording says</th></tr></thead>
                  <tbody>{[...hist].reverse().slice(0, 12).map((p) => (
                    <tr key={p.window_start}><td data-label="Window" className="num-cell">{p.window_index}{p.warmup ? ' (warm-up)' : ''}</td><td data-label="Time (UTC)">{clock(p.window_start)}</td><td data-label="Risk" className="num-cell">{p.risk.toFixed(4)}</td><td data-label="Alert">{p.flagged ? 'yes' : 'no'}</td><td data-label="Recording says">{p.truth ?? '–'}</td></tr>
                  ))}</tbody></table></div>
              </details>
            </Detailed>
          </section>

          <section aria-labelledby="cnt-h">
            <h2 id="cnt-h" style={{ fontSize: '1.3rem' }}>3. How is it doing? <span className="small muted" style={{ fontFamily: 'var(--font-sans)', fontWeight: 400 }}>(checked against the answers <Term k="recording">known from the recording</Term>)</span></h2>
            <ul className="tiles" aria-label="Live counters">
              <li><div className="n">{totals.windows}</div><div className="c">Windows checked</div></li>
              <li><div className="n">{totals.alerts}</div><div className="c">Alerts raised</div></li>
              <li className="good"><div className="n">{totals.correct}</div><div className="c"><Term k="correctalert">Correct alerts</Term></div></li>
              <li className="bad"><div className="n">{totals.false}</div><div className="c"><Term k="falsealarm">False alarms</Term></div></li>
              <li className="bad"><div className="n">{totals.missed}</div><div className="c"><Term k="missed">Missed failures</Term></div></li>
            </ul>
            <p className="takeaway">
              {totals.labelled === 0 ? 'The score appears once windows have been checked.' :
                failureWindows === 0 ? `${totals.false} false alarm${totals.false === 1 ? '' : 's'} in ${totals.false + totals.quiet} normal windows.` :
                  `It caught ${totals.correct} of ${failureWindows} failure windows and raised ${totals.false} false alarm${totals.false === 1 ? '' : 's'}.`}
              {' '}The first two windows of each replay are a warm-up and are not counted.
            </p>
          </section>

          <section aria-labelledby="al-h">
            <h2 id="al-h" style={{ fontSize: '1.3rem' }}>4. Alerts and why</h2>
            {alerts.length === 0 ? (
              <div className="state"><h3>No alerts yet</h3><p>When a window crosses the alert line it appears here. Select one to see why it was raised.</p></div>
            ) : (
              <ul className="alerts" aria-label="Alert feed, newest first">
                {alerts.map((a) => {
                  const key = `${a.system}-${a.session}-${a.window_index}`
                  const cur = selected && selected.system === a.system && selected.session === a.session && selected.window_index === a.window_index
                  return (
                    <li key={key}>
                      <button type="button" className="alert-btn" aria-current={cur ? 'true' : undefined} aria-controls="why-panel" onClick={(e) => { lastBtn.current = e.currentTarget; setSelected(a) }}>
                        <span className="t">{SYSTEM_NAME[a.system]} &middot; {clock(a.window_start)} UTC &middot; risk {percent(a.risk)}{a.truth ? (a.truth.label_binary === 'Anomaly' ? ' · correct alert' : ' · false alarm') : ''}</span>
                        <span className="s">{a.summary}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>
        </div>
        {selected && <WhyPanel ev={selected} onClose={close} />}
      </div>
      {detailed && <p className="small muted">Counters: {systems.SN.counters.events_api + systems.TT.counters.events_api} API, {systems.SN.counters.events_logs + systems.TT.counters.events_logs} log and {systems.SN.counters.events_metrics + systems.TT.counters.events_metrics} metric events ingested.</p>}
    </>
  )
}
