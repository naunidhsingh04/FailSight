import { useEffect, useRef } from 'react'
import Term from './Term'
import { Detailed } from '../mode'
import { SYSTEM_NAME, clock, percent } from '../monitor'
import type { WinEvent } from '../monitor'

const LEVEL_PLAIN: Record<string, string> = { Performance: 'slow performance', Service: 'a service going down', Database: 'database trouble', Code: 'a code error', Normal: 'normal operation' }

/** "Why this alert?": plain sentences built from the real top signals and their measured values; technical table in Detailed mode. */
export default function WhyPanel({ ev, onClose }: { ev: WinEvent; onClose: () => void }) {
  const head = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    head.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [ev, onClose])
  const t = ev.truth
  const verdict = t ? (ev.flagged ? (t.label_binary === 'Anomaly' ? 'correct alert' : 'false alarm') : (t.label_binary === 'Anomaly' ? 'missed failure' : 'correct quiet')) : null
  const mt = ev.missing
  return (
    <aside className="sheet" role="dialog" aria-modal="false" aria-labelledby="why-h" id="why-panel">
      <button type="button" className="btn secondary sheet-close" onClick={onClose} aria-label="Close this explanation">Close</button>
      <h2 id="why-h" tabIndex={-1} ref={head}>Why this alert?</h2>
      <p className="small muted" style={{ marginTop: 0 }}>{SYSTEM_NAME[ev.system]} &middot; recording time {clock(ev.window_start)}&ndash;{clock(ev.window_end)} UTC &middot; failure risk <strong>{percent(ev.risk)}</strong></p>
      <p style={{ fontSize: '1.05rem', maxWidth: 'none' }}><strong>{ev.summary}</strong></p>
      {ev.reasons.length > 0 && (
        <>
          <h3 style={{ fontSize: '0.95rem', marginTop: 'var(--space-4)' }}>What the system looked at</h3>
          <ul className="reason-list">{ev.reasons.map((r) => <li key={r.feature}>{r.sentence.charAt(0).toUpperCase() + r.sentence.slice(1)}.</li>)}</ul>
        </>
      )}
      <div className={`dq ${mt.tier !== 'normal' ? 'bad' : ''}`}>
        <strong>Data quality:</strong> {mt.missing} of {mt.expected} readings this system normally provides were missing in this window ({percent(mt.fraction)}).
        {mt.tier !== 'normal' && ' With so much missing the alert is less trustworthy.'}
      </div>
      {t && (
        <div className="muted-box">
          <strong>What the recording says</strong> <Term k="recording" />: this window was {t.label_binary === 'Anomaly' ? <>a failure ({LEVEL_PLAIN[t.label_level ?? ''] ?? t.label_level})</> : 'normal'}. So this alert was a <strong>{verdict}</strong>.
        </div>
      )}
      <p className="small muted" style={{ marginTop: 'var(--space-3)' }}>These are the signals the model leaned on. They show what it reacted to, not a proven root cause.</p>
      <Detailed>
        <h3 style={{ fontSize: '0.95rem' }}>Technical detail</h3>
        <p className="small">Model {ev.model}; decision rule: alert when risk &ge; {ev.decision_rule.value ?? '–'}; window {ev.window_index} of this session{ev.warmup ? ' (warm-up)' : ''}; {ev.n_features_built} features built.</p>
        <div className="table-wrap stacked">
          <table className="rt">
            <caption className="visually-hidden">Top 10 features by contribution (SHAP)</caption>
            <thead><tr><th scope="col">Feature</th><th scope="col">Group</th><th scope="col">Value</th><th scope="col">Normal</th><th scope="col">Push</th></tr></thead>
            <tbody>
              {ev.top_features.map((f) => (
                <tr key={f.feature}>
                  <td data-label="Feature" className="name-cell"><code>{f.feature}</code></td>
                  <td data-label="Group">{f.group}</td>
                  <td data-label="Value" className="num-cell">{f.raw_value === null ? 'missing' : Number(f.raw_value).toPrecision(4)}</td>
                  <td data-label="Normal" className="num-cell">{f.normal_value === null || f.normal_value === undefined ? '–' : Number(f.normal_value).toPrecision(4)}</td>
                  <td data-label="Push" className="num-cell">{f.shap >= 0 ? '+' : ''}{f.shap.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details><summary>Raw event</summary><pre className="evidence">{JSON.stringify({ ...ev, top_features: undefined, reasons: undefined }, null, 1)}</pre></details>
      </Detailed>
    </aside>
  )
}
