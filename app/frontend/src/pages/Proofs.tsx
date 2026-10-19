import { useRef, useState } from 'react'
import { useApi } from '../api'
import Term from '../components/Term'
import { Async } from '../components/States'
import { Detailed, Simple } from '../mode'
import { usePageTitle } from '../usePageTitle'

interface Item { claim: string; status: string; evidence: string[]; source: string; result?: string | null }
interface Section { id: string; title: string; status: string; items: Item[]; generated_at?: string }
interface Report { generated_at: string; partial: boolean; refreshed_sections?: string[]; runtime_seconds: number; totals: Record<string, number>; git_commit: string; sections: Section[]; summary: { section: string; claim: string; status: string; source: string }[]; text: string }

const PLAIN: Record<string, string> = {
  P1: 'The data exists and is as big as we say. Counted again from the raw files.',
  P2: 'Practice data and test data are kept apart, and nothing sits in both.',
  P3: 'Extra practice examples for rare cases are made from practice data only. Test and checking data are never touched.',
  P4: 'Practice data is cut into ten parts for checking, without mixing neighbouring windows.',
  P5: 'Every saved model is loaded again and scored on the test windows. The new scores are compared with the report.',
  P6: 'We looked for models that memorise their practice data, retrained those that did, and compared the first run with the full retrain.',
  P7: 'Checks for peeking at the test answers (data leakage), exactly as reported by the audit.',
  P8: 'The explanations for three test windows can be recomputed, and their parts add up to the model’s output.',
  P9: 'On a different system the models did not work. This is reported exactly as it is.',
  P10: 'The seeds, library versions and code version, so anyone can repeat this.',
}
const QUICK = ['P1', 'P2', 'P3', 'P4', 'P10']
const badge = (s: string) => <span className={`badge ${s.split(' ')[0].toLowerCase()}`}>{s}</span>

export default function Proofs() {
  usePageTitle('Proofs')
  const rep = useApi<Report>('/api/proofs')
  const [view, setView] = useState<'checklist' | 'terminal'>('checklist')
  const [scope, setScope] = useState<'quick' | 'full'>('quick')
  const [running, setRunning] = useState(false)
  const [out, setOut] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const pre = useRef<HTMLPreElement>(null)

  async function run() {
    setRunning(true); setErr(null); setOut('')
    try {
      const res = await fetch('/api/proofs/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(scope === 'quick' ? { sections: QUICK } : {}) })
      if (!res.ok || !res.body) {
        const j = await res.json().catch(() => ({}))
        throw new Error(j.error ?? `The run could not start (HTTP ${res.status}).`)
      }
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        setOut((o) => o + dec.decode(value, { stream: true }))
        requestAnimationFrame(() => { if (pre.current) pre.current.scrollTop = pre.current.scrollHeight })
      }
    } catch (e: any) {
      setErr(e?.message ?? String(e))
    } finally {
      setRunning(false)
      rep.reload()
    }
  }

  return (
    <>
      <h1>Proofs</h1>
      <p className="lede"><Simple>Every number in this app can be checked. These checks recompute them from the data and the saved models. You can run them again here, or in a terminal with <code>python scripts/proofs.py</code>.</Simple><Detailed>Numbered proof report P1–P10 produced by <code>scripts/proofs.py</code>; this page renders <code>report_outputs/proofs.json</code> and shows the same text the terminal prints.</Detailed></p>

      <Async state={rep} rows={6} label="Loading the proof report" level={2}>
        {(r) => (
          <>
            <ul className="tiles six" aria-label="Result of the checks" style={{ marginBottom: 'var(--space-4)' }}>
              <li className="good"><div className="n">{r.totals.PASS ?? 0}</div><div className="c">Passed</div></li>
              <li><div className="n">{r.totals.WARN ?? 0}</div><div className="c">Passed with a caveat (warn)</div></li>
              <li className="bad"><div className="n">{r.totals.FAIL ?? 0}</div><div className="c">Failed</div></li>
              {(r.totals['REPORTED ACCURATELY'] ?? 0) > 0 && <li className="neutral"><div className="n">{r.totals['REPORTED ACCURATELY']}</div><div className="c">Reported accurately (a result, not a pass)</div></li>}
              <li><div className="n" style={{ fontSize: '1.2rem' }}>{r.generated_at}</div><div className="c">Last run{r.partial ? ' (some sections older)' : ''}</div></li>
              <li><div className="n" style={{ fontSize: '1.2rem' }}>{r.git_commit.slice(0, 8)}</div><div className="c">Code version</div></li>
            </ul>
            {(r.totals.FAIL ?? 0) > 0 && <div className="callout" role="alert"><span className="label">A check failed</span><p>Read the evidence below before trusting the affected numbers.</p></div>}
            <p className="takeaway">
              {(r.totals.FAIL ?? 0) === 0 ? `All ${r.summary.length} checks ran without a failure; ${r.totals.WARN ?? 0} carry a caveat that is stated next to them${(r.totals['REPORTED ACCURATELY'] ?? 0) > 0 ? `; ${r.totals['REPORTED ACCURATELY']} are results that are not successes (the LO2 test), checked only for being reported faithfully` : ''}.` : 'At least one check failed.'}
            </p>

            <div className="card stack" style={{ marginTop: 'var(--space-5)' }}>
              <h2 style={{ margin: 0, fontSize: '1.3rem' }}>Run the checks again</h2>
              <div className="row">
                <div className="field">
                  <label htmlFor="scope">How much to run</label>
                  <select id="scope" value={scope} onChange={(e) => setScope(e.target.value as 'quick' | 'full')} disabled={running}>
                    <option value="quick">Quick (data, split, folds, versions; about a minute)</option>
                    <option value="full">Full (also reloads every model; several minutes)</option>
                  </select>
                </div>
                <button type="button" className="btn" onClick={run} disabled={running}>{running ? 'Running…' : 'Re-run the checks'}</button>
                <a className="btn secondary" style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }} href="/api/proofs/download.json" download>Download proof report (JSON)</a>
                <a className="btn secondary" style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }} href="/api/proofs/report.html?download=1">Download printable report (HTML)</a>
                <a href="/api/proofs/report.html" target="_blank" rel="noreferrer">Open printable view</a>
              </div>
              <p className="small muted" style={{ margin: 0 }}>A quick run refreshes only its own sections in the report; the rest keep their earlier results (shown with their own time). To get a PDF, open the printable view and use your browser’s Print command.</p>
              {(running || out || err) && (
                <div>
                  <h3 style={{ fontSize: '0.95rem', margin: '0 0 var(--space-2)' }}>Live output of <code>python scripts/proofs.py</code></h3>
                  {err && <p role="alert"><strong>{err}</strong></p>}
                  <pre ref={pre} className="console-pre" role="log" aria-label="Output of the proof run" tabIndex={0} data-testid="proof-console">{out || (running ? 'Starting…' : '')}</pre>
                </div>
              )}
            </div>

            <div className="tabs" role="tablist" aria-label="View" style={{ marginTop: 'var(--space-6)' }}>
              <button role="tab" aria-selected={view === 'checklist'} onClick={() => setView('checklist')}>Checklist</button>
              <button role="tab" aria-selected={view === 'terminal'} onClick={() => setView('terminal')}>Terminal output (identical)</button>
            </div>
            {view === 'terminal' ? (
              <pre className="console-pre" style={{ maxHeight: '70vh' }} tabIndex={0} aria-label="The proof report exactly as the terminal prints it">{r.text}</pre>
            ) : (
              <div>
                {r.sections.map((s) => (
                  <details key={s.id} className="acc" open={s.id === 'P5' || s.id === 'P7' || s.id === 'P9'}>
                    <summary>{badge(s.status)} <strong>{s.id}</strong> &nbsp;{s.title}{s.status === 'REPORTED ACCURATELY' && s.items.find((i) => i.result)?.result ? <span className="result-line"> &mdash; Result: {s.items.find((i) => i.result)?.result}</span> : null}</summary>
                    <p className="muted" style={{ marginTop: 0 }}>{PLAIN[s.id]}{s.generated_at ? <span className="small"> (run {s.generated_at})</span> : null}</p>
                    <ul style={{ listStyle: 'none', padding: 0, maxWidth: 'none' }}>
                      {s.items.map((it, i) => (
                        <li key={i} style={{ marginBottom: 'var(--space-4)' }}>
                          <div>{badge(it.status)} <strong>{it.claim}</strong></div>
                          {it.result && <div className="result-line">Result: {it.result}</div>}
                          <Simple>
                            <details><summary className="small">Show the numbers and the source</summary><pre className="evidence">{it.evidence.join('\n')}</pre><span className="small muted">source: {it.source}</span></details>
                          </Simple>
                          <Detailed>
                            <pre className="evidence">{it.evidence.join('\n')}</pre><span className="small muted">source: <code>{it.source}</code></span>
                          </Detailed>
                        </li>
                      ))}
                    </ul>
                  </details>
                ))}
              </div>
            )}
            <div className="callout" style={{ marginTop: 'var(--space-6)' }}>
              <span className="label">Two things to keep in mind</span>
              <ul>
                <li>The audit check “files unchanged” (C7) recorded its baseline <strong>during the audit run itself</strong>. It proves that no file changed <em>since then</em>, not before.</li>
                <li>The warning on the time overlap (C3) stays a warning: the overlap is real and its measured effect is stated in the evidence.</li>
              </ul>
              <p className="small"><Term k="leakage">What is data leakage?</Term></p>
            </div>
          </>
        )}
      </Async>
    </>
  )
}
