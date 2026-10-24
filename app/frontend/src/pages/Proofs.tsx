import { useRef, useState } from 'react'
import { useApi } from '../api'
import Term from '../components/Term'
import { Link } from 'react-router-dom'
import { Async, EmptyState } from '../components/States'
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
  P11: 'Do features that exist in both systems, shown relative to each system’s own normal, let a model trained on one system work on the other? This is the key test for using FailSight on other applications.',
  P14: 'Can a model tell which recording a window comes from? If so, recognising the recording alone would reproduce the scores, and they would say little about spotting faults.',
  P15: 'Do exact fault start and end times exist, so that windows before, during and after a fault can be labelled? If not, the within-recording test and early-warning labels cannot be built honestly.',
  P12: 'The live demo: three 12-minute runs of the real shop with the real agent and monitor, on new seeds, with the fault schedule written down before each run. Reported as it came out.',
  P18: 'A test on a public outside benchmark (RCAEval): do the per-app detectors from the testbed find injected faults in other people’s systems? Testing only; nothing was tuned on it.',
  P19: 'Can another app be connected? The agent, the log tailer, the Node middleware, the optional API key and the Slack-compatible webhook are checked here; whether the detector works on that app is not.',
  P16: 'The testbed results on whole runs the models never saw: how well faults are detected, how fast, with how many false alarms.',
  P17: 'Can the testbed result be explained by recognising which run a window comes from? Here every run contains both normal and faulty windows, so it should not be.',
  P20: 'The five hybrid models on the benchmark data are loaded again and scored again; the split, the extra practice examples and the shuffled-answer control are checked.',
  P21: 'The second data set (LO2) is kept completely apart from the first: no window or run in both practice and test parts, the test part is untouched, and no file of one is used by the other.',
  P22: 'The new 5-part checking folds never put neighbouring windows in two different parts, and every kind of window appears in every part where that is possible.',
  P23: 'The Google Colab notebook runs from top to bottom on a clean computer, and the numbers it prints are the same as the numbers here.',
  P24: 'The second data set (LO2) is split 80% practice and 20% test by whole runs; the practice part is balanced with SMOTE and the test part is left exactly as recorded. The same is checked for training on both data sets together.',
  P13: 'Do the models lean on patterns of missing data instead of real signals? We hide parts of the data, train small look-alike models on the missing-data patterns alone, and remove those columns, then compare.',
}
const QUICK = ['P1', 'P2', 'P3', 'P4', 'P10']
const badge = (s: string) => <span className={`badge ${s.split(' ')[0].toLowerCase()}`}>{s}</span>

interface TbResults {
  frozen: { model: string; feature_set: string; alert_rule: string } | null
  overall: { method: string; balanced_accuracy: number; balanced_accuracy_ci_low: number; balanced_accuracy_ci_high: number; n_test_runs: number }[]
  instances: { method: string; rule: string; detected: boolean }[]
  false_alarms: { method: string; rule: string; fault_free_hours: number; false_alarms_per_hour_fault_free: number; false_alarms_per_hour_outside_faults: number }[]
  live: { actual_fault: string; detected: boolean | null }[]
}

/** The three numbers an examiner asks for first, read from the saved result files of the held-out test (never typed in here). */
function Headline() {
  const res = useApi<TbResults>('/api/testbed/results')
  return (
    <Async state={res} rows={3} label="Loading the headline numbers" level={2}>
      {(r) => {
        if (!r.frozen) return <EmptyState title="No test results on this computer yet" level={2}><p>The testbed results are missing. Create them with <code>python -m testbed.freeze</code>, or open the full list of checks below.</p></EmptyState>
        const method = `${r.frozen.model} (${r.frozen.feature_set})`
        const rule = r.frozen.alert_rule.replace(' windows', '')
        const o = r.overall.find((x) => x.method === method)
        const inst = r.instances.filter((x) => x.method === method && x.rule === rule)
        const caught = inst.filter((x) => x.detected).length
        const fa = r.false_alarms.find((x) => x.method === method && x.rule === rule)
        const liveFaults = r.live.filter((x) => x.actual_fault && x.actual_fault !== 'none')
        const liveCaught = liveFaults.filter((x) => x.detected).length
        return (
          <>
            <p className="proof-sentence">
              {inst.length ? `On test runs it had never seen, FailSight noticed ${caught} of the ${inst.length} problems we planted` : 'FailSight was checked on test runs it had never seen'}{fa ? `, with ${fa.false_alarms_per_hour_fault_free.toFixed(0)} false alarms per hour on quiet runs` : ''}. But the test is small and covers only one shop.
            </p>
            <table className="rt proof-table">
              <caption className="visually-hidden">The three numbers to know</caption>
              <thead><tr><th scope="col">Question</th><th scope="col">Answer</th><th scope="col">What it means</th></tr></thead>
              <tbody>
                <tr>
                  <td data-label="Question" className="name-cell">Score on test runs it never saw</td>
                  <td data-label="Answer" className="big-num">{o ? o.balanced_accuracy.toFixed(2) : '–'}<span className="of"> out of 1</span></td>
                  <td data-label="What it means">1.00 is perfect and 0.50 is a coin toss. {o ? `Likely between ${o.balanced_accuracy_ci_low.toFixed(2)} and ${o.balanced_accuracy_ci_high.toFixed(2)} (${o.n_test_runs} test runs).` : ''}</td>
                </tr>
                <tr>
                  <td data-label="Question" className="name-cell">Problems caught</td>
                  <td data-label="Answer" className="big-num">{inst.length ? `${caught} of ${inst.length}` : '–'}</td>
                  <td data-label="What it means">Planted problems in the test runs.{liveFaults.length ? ` In three live demos on fresh seeds: ${liveCaught} of ${liveFaults.length}.` : ''}</td>
                </tr>
                <tr>
                  <td data-label="Question" className="name-cell">False alarms per hour</td>
                  <td data-label="Answer" className="big-num">{fa ? fa.false_alarms_per_hour_fault_free.toFixed(0) : '–'}<span className="of"> per hour</span></td>
                  <td data-label="What it means">{fa ? `Measured on only ${Math.round(fa.fault_free_hours * 60)} minutes of quiet running, so it is rough. In the calm parts of the runs with problems it was ${fa.false_alarms_per_hour_outside_faults.toFixed(1)} per hour.` : ''}</td>
                </tr>
              </tbody>
            </table>
            <p className="honest"><strong>Honest limits:</strong> one app, {o?.n_test_runs ?? 12} test runs, {inst.length || 15} planted problems. Scores for single problem types are unreliable, and other apps are unproven. <Link to="/examiners/limits">All limits</Link>.</p>
          </>
        )
      }}
    </Async>
  )
}

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
      <h1>Proof</h1>
      <Headline />
      <p className="small" style={{ marginTop: 'var(--space-4)' }}>New: the <Link to="/examiners/hybrids">Hybrid models</Link> page shows ten two-part models (five on the benchmark data, five on a slice of a second data set); sections P20 to P24 below check them.</p>

      <details className="acc all-checks" style={{ marginTop: 'var(--space-6)' }}>
        <summary>Show all the checks behind these numbers (technical)</summary>
      <p className="lede"><Simple>Every number in this app can be checked. We recompute them from the saved data and models. You can run the checks again here, or in a terminal with <code>python scripts/proofs.py</code>.</Simple><Detailed>Numbered proof report P1–P24 produced by <code>scripts/proofs.py</code>; this page renders <code>report_outputs/proofs.json</code> and shows the same text the terminal prints.</Detailed></p>

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
      </details>
    </>
  )
}
