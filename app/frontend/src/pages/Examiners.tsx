import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { EXAMINER_PAGES } from '../components/Layout'
import { usePageTitle } from '../usePageTitle'

/** Wraps every sub-page of "For examiners" with the same small menu, so you can always see where you are. */
export function ExaminerLayout() {
  const { pathname } = useLocation()
  const hub = pathname === '/examiners' || pathname === '/examiners/'
  return (
    <>
      {!hub && (
        <nav aria-label="For examiners" className="subnav">
          <Link to="/examiners">For examiners</Link>
          <ul>
            {EXAMINER_PAGES.map((p) => <li key={p.to}><NavLink to={p.to}>{p.label}</NavLink></li>)}
          </ul>
        </nav>
      )}
      <Outlet />
    </>
  )
}

export default function Examiners() {
  usePageTitle('For examiners')
  return (
    <>
      <h1>For examiners</h1>
      <p className="lede">The detail behind the live monitor: the full results, the limits, and how the test data was made. Start with <Link to="/proofs">Proof</Link> if you only have a minute.</p>
      <ul className="link-grid">
        {EXAMINER_PAGES.map((p) => (
          <li key={p.to}><Link className="link-card" to={p.to}><h2>{p.label}</h2><p>{p.what}</p><span className="go">Open</span></Link></li>
        ))}
      </ul>
      <div className="colab-box" data-testid="colab-box">
        <h2 style={{ marginTop: 0, fontSize: '1.15rem' }}>Run it yourself in Colab</h2>
        <p style={{ margin: '0 0 var(--space-2)' }}>A notebook walks through the whole workflow step by step, with the proofs, in your browser. No installation. It takes about 20 to 30 minutes on a fresh Google Colab session.</p>
        <p style={{ margin: 0 }}><a href="https://colab.research.google.com/github/naunidhsingh04/FailSight/blob/main/notebooks/FailSight_Workflow.ipynb" target="_blank" rel="noreferrer">Open the notebook in Google Colab</a> (Runtime &rarr; Run all). Steps are in <code>docs/COLAB.md</code>.</p>
      </div>
    </>
  )
}
