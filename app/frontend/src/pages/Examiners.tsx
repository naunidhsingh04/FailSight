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
      <p className="lede">The detail behind the demo: the full results, the limits, and how the test data was made. Start with <Link to="/proofs">Proof</Link> if you only have a minute.</p>
      <ul className="link-grid">
        {EXAMINER_PAGES.map((p) => (
          <li key={p.to}><Link className="link-card" to={p.to}><h2>{p.label}</h2><p>{p.what}</p><span className="go">Open</span></Link></li>
        ))}
      </ul>
    </>
  )
}
