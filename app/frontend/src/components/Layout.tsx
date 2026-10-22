import { useEffect, useRef } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { useMode } from '../mode'

/** Everything that used to be a top-level tab now lives under "For examiners". Old addresses still work. */
export const EXAMINER_PAGES: { to: string; label: string; what: string }[] = [
  { to: '/examiners/results', label: 'Results', what: 'The scores of every model on the recorded benchmark data.' },
  { to: '/examiners/hybrids', label: 'Hybrid models', what: 'Ten models made of two learning parts: five on the benchmark data, five on a slice of a second data set.' },
  { to: '/examiners/connect', label: 'Connect', what: 'How to plug another app into FailSight.' },
  { to: '/examiners/testbed', label: 'Testbed', what: 'Our own fictional shop: how the test data was made and the held-out results.' },
  { to: '/examiners/limits', label: 'Limits', what: 'Everything FailSight cannot do or has not shown, in full.' },
  { to: '/examiners/glossary', label: 'Glossary', what: 'Every technical word, in plain language.' },
  { to: '/examiners/replay', label: 'Recorded benchmark data', what: 'Replay of the recorded data. It cannot prove that faults are detected.' },
]
const LEGACY = ['/results', '/connect', '/testbed', '/limits', '/glossary', '/console', '/hybrids']

const LINKS: { to: string; label: string; end?: boolean }[] = [
  { to: '/', label: 'Live monitor', end: true },
  { to: '/how-it-works', label: 'How it works' },
  { to: '/proofs', label: 'Proof' },
]

export default function Layout() {
  const { pathname } = useLocation()
  const { mode, setMode, theme, setTheme } = useMode()
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }   // keep the natural tab order (skip link first) on the initial load
    window.scrollTo(0, 0)
    document.getElementById('main')?.focus({ preventScroll: true })
  }, [pathname])
  const examiner = pathname.startsWith('/examiners') || LEGACY.some((p) => pathname === p)
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <div className="container">
          <NavLink to="/" className="brand" aria-label="FailSight, home">Fail<span>Sight</span></NavLink>
          <nav aria-label="Main">
            <ul className="nav">
              {LINKS.map((l) => <li key={l.to}><NavLink to={l.to} end={l.end}>{l.label}</NavLink></li>)}
              <li><Link to="/examiners" aria-current={examiner ? 'page' : undefined}>For examiners</Link></li>
            </ul>
          </nav>
          <div className="mode-toggle">
            <div className="segmented" role="group" aria-label="Amount of detail">
              <button type="button" aria-pressed={mode === 'simple'} onClick={() => setMode('simple')}>Simple</button>
              <button type="button" aria-pressed={mode === 'detailed'} onClick={() => setMode('detailed')}>Detailed</button>
            </div>
            <button type="button" className="theme-btn" aria-pressed={theme === 'dark'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? 'Light colours' : 'Dark colours'}</button>
          </div>
        </div>
      </header>
      <main id="main" tabIndex={-1} style={{ outline: 'none' }}>
        <div className="container"><Outlet /></div>
      </main>
      <footer className="site-footer">
        <div className="container">
          <p><strong>FailSight is a student research project, not a product.</strong> The monitor watches a fictional shop on this computer, with pretend visitors.</p>
          <p>Every number comes from saved result files or is worked out again from them. Nothing is trained in this app.</p>
        </div>
      </footer>
    </>
  )
}
