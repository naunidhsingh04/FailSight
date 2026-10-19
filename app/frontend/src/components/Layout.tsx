import { useEffect, useRef } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useMode } from '../mode'

const LINKS: { to: string; label: string; end?: boolean }[] = [
  { to: '/', label: 'Console', end: true },
  { to: '/how-it-works', label: 'How it works' },
  { to: '/proofs', label: 'Proofs' },
  { to: '/results', label: 'Results' },
  { to: '/limits', label: 'Limits' },
  { to: '/glossary', label: 'Glossary' },
]

export default function Layout() {
  const { pathname } = useLocation()
  const { mode, setMode } = useMode()
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }   // keep the natural tab order (skip link first) on the initial load
    window.scrollTo(0, 0)
    document.getElementById('main')?.focus({ preventScroll: true })
  }, [pathname])
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <div className="container">
          <NavLink to="/" className="brand" aria-label="FailSight, home">Fail<span>Sight</span></NavLink>
          <nav aria-label="Main">
            <ul className="nav">
              {LINKS.map((l) => <li key={l.to}><NavLink to={l.to} end={l.end}>{l.label}</NavLink></li>)}
            </ul>
          </nav>
          <div className="mode-toggle" role="group" aria-label="Level of detail">
            <span id="mode-l">View:</span>
            <div className="segmented" role="group" aria-labelledby="mode-l">
              <button type="button" aria-pressed={mode === 'simple'} onClick={() => setMode('simple')}>Simple</button>
              <button type="button" aria-pressed={mode === 'detailed'} onClick={() => setMode('detailed')}>Detailed</button>
            </div>
          </div>
        </div>
      </header>
      <main id="main" tabIndex={-1} style={{ outline: 'none' }}>
        <div className="container"><Outlet /></div>
      </main>
      <footer className="site-footer">
        <div className="container">
          <p><strong>Replay of recorded AnoMod data.</strong> FailSight is a research prototype, not a monitoring product. The demonstration is not connected to a live system.</p>
          <p>Every figure in this app is read from result files, saved models or recomputation; nothing is retrained here.</p>
        </div>
      </footer>
    </>
  )
}
