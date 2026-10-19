import { useEffect, useRef } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

const LINKS: { to: string; label: string; end?: boolean }[] = [
  { to: '/', label: 'Overview', end: true },
  { to: '/predict', label: 'Predict' },
  { to: '/models', label: 'Models' },
  { to: '/explain', label: 'Explain' },
  { to: '/data', label: 'Data & method' },
  { to: '/limits', label: 'Limits' },
]

export default function Layout() {
  const { pathname } = useLocation()
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
              {LINKS.map((l) => (
                <li key={l.to}><NavLink to={l.to} end={l.end}>{l.label}</NavLink></li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <main id="main" tabIndex={-1} style={{ outline: 'none' }}>
        <div className="container"><Outlet /></div>
      </main>
      <footer className="site-footer">
        <div className="container">
          <p>FailSight: AI-based API failure prediction on AnoMod with explainable AI (SHAP/LIME). Research prototype, not a monitoring product.</p>
          <p>Every figure in this app is read from result files produced by the training pipeline; nothing is retrained or tuned here.</p>
        </div>
      </footer>
    </>
  )
}
