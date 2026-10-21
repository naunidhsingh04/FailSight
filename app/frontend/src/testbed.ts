import { useCallback, useEffect, useRef, useState } from 'react'

export interface TbReason { feature: string; plain_name: string; value: number; normal: number | null; shap: number; sentence: string }
export interface TbWindow {
  type: 'window'; app: string; window_index: number; window_start: number; window_end: number; emitted_at?: number; learning: boolean; learning_progress: number
  risk: number; threshold: number; flagged: boolean; alert: boolean; new_alert: boolean; status: 'learning' | 'healthy' | 'watch' | 'failing'
  reasons: TbReason[]; summary: string | null; robust_z_score: number | null; robust_z_flag: boolean; has_baseline: boolean
  validated: boolean; sources: { api: boolean; logs: boolean; process_metrics: boolean }
  features: Record<string, number | null>; model: { name: string; feature_set: string; alert_rule: string; frozen_at: string }
}
export interface TbApp {
  app: string; anchor: number | null; status: string; idle: boolean; last_event_at: number | null; learning: { done: boolean; progress: number }
  last: TbWindow | null; counters: Record<string, number>; history: { window_index: number; window_end: number; risk: number; flagged: boolean; alert: boolean; learning: boolean; status: string }[]; alerts: TbWindow[]
}
export interface TbIntegration { api_key_required: boolean; apps_with_keys: string[]; webhook_configured: boolean; webhook_host: string | null; last_webhook: { status: unknown } | null }
export interface TbState { integration?: TbIntegration; config: { frozen_at: string; model: string; feature_set: string; n_features: number; threshold: number; learning_period_s: number; alert_rule: string }; apps: Record<string, TbApp> }
export interface TbFault { id: string; type: string; klass: string; kind: string; start: number; until: number; failed: boolean; params: Record<string, unknown> }

export const FAULTS: { id: string; label: string; klass: string; hint: string }[] = [
  { id: 'latency', label: 'Slow responses', klass: 'Performance', hint: 'Every request gets extra waiting time.' },
  { id: 'cpu_burn', label: 'CPU hog', klass: 'Performance', hint: 'A busy loop steals processor time.' },
  { id: 'latency_ramp', label: 'Getting slower and slower', klass: 'Performance', hint: 'Gradual: the delay grows until requests time out.' },
  { id: 'route_500', label: 'One route fails at random', klass: 'Service', hint: 'Some requests to one route get a server error.' },
  { id: 'crash', label: 'The shop crashes', klass: 'Service', hint: 'The process dies and is restarted after a while.' },
  { id: 'pool_exhaustion', label: 'Out of database connections', klass: 'Service', hint: 'Gradual: connections are used up one by one.' },
  { id: 'db_slow', label: 'Slow database queries', klass: 'Database', hint: 'Every query does extra real work.' },
  { id: 'db_lock', label: 'Database lock', klass: 'Database', hint: 'Writes wait for a lock.' },
  { id: 'exception_route', label: 'A bug in one route', klass: 'Code', hint: 'One route raises an unhandled error.' },
  { id: 'leak', label: 'Memory leak', klass: 'Code', hint: 'Gradual: memory grows until the shop fails.' },
]

export function useTestbed(app = 'shop') {
  const [state, setState] = useState<TbState | null>(null)
  const [windows, setWindows] = useState<TbWindow[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const seen = useRef(new Set<string>())

  useEffect(() => {
    let alive = true
    let es: EventSource | null = null
    fetch('/api/testbed/state').then(async (r) => {
      const j = await r.json()
      if (!r.ok) throw new Error(j.error ?? `HTTP ${r.status}`)
      return j as TbState
    }).then((st) => {
      if (!alive) return
      setState(st)
      const w = st.apps[app]
      if (w?.last) for (const h of w.history) seen.current.add(`${w.anchor}|${h.window_index}`)
      fetch(`/api/testbed/windows?app_id=${encodeURIComponent(app)}`).then((r) => r.json()).then((past: TbWindow[]) => {
        if (!alive || !Array.isArray(past)) return
        for (const p of past) seen.current.add(`${p.window_start}|${p.window_index}`)
        setWindows((cur) => [...past, ...cur.filter((c) => !past.some((p) => p.window_start === c.window_start && p.window_index === c.window_index))].slice(-120))
      }).catch(() => undefined)
      es = new EventSource('/api/testbed/stream?replay=0')
      es.onopen = () => alive && setConnected(true)
      es.onerror = () => alive && setConnected(false)
      es.onmessage = (m) => {
        try {
          const ev = JSON.parse(m.data)
          if (ev.type === 'session') { seen.current.clear(); setWindows([]); fetch('/api/testbed/state').then((r) => r.json()).then((s) => alive && setState(s)).catch(() => undefined) }
          if (ev.type !== 'window' || ev.app !== app) return
          const key = `${ev.window_start}|${ev.window_index}`
          if (seen.current.has(key)) return
          seen.current.add(key)
          setWindows((w) => [...w, ev].slice(-120))
          fetch('/api/testbed/state').then((r) => r.json()).then((s) => alive && setState(s)).catch(() => undefined)
        } catch { /* ignore a malformed frame */ }
      }
    }).catch((e) => alive && setError(String(e.message ?? e)))
    return () => { alive = false; es?.close() }
  }, [app])

  return { state, windows, connected, error, app: state?.apps[app] ?? null }
}

export function useFaults() {
  const [active, setActive] = useState<TbFault[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [ready, setReady] = useState(false)               // true once the first answer (even "not reachable") has arrived
  const timer = useRef<number | null>(null)
  const alive = useRef(true)
  const refresh = useCallback(async () => {
    let failed = false
    try {
      const r = await fetch('/api/testbed/faults')
      const j = await r.json()
      if (!r.ok || j.reachable === false) { failed = true; setErr(j.error ?? 'the demo shop is not reachable'); setActive([]) } else { setErr(null); setActive(j.active ?? []) }
    } catch (e: any) { failed = true; setErr(String(e?.message ?? e)) }
    if (alive.current) { setReady(true); timer.current = window.setTimeout(refresh, failed ? 4000 : 2000) }       // one request at a time, slower while the shop is unreachable
  }, [])
  useEffect(() => { alive.current = true; refresh(); return () => { alive.current = false; if (timer.current) window.clearTimeout(timer.current) } }, [refresh])
  const inject = async (type: string, severity: string, duration_s?: number) => {
    const r = await fetch('/api/testbed/fault', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(duration_s ? { type, severity, duration_s } : { type, severity }) })
    const j = await r.json().catch(() => ({}))
    if (!r.ok) setErr(j.error ?? `HTTP ${r.status}`)
    if (timer.current) window.clearTimeout(timer.current)
    refresh()
    return r.ok
  }
  const clear = async () => { await fetch('/api/testbed/clear', { method: 'POST' }); if (timer.current) window.clearTimeout(timer.current); refresh() }
  return { active, err, ready, inject, clear, refresh }
}

export const TESTBED_LIMITS: string[] = [
  'One app: the detector was built and tested on a single small ticket shop.',
  'One machine: the shop, the load generator and the monitor run on one laptop.',
  'Synthetic traffic and injected faults: the users are a script and the faults are ones we wrote.',
  'Small held-out test: 12 runs and 15 faults.',
  'Per-type results are unreliable: most fault types have only one to three test instances.',
  'Transfer to other apps has not been shown: a new app needs its own learning period and its own check.',
]
