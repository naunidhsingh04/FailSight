import { useCallback, useEffect, useRef, useState } from 'react'
import { getJson, postJson } from './api'

export type Sys = 'SN' | 'TT'
export type Status = 'idle' | 'healthy' | 'watch' | 'failing' | 'nodata'
export const SYSTEMS: Sys[] = ['SN', 'TT']
export const SYSTEM_NAME: Record<Sys, string> = { SN: 'Social Network app', TT: 'Train Ticket app' }

export interface Counters { windows: number; alerts: number; correct_alerts: number; false_alarms: number; missed: number; correct_quiet: number; labelled: number; late_dropped: number; before_start: number; events_api: number; events_logs: number; events_metrics: number; rejected: number; low_data: number; low_data_would_alert: number }
export interface TopFeature { feature: string; group: string; shap: number; scaled_value: number; raw_value: number | null; normal_value: number | null }
export interface Reason { feature: string; plain_name: string; value: number | null; normal: number | null; value_text: string; normal_text: string; direction: string; sentence: string }
export interface Truth { label_binary: 'Normal' | 'Anomaly'; label_level: string | null; scenario: string | null; known_from: string }
export interface WinEvent {
  type: 'window'; system: Sys; session: string; window_index: number; window_start: number; window_end: number; model: string; label_version: string
  risk: number; verdict: string; flagged: boolean; low_data: boolean; model_flagged: boolean; status: Status; decision_rule: { type: string; value: number | null }
  top_features: TopFeature[]; reasons: Reason[]; summary: string
  missing: { fraction: number; tier: 'normal' | 'degraded' | 'unreliable'; expected: number; missing: number; by_group: Record<string, { expected: number; missing: number }> }
  warmup: boolean; scored: boolean; truth: Truth | null; counters: Counters; n_features_built: number; emitted_at: number
}
export interface HistPoint { window_index: number; window_start: number; risk: number; flagged: boolean; truth: string | null; warmup?: boolean; low_data?: boolean }
export interface SysState { status: Status; risk: number | null; counters: Counters; last: WinEvent | null; history: HistPoint[]; session: string | null; meta: Record<string, unknown> }
export interface MonitorState { config: { model: string; label_version: string }; models: string[]; systems: Record<Sys, SysState>; alerts: WinEvent[] }
export interface ReplayStatus { running: boolean; paused: boolean; speed: number | null; info: { scenario?: string; speed?: number; system?: string; blocks?: number }; log: string[] }

const ZERO: Counters = { windows: 0, alerts: 0, correct_alerts: 0, false_alarms: 0, missed: 0, correct_quiet: 0, labelled: 0, late_dropped: 0, before_start: 0, events_api: 0, events_logs: 0, events_metrics: 0, rejected: 0, low_data: 0, low_data_would_alert: 0 }
const empty = (): SysState => ({ status: 'idle', risk: null, counters: { ...ZERO }, last: null, history: [], session: null, meta: {} })
const KEEP = 60

export function useMonitor() {
  const [systems, setSystems] = useState<Record<Sys, SysState>>({ SN: empty(), TT: empty() })
  const [alerts, setAlerts] = useState<WinEvent[]>([])
  const [config, setConfig] = useState<{ model: string; label_version: string }>({ model: 'XGB', label_version: 'label_binary' })
  const [models, setModels] = useState<string[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const seen = useRef(new Set<string>())

  const apply = useCallback((ev: any) => {
    if (ev.type === 'window') {
      const w = ev as WinEvent
      setSystems((prev) => {
        const s = prev[w.system]
        const key = `${w.session}|${w.window_index}`
        if (seen.current.has(w.system + key)) return prev
        seen.current.add(w.system + key)
        const point: HistPoint = { window_index: w.window_index, window_start: w.window_start, risk: w.risk, flagged: w.flagged, truth: w.truth?.label_binary ?? null, warmup: w.warmup, low_data: w.low_data }
        return { ...prev, [w.system]: { ...s, status: w.status, risk: w.risk, counters: w.counters, last: w, session: w.session, history: [...s.history, point].slice(-KEEP) } }
      })
      if (w.flagged) setAlerts((a) => (a.some((x) => x.system === w.system && x.session === w.session && x.window_index === w.window_index) ? a : [w, ...a].slice(0, 100)))
    } else if (ev.type === 'session') {
      setSystems((prev) => {
        const s = prev[ev.system as Sys]
        // next block of the same replay keeps the running tally and the timeline; a new replay starts clean
        return { ...prev, [ev.system as Sys]: ev.keep_counters ? { ...s, session: ev.session, meta: ev.meta ?? {} } : { ...empty(), session: ev.session, meta: ev.meta ?? {} } }
      })
    } else if (ev.type === 'config') {
      setConfig(ev.config)
    }
  }, [])

  useEffect(() => {
    let es: EventSource | null = null
    let alive = true
    fetch('/api/monitor/state')
      .then((r) => r.json())
      .then((st: MonitorState) => {
        if (!alive) return
        setSystems({ SN: { ...empty(), ...st.systems.SN }, TT: { ...empty(), ...st.systems.TT } })
        setAlerts([...st.alerts].reverse())
        setConfig(st.config)
        setModels(st.models)
        for (const s of SYSTEMS) for (const h of st.systems[s].history ?? []) seen.current.add(`${s}${st.systems[s].session}|${h.window_index}`)
      })
      .catch((e) => alive && setError(String(e.message ?? e)))
      .finally(() => {
        if (!alive) return
        es = new EventSource('/api/stream?replay=0')
        es.onopen = () => alive && setConnected(true)
        es.onerror = () => alive && setConnected(false)
        es.onmessage = (m) => {
          try { apply(JSON.parse(m.data)) } catch { /* ignore a malformed frame */ }
        }
      })
    return () => {
      alive = false
      es?.close()
    }
  }, [apply])

  const reset = useCallback(() => { seen.current.clear(); setAlerts([]); setSystems({ SN: empty(), TT: empty() }) }, [])
  return { systems, alerts, config, models, connected, error, reset, setConfig }
}

export function useReplay() {
  const [status, setStatus] = useState<ReplayStatus>({ running: false, paused: false, speed: null, info: {}, log: [] })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const refresh = useCallback(async () => {
    try { setStatus(await (await fetch('/api/replay/status')).json()) } catch { /* offline */ }
  }, [])
  useEffect(() => {
    refresh()
    const id = window.setInterval(refresh, 1500)
    return () => window.clearInterval(id)
  }, [refresh])
  const wrap = async (f: () => Promise<unknown>) => {
    setBusy(true); setErr(null)
    try { await f() } catch (e: any) { setErr(e?.message ?? String(e)) } finally { setBusy(false); refresh() }
  }
  return {
    status, busy, err,
    start: (scenario: string, speed: number, blocks = 2) => wrap(() => postJson('/api/replay/start', { scenario, speed, system: 'both', blocks })),
    pause: (paused: boolean) => wrap(() => postJson('/api/replay/control', { paused })),
    speed: (speed: number) => wrap(() => postJson('/api/replay/control', { speed })),
    stop: () => wrap(() => postJson('/api/replay/control', { stop: true })),
  }
}

export async function setModel(model: string) {
  return fetch('/api/monitor/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, label_version: 'label_binary' }) }).then(async (r) => {
    if (!r.ok) throw new Error((await r.json()).error ?? 'could not change the model')
    return r.json()
  })
}
export { getJson }

export const clock = (ts: number) => new Date(ts * 1000).toISOString().slice(11, 19)
export const percent = (x: number | null | undefined, d = 0) => (x === null || x === undefined ? '–' : `${(x * 100).toFixed(d)}%`)
