import { useEffect, useRef, useState } from 'react'
import type { Level } from '../plain'
import { clock12 } from '../plain'
import type { Status } from '../monitor'
import { StatusIcon } from './Icons'

const ICON: Record<Level, Status> = { idle: 'idle', learning: 'idle', ok: 'healthy', watch: 'watch', problem: 'failing' }

/** Big status sentence: colour, icon and words together. */
export function StatusSentence({ level, text }: { level: Level; text: string }) {
  return (
    <div className={`answer ${level}`} role="status" aria-live="polite" aria-atomic="true">
      <StatusIcon status={ICON[level]} />
      <span>{text}</span>
    </div>
  )
}

/** Three boxes in a row, OK / Watch / Problem; the current one is filled. */
export function StatusBar({ level }: { level: Level }) {
  const cells: { id: Level; label: string; status: Status }[] = [
    { id: 'ok', label: 'OK', status: 'healthy' },
    { id: 'watch', label: 'Watch', status: 'watch' },
    { id: 'problem', label: 'Problem', status: 'failing' },
  ]
  return (
    <ol className="statusbar" aria-label="Status scale: OK, Watch, Problem">
      {cells.map((c) => (
        <li key={c.id} className={`${c.id} ${level === c.id ? 'on' : ''}`} aria-current={level === c.id ? 'true' : undefined}>
          <StatusIcon status={c.status} /> <span>{c.label}</span>
        </li>
      ))}
    </ol>
  )
}

function useWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(600)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const set = () => setW(Math.max(260, Math.floor(el.getBoundingClientRect().width)))
    set()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(set)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

export interface Pt { t: number; ms: number }

/** Tiny line chart: response time only. Marks the moment you broke something and the moment FailSight noticed. */
export function ResponseChart({ points, brokeAt, noticedAt }: { points: Pt[]; brokeAt: number | null; noticedAt: number | null }) {
  const [ref, W] = useWidth()
  const H = 200, L = 46, R = 12, T = 44, B = 28
  const pts = points.slice(-30)
  if (pts.length < 2) return null                 // the page shows a sentence saying what to do instead of an empty chart
  const t0 = pts[0].t, t1 = Math.max(pts[pts.length - 1].t, brokeAt ?? 0, noticedAt ?? 0)
  const max = Math.max(50, ...pts.map((p) => p.ms)) * 1.1
  const x = (t: number) => L + ((t - t0) / Math.max(1, t1 - t0)) * (W - L - R)
  const y = (v: number) => T + (1 - v / max) * (H - T - B)
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)} ${y(p.ms).toFixed(1)}`).join(' ')
  const inRange = (t: number | null) => t != null && t >= t0 && t <= t1
  const lab = (t: number, text: string, row: number) => {
    const px = x(t), right = px > W - 150
    return <text x={px + (right ? -6 : 6)} y={14 + row * 18} textAnchor={right ? 'end' : 'start'} className="mark">{text}</text>
  }
  const last = pts[pts.length - 1].ms
  const ticks = [0, max / 2, max].map((v) => Math.round(v / 10) * 10)
  return (
    <div ref={ref} className="resp-wrap">
      <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Response time of the last ${pts.length} checks. Latest ${Math.round(last)} milliseconds.${brokeAt ? ` You broke the shop at ${clock12(brokeAt)}.` : ''}${noticedAt ? ` FailSight noticed at ${clock12(noticedAt)}.` : ''}`}>
        {ticks.map((v) => <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid-line" /><text x={L - 6} y={y(v) + 4} textAnchor="end" className="dim">{v}</text></g>)}
        <text x={L} y={H - 6} className="dim">{clock12(t0)}</text>
        <text x={W - R} y={H - 6} textAnchor="end" className="dim">{clock12(t1)}</text>
        <text x={L - 6} y={T - 10} textAnchor="end" className="dim">ms</text>
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeLinejoin="round" />
        <circle cx={x(pts[pts.length - 1].t)} cy={y(last)} r={4} fill="var(--accent)" />
        {inRange(brokeAt) && <g><line x1={x(brokeAt as number)} x2={x(brokeAt as number)} y1={T - 8} y2={H - B} stroke="var(--fail)" strokeWidth={2} strokeDasharray="5 4" />{lab(brokeAt as number, 'You broke it', 0)}</g>}
        {inRange(noticedAt) && <g><line x1={x(noticedAt as number)} x2={x(noticedAt as number)} y1={T - 8} y2={H - B} stroke="var(--text)" strokeWidth={2} />{lab(noticedAt as number, 'FailSight noticed', 1)}</g>}
      </svg>
    </div>
  )
}
