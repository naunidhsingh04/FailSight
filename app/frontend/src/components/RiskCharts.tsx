import type { HistPoint, Status } from '../monitor'
import { clock, percent } from '../monitor'

const COLOR: Record<Status, string> = { idle: 'var(--ink-3)', healthy: 'var(--ok)', watch: 'var(--watch)', failing: 'var(--fail)' }

/** Half-circle risk gauge, 0-100%, with a tick at the alert line. */
export function Gauge({ risk, threshold, status, label }: { risk: number | null; threshold: number | null; status: Status; label: string }) {
  const R = 78, CX = 100, CY = 92
  const len = Math.PI * R
  const r = risk ?? 0
  const t = threshold ?? null
  const ang = (v: number) => Math.PI * v
  const tx = (v: number, rad: number) => CX - rad * Math.cos(ang(v))
  const ty = (v: number, rad: number) => CY - rad * Math.sin(ang(v))
  return (
    <svg viewBox="0 0 200 112" className="chart" role="img" aria-label={`${label}: failure risk ${risk === null ? 'not available yet' : percent(risk)}${t !== null ? `, alert line at ${percent(t)}` : ''}`}>
      <path d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`} fill="none" stroke="var(--surface-2)" strokeWidth="16" strokeLinecap="butt" />
      <path d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`} fill="none" stroke="var(--border)" strokeWidth="16" strokeLinecap="butt" opacity="0.6" />
      {risk !== null && <path d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`} fill="none" stroke={COLOR[status]} strokeWidth="16" strokeLinecap="butt" strokeDasharray={`${Math.max(len * r, 1)} ${len}`} />}
      {t !== null && (
        <g>
          <line x1={tx(t, R - 13)} y1={ty(t, R - 13)} x2={tx(t, R + 13)} y2={ty(t, R + 13)} stroke="var(--text)" strokeWidth="3" />
          <text x={tx(t, R - 22)} y={ty(t, R - 22) + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--text)' }}>alert line</text>
        </g>
      )}
      <text x={CX} y={CY - 8} textAnchor="middle" fontSize="26" fontWeight="650" style={{ fill: 'var(--text)', fontFamily: 'var(--font-serif)' }}>{risk === null ? '–' : percent(risk)}</text>
      <text x={CX} y={CY + 10} textAnchor="middle" fontSize="10" style={{ fill: 'var(--muted)' }}>failure risk</text>
    </svg>
  )
}

/** Risk per window with the alert line; the strip underneath marks windows the recording says were failures. */
export function RiskTimeline({ points, threshold, ariaLabel }: { points: HistPoint[]; threshold: number | null; ariaLabel: string }) {
  const W = 640, H = 236, L = 44, R = 12, T = 14, B = 70
  const n = Math.max(points.length, 2)
  const x = (i: number) => L + (i / (n - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - v) * (H - T - B - 12)
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.risk).toFixed(1)}`).join(' ')
  const stripY = H - B + 4
  const bw = Math.max((W - L - R) / n, 2)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={ariaLabel}>
      <defs>
        <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="var(--text)" strokeWidth="2.5" /></pattern>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}><line className="grid-line" x1={L} x2={W - R} y1={y(v)} y2={y(v)} /><text x={L - 6} y={y(v) + 4} textAnchor="end">{Math.round(v * 100)}%</text></g>
      ))}
      {threshold !== null && (
        <g>
          <line x1={L} x2={W - R} y1={y(threshold)} y2={y(threshold)} stroke="var(--fail)" strokeWidth="2" strokeDasharray="7 4" />
          <text x={W - R - 2} y={y(threshold) - 5} textAnchor="end" style={{ fill: 'var(--fail)', fontWeight: 700 }}>alert line {percent(threshold)}</text>
        </g>
      )}
      {points.length > 1 && <path d={line} fill="none" stroke="var(--ink-2)" strokeWidth="1.8" />}
      {points.map((p, i) => (
        <g key={`${p.window_start}-${i}`}>
          {p.truth === 'Anomaly' && <rect x={x(i) - bw / 2} y={stripY} width={bw} height={10} fill="url(#hatch)" stroke="var(--text)" strokeWidth="0.6" />}
          {p.truth === 'Normal' && <rect x={x(i) - bw / 2} y={stripY} width={bw} height={10} fill="var(--surface-2)" stroke="var(--border-strong)" strokeWidth="0.6" />}
          {p.flagged
            ? <rect x={x(i) - 5} y={y(p.risk) - 5} width="10" height="10" fill="var(--fail)" stroke="var(--bg)" strokeWidth="1.5"><title>{`${clock(p.window_start)} UTC: alert, risk ${percent(p.risk)}`}</title></rect>
            : <circle cx={x(i)} cy={y(p.risk)} r={p.warmup ? 2.5 : 4} fill={p.warmup ? 'var(--ink-3)' : 'var(--bg)'} stroke="var(--ink-1)" strokeWidth="1.6"><title>{`${clock(p.window_start)} UTC: risk ${percent(p.risk)}${p.warmup ? ' (warm-up)' : ''}`}</title></circle>}
        </g>
      ))}
      {points.length > 0 && <>
        <text x={L} y={H - 24}>{clock(points[0].window_start)} UTC</text>
        <text x={W - R} y={H - 24} textAnchor="end">{clock(points[points.length - 1].window_start)} UTC</text>
      </>}
      <text x={L} y={H - 4} style={{ fill: 'var(--muted)', fontSize: 11 }}>hatched strip = the recording says a failure was happening</text>
    </svg>
  )
}
