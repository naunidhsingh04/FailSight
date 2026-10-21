import type { HistPoint } from '../monitor'
import { clock, percent } from '../monitor'

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
          {p.low_data
            ? <rect x={x(i) - 4} y={y(p.risk) - 4} width="8" height="8" transform={`rotate(45 ${x(i)} ${y(p.risk)})`} fill="var(--bg)" stroke="var(--muted)" strokeWidth="1.6" strokeDasharray="2 1.5"><title>{`${clock(p.window_start)} UTC: not enough data to judge (risk ${percent(p.risk)} not used)`}</title></rect>
            : p.flagged
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
