import { fmt } from '../format'

/* ---------------------------------------------------------------- range plot: point estimate with 95% CI, one row per item */
export interface RangeItem { label: string; point: number | null; low: number | null; high: number | null; emphasis?: boolean }

export function RangePlot({ items, title, domain = [0, 1], ariaLabel }: { items: RangeItem[]; title?: string; domain?: [number, number]; ariaLabel: string }) {
  const rowH = 26, left = 150, right = 16, top = 22, W = 720
  const H = top + items.length * rowH + 28
  const x = (v: number) => left + ((v - domain[0]) / (domain[1] - domain[0])) * (W - left - right)
  const ticks = [0, 0.25, 0.5, 0.75, 1].filter((t) => t >= domain[0] && t <= domain[1])
  return (
    <figure style={{ margin: 0 }}>
      {title && <figcaption className="small muted" style={{ marginBottom: 4 }}>{title}</figcaption>}
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 24} />
            <text x={x(t)} y={H - 8} textAnchor="middle">{t}</text>
          </g>
        ))}
        {items.map((it, i) => {
          const y = top + i * rowH + rowH / 2
          return (
            <g key={it.label}>
              <text x={left - 8} y={y + 4} textAnchor="end" fontWeight={it.emphasis ? 700 : 400}>{it.label}</text>
              {it.low !== null && it.high !== null && (
                <>
                  <line x1={x(it.low)} x2={x(it.high)} y1={y} y2={y} stroke="var(--ink-2)" strokeWidth={2} />
                  <line x1={x(it.low)} x2={x(it.low)} y1={y - 5} y2={y + 5} stroke="var(--ink-2)" strokeWidth={2} />
                  <line x1={x(it.high)} x2={x(it.high)} y1={y - 5} y2={y + 5} stroke="var(--ink-2)" strokeWidth={2} />
                </>
              )}
              {it.point !== null && <circle cx={x(it.point)} cy={y} r={4.5} fill="var(--ink-1)" stroke="var(--bg)" strokeWidth={1.5}><title>{`${it.label}: ${fmt(it.point)} (95% CI ${fmt(it.low)} to ${fmt(it.high)})`}</title></circle>}
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

/* ---------------------------------------------------------------- line chart (ROC / PR): grayscale, distinguished by dash pattern */
export interface Series { name: string; points: number[][]; dash?: string; width?: number }
const DASHES = ['', '6 3', '2 3', '10 3 2 3', '1 4']

export function LineChart({ series, xLabel, yLabel, ariaLabel, chance }: { series: Series[]; xLabel: string; yLabel: string; ariaLabel: string; chance?: number[][] }) {
  const W = 420, H = 360, l = 48, r = 12, t = 12, b = 44
  const x = (v: number) => l + v * (W - l - r)
  const y = (v: number) => H - b - v * (H - t - b)
  const path = (pts: number[][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ')
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="grid-line" x1={x(v)} x2={x(v)} y1={t} y2={H - b} />
            <line className="grid-line" x1={l} x2={W - r} y1={y(v)} y2={y(v)} />
            <text x={x(v)} y={H - b + 16} textAnchor="middle">{v}</text>
            <text x={l - 8} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        <rect x={l} y={t} width={W - l - r} height={H - t - b} fill="none" stroke="var(--border-strong)" />
        {chance && <path d={path(chance)} fill="none" stroke="var(--ink-3)" strokeWidth={1} strokeDasharray="3 3" />}
        {series.map((s, i) => (
          <path key={s.name} d={path(s.points)} fill="none" stroke={i === 0 ? 'var(--ink-1)' : 'var(--ink-2)'} strokeWidth={s.width ?? (i === 0 ? 2.4 : 1.6)} strokeDasharray={s.dash ?? DASHES[i % DASHES.length]} />
        ))}
        <text x={(l + W - r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(12 ${(t + H - b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
      </svg>
      <ul className="chart-legend">
        {series.map((s, i) => (
          <li key={s.name}>
            <svg width="28" height="10" aria-hidden="true"><line x1="0" x2="28" y1="5" y2="5" stroke={i === 0 ? 'var(--ink-1)' : 'var(--ink-2)'} strokeWidth={s.width ?? (i === 0 ? 2.4 : 1.6)} strokeDasharray={s.dash ?? DASHES[i % DASHES.length]} /></svg>
            {s.name}
          </li>
        ))}
      </ul>
    </figure>
  )
}

/* ---------------------------------------------------------------- confusion matrix (counts, grayscale shading by row share) */
export function ConfusionMatrix({ labels, matrix, ariaLabel }: { labels: string[]; matrix: number[][]; ariaLabel: string }) {
  const n = labels.length
  const cell = n > 2 ? 64 : 92
  const left = 92, top = 44
  const W = left + n * cell + 8, H = top + n * cell + 34
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} style={{ maxWidth: W * 1.5 }}>
      <text x={left + (n * cell) / 2} y={14} textAnchor="middle" fontWeight={600}>predicted</text>
      {labels.map((c, j) => <text key={c} x={left + j * cell + cell / 2} y={top - 8} textAnchor="middle">{c.length > 9 ? c.slice(0, 8) + '.' : c}</text>)}
      {matrix.map((row, i) => {
        const sum = row.reduce((a, b) => a + b, 0) || 1
        return (
          <g key={i}>
            <text x={left - 8} y={top + i * cell + cell / 2 + 4} textAnchor="end">{labels[i]}</text>
            {row.map((v, j) => {
              const share = v / sum
              const dark = share > 0.5
              return (
                <g key={j}>
                  <rect x={left + j * cell} y={top + i * cell} width={cell - 2} height={cell - 2} fill={`rgba(${dark ? '27,26,23' : '27,26,23'},${(0.08 + share * 0.85).toFixed(2)})`} stroke="var(--border-strong)" />
                  <text x={left + j * cell + (cell - 2) / 2} y={top + i * cell + cell / 2 + 4} textAnchor="middle" fontWeight={700} fill={dark ? '#fff' : 'var(--text)'} style={{ fill: dark ? '#fff' : undefined }}>{v}</text>
                </g>
              )
            })}
          </g>
        )
      })}
      <text x={12} y={top + (n * cell) / 2} textAnchor="middle" transform={`rotate(-90 12 ${top + (n * cell) / 2})`} fontWeight={600}>true</text>
    </svg>
  )
}

/* ---------------------------------------------------------------- probability bars */
export function ProbBars({ probs, predicted }: { probs: Record<string, number>; predicted: string }) {
  const entries = Object.entries(probs)
  return (
    <div role="group" aria-label="Predicted probability per class" className="stack" style={{ gap: 8 }}>
      {entries.map(([name, p]) => (
        <div key={name} style={{ display: 'grid', gridTemplateColumns: 'minmax(5.5rem, 8rem) 1fr 4.5rem', alignItems: 'center', gap: 12 }}>
          <span style={{ fontWeight: name === predicted ? 700 : 400 }}>{name}{name === predicted ? ' ◀' : ''}</span>
          <span style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', height: 18, borderRadius: 3, overflow: 'hidden', display: 'block' }} role="img" aria-label={`${name} ${(p * 100).toFixed(1)} percent`}>
            <span style={{ display: 'block', height: '100%', width: `${Math.max(p * 100, 0.4)}%`, background: name === predicted ? 'var(--ink-1)' : 'var(--ink-3)' }} />
          </span>
          <span className="num-cell" style={{ textAlign: 'right' }}>{(p * 100).toFixed(p < 0.001 ? 3 : 1)}%</span>
        </div>
      ))}
    </div>
  )
}

/* ---------------------------------------------------------------- feature contribution bars (SHAP / LIME), group shown by pattern + text, value on hover/focus */
export interface FeatureBar { feature: string; group: string; value: number; raw: number | null; pre: number }

const PATTERN: Record<string, string> = { api: 'var(--ink-1)', log: 'var(--ink-2)', metric: 'var(--ink-3)', system: 'var(--ink-4)' }

export function FeatureBars({ items, valueLabel }: { items: FeatureBar[]; valueLabel: string }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9)
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0, maxWidth: 'none' }} aria-label={`${valueLabel} per feature`}>
      {items.map((it) => {
        const w = (Math.abs(it.value) / max) * 50
        const pos = it.value >= 0
        const tip = `${it.feature}\ngroup: ${it.group}\n${valueLabel}: ${it.value.toFixed(4)}\nscaled value: ${it.pre.toFixed(3)}\nraw value: ${it.raw === null ? 'missing' : it.raw}`
        return (
          <li key={it.feature} tabIndex={0} title={tip} aria-label={tip.replace(/\n/g, ', ')} className="fbar">
            <span className="fbar-name">{it.feature.replace(/^m_/, '')}<span className="muted small"> {it.group}</span></span>
            <span className="fbar-track" aria-hidden="true">
              <span className="fbar-zero" />
              <span className="fbar-fill" style={{ [pos ? 'left' : 'right']: '50%', width: `${w}%`, background: PATTERN[it.group] ?? 'var(--ink-2)', opacity: pos ? 1 : 0.55, borderStyle: pos ? 'solid' : 'dashed' }} />
            </span>
            <span className="num-cell fbar-val">{it.value >= 0 ? '+' : ''}{it.value.toFixed(3)}</span>
            <span className="fbar-tip" role="tooltip">
              <strong>{it.feature}</strong><br />group {it.group} &middot; {valueLabel} {it.value.toFixed(4)}<br />scaled {it.pre.toFixed(3)} &middot; raw {it.raw === null ? 'missing' : it.raw}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
