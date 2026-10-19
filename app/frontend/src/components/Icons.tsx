import type { Status } from '../monitor'

/** Status icons: shape differs per status so colour is never the only cue. */
export function StatusIcon({ status }: { status: Status }) {
  const common = { className: 'icon', viewBox: '0 0 32 32', 'aria-hidden': true as const, fill: 'none', stroke: 'currentColor', strokeWidth: 3, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  if (status === 'healthy') return <svg {...common}><circle cx="16" cy="16" r="13" /><path d="M9 16.5l5 5 9-10" /></svg>
  if (status === 'watch') return <svg {...common}><path d="M16 3.5L29 27H3z" /><path d="M16 12v7M16 23.2v.1" /></svg>
  if (status === 'failing') return <svg {...common}><circle cx="16" cy="16" r="13" /><path d="M10.5 10.5l11 11M21.5 10.5l-11 11" /></svg>
  if (status === 'nodata') return <svg {...common}><circle cx="16" cy="16" r="13" /><path d="M12.3 12.6a3.8 3.8 0 1 1 5.4 3.4c-1.2.7-1.7 1.4-1.7 2.6M16 23v.1" /></svg>
  return <svg {...common}><circle cx="16" cy="16" r="13" strokeDasharray="4 4" /></svg>
}

export function StepIcon({ n }: { n: 1 | 2 | 3 | 4 }) {
  const c = { width: 40, height: 40, viewBox: '0 0 40 40', 'aria-hidden': true as const, fill: 'none', stroke: 'currentColor', strokeWidth: 2.4, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  if (n === 1) return <svg {...c}><path d="M3 20s6-11 17-11 17 11 17 11-6 11-17 11S3 20 3 20z" /><circle cx="20" cy="20" r="5" /></svg>          // watch
  if (n === 2) return <svg {...c}><path d="M5 33V20M13 33V9M21 33V15M29 33V24M37 33H3" /></svg>                                                  // measure
  if (n === 3) return <svg {...c}><path d="M4 28a16 16 0 0 1 32 0" /><path d="M20 28l8-11" /><circle cx="20" cy="28" r="2.2" /></svg>          // risk gauge
  return <svg {...c}><path d="M6 8h28v18H17l-7 6v-6H6z" /><path d="M12 14h16M12 19h10" /></svg>                                                   // explain
}
