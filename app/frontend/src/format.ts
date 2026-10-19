export const fmt = (x: number | null | undefined, d = 3): string => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))
export const pct = (x: number | null | undefined, d = 0): string => (x === null || x === undefined || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(d)}%`)
export const int = (x: number | null | undefined): string => (x === null || x === undefined ? '–' : Math.round(x).toLocaleString('en-US'))
export const range = (lo: number | null | undefined, hi: number | null | undefined, d = 2): string => `${fmt(lo, d)}–${fmt(hi, d)}`
export const sci = (x: number): string => (Math.abs(x) >= 1e5 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(2) : x.toFixed(3))
export const FAMILY_LABEL: Record<string, string> = { tree: 'Tree', linear: 'Linear', deep: 'Deep', ensemble: 'Ensemble' }
export const GROUP_LABEL: Record<string, string> = { api: 'API', log: 'Log', metric: 'Metric', system: 'System' }
