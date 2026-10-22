import type { TbReason, TbWindow } from './testbed'

export type Level = 'idle' | 'learning' | 'ok' | 'watch' | 'problem'

export interface Plain { headline: string; reason: string }

const times = (value: number, normal: number | null) => {
  if (normal == null || normal <= 0) return null
  const r = value / normal
  return r >= 1.5 ? Math.round(r) : null
}

/** Turn the strongest reading behind a window into a short plain sentence ("Response time went up 4 times"). Never shows a feature code. */
export function plainReason(r: TbReason | undefined): Plain {
  if (!r) return { headline: 'something looks wrong', reason: 'The shop behaves differently from how it did while FailSight was learning.' }
  const f = r.feature.replace(/^d_/, '')                       // "d_" marks a change in a reading
  const x = times(r.value, r.normal)
  const more = x ? ` (${x} times more than usual)` : ''
  if (f.startsWith('api_latency')) return { headline: 'pages are getting slow', reason: x && r.normal ? `Response time went up ${x} times (from ${Math.round(r.normal)} to ${Math.round(r.value)} milliseconds).` : 'Pages take longer to answer than usual.' }
  if (f === 'api_5xx_count' || f === 'api_error_rate' || f.startsWith('log_error') || f === 'api_4xx_count') return { headline: 'pages are showing errors', reason: `More visitors than usual are getting an error instead of a page${more}.` }
  if (f.includes('db_')) return { headline: 'the database is slow', reason: 'The shop waits much longer than usual for its database.' }
  if (f.includes('cpu')) return { headline: 'the server is working too hard', reason: `The server is using much more processor power than usual${more}.` }
  if (f.includes('rss')) return { headline: 'the server is using too much memory', reason: 'The shop is using much more memory than usual.' }
  if (f.includes('threads')) return { headline: 'the server is busier than usual', reason: 'The shop is running more things at the same time than usual.' }
  if (f.includes('event_loop') || f.includes('inflight')) return { headline: 'the server is slow to react', reason: 'The shop is slow to start work on each request, or requests are piling up.' }
  if (f === 'api_request_count' || f === 'log_line_count') return r.value < (r.normal ?? 0) ? { headline: 'visitors are not getting through', reason: 'Far fewer requests are being answered than usual.' } : { headline: 'there is a rush of visitors', reason: 'Many more requests than usual are arriving.' }
  const name = (r.plain_name || 'a reading').replace(/\s*\([^)]*\)/g, '')
  return { headline: 'something looks wrong', reason: `${name.charAt(0).toUpperCase()}${name.slice(1)} is far from normal.` }
}

export function levelOf(status: string | undefined | null, shopUp: boolean): Level {
  if (!shopUp || !status) return 'idle'
  if (status === 'learning') return 'learning'
  if (status === 'healthy') return 'ok'
  if (status === 'watch') return 'watch'
  if (status === 'failing') return 'problem'
  return 'idle'
}

/** The one sentence in Step 3. */
export function sentence(level: Level, w: TbWindow | null, learnedPct: number): string {
  if (level === 'idle') return 'Ready. Press Start.'
  if (level === 'learning') return `Learning what normal looks like (${learnedPct}% done)`
  if (level === 'ok') return 'Everything looks normal'
  const top = w?.reasons?.[0]
  const p = plainReason(top)
  if (level === 'watch') return `Keeping an eye on it: ${p.headline}`
  return `Something is wrong: ${p.headline}`
}

export const clock12 = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
