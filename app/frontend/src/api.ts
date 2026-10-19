import { useCallback, useEffect, useState } from 'react'

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, message: string, detail: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

function messageOf(body: any, status: number): string {
  const e = body?.error ?? body?.detail
  if (typeof e === 'string') return e
  if (e && typeof e === 'object') return e.message ?? JSON.stringify(e)
  return `Request failed (${status})`
}

async function parse(res: Response) {
  const text = await res.text()
  let body: any = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { error: text.slice(0, 300) }
  }
  if (!res.ok) throw new ApiError(res.status, messageOf(body, res.status), body)
  return body
}

const cache = new Map<string, Promise<any>>()

export function getJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(path, { headers: { Accept: 'application/json' } }).then(parse)
    p.catch(() => cache.delete(path))
    cache.set(path, p)
  }
  return cache.get(path) as Promise<T>
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(body) }).then(parse)
}

export function postForm<T>(path: string, form: FormData): Promise<T> {
  return fetch(path, { method: 'POST', body: form, headers: { Accept: 'application/json' } }).then(parse)
}

export interface ApiState<T> {
  data: T | null
  error: ApiError | Error | null
  loading: boolean
  reload: () => void
}

export function useApi<T>(path: string | null): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [loading, setLoading] = useState<boolean>(path !== null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (path === null) return
    let alive = true
    setLoading(true)
    setError(null)
    if (tick > 0) cache.delete(path)
    getJson<T>(path)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [path, tick])
  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}
