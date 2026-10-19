import type { ReactNode } from 'react'
import type { ApiState } from '../api'
import { ApiError } from '../api'

export function Skeleton({ h = '1rem', w = '100%' }: { h?: string; w?: string }) {
  return <div className="skeleton" style={{ height: h, width: w }} aria-hidden="true" />
}

export function SkeletonBlock({ rows = 4, label = 'Loading' }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-live="polite" aria-busy="true" className="stack">
      <span className="visually-hidden">{label}...</span>
      <Skeleton h="1.6rem" w="45%" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} h="1.1rem" w={`${92 - i * 9}%`} />
      ))}
    </div>
  )
}

export function ErrorState({ error, onRetry, title = 'Could not load this section' }: { error: Error; onRetry?: () => void; title?: string }) {
  const status = error instanceof ApiError ? ` (HTTP ${error.status})` : ''
  return (
    <div className="state error" role="alert">
      <h3>{title}{status}</h3>
      <p>{error.message}</p>
      {error instanceof ApiError && error.status === 0 ? null : <p className="muted small">If the backend is not running, start it with <code>uvicorn app.backend.main:app</code> from the repository root.</p>}
      {onRetry && <button className="btn secondary" onClick={onRetry}>Try again</button>}
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      {children}
    </div>
  )
}

/** Renders loading / error / content for one API call. */
export function Async<T>({ state, children, rows = 4, label }: { state: ApiState<T>; children: (data: T) => ReactNode; rows?: number; label?: string }) {
  if (state.loading && !state.data) return <SkeletonBlock rows={rows} label={label} />
  if (state.error) return <ErrorState error={state.error} onRetry={state.reload} />
  if (!state.data) return <EmptyState title="No data" />
  return <>{children(state.data)}</>
}
