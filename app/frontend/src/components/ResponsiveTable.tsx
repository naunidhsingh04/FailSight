import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export interface Column<R> {
  key: string
  label: string
  render: (row: R) => ReactNode
  sortValue?: (row: R) => number | string | null
  numeric?: boolean
  nameCell?: boolean
}

interface Props<R> {
  caption: string
  columns: Column<R>[]
  rows: R[]
  rowKey: (row: R) => string
  initialSort?: { key: string; dir: 'asc' | 'desc' }
  activeKey?: string | null
}

/** Sortable table; below 640px every row becomes a stacked card (the labels come from data-label). */
export default function ResponsiveTable<R>({ caption, columns, rows, rowKey, initialSort, activeKey }: Props<R>) {
  const [sort, setSort] = useState(initialSort ?? null)
  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    if (!col?.sortValue) return rows
    const f = col.sortValue
    const dir = sort.dir === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const x = f(a), y = f(b)
      if (x === null && y === null) return 0
      if (x === null) return 1
      if (y === null) return -1
      return (x < y ? -1 : x > y ? 1 : 0) * dir
    })
  }, [rows, sort, columns])
  const toggle = (key: string) => setSort((s) => (s && s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }))
  return (
    <div className="table-wrap">
      <table className="rt">
        <caption>{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" aria-sort={sort?.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined} className={c.numeric ? 'num-cell' : undefined}>
                {c.sortValue ? <button type="button" onClick={() => toggle(c.key)}>{c.label}</button> : c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={rowKey(r)} data-active={activeKey === rowKey(r) ? 'true' : undefined}>
              {columns.map((c) => (
                <td key={c.key} data-label={c.label} className={[c.numeric ? 'num-cell' : '', c.nameCell ? 'name-cell' : ''].join(' ').trim() || undefined}>{c.render(r)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
