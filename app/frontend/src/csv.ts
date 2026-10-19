/** Minimal RFC 4180 CSV reader (quotes, escaped quotes, CRLF). Used only to preview an uploaded file and to pick one row to explain. */
export function parseCsv(text: string, maxRows = 2000): { header: string[]; rows: Record<string, string | number | null>[]; truncated: boolean } {
  const out: string[][] = []
  let row: string[] = []
  let cell = ''
  let q = false
  const src = text.replace(/^﻿/, '')
  for (let i = 0; i < src.length; i++) {
    const c = src[i]
    if (q) {
      if (c === '"') {
        if (src[i + 1] === '"') { cell += '"'; i++ } else q = false
      } else cell += c
    } else if (c === '"') q = true
    else if (c === ',') { row.push(cell); cell = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++
      row.push(cell); cell = ''
      if (row.length > 1 || row[0] !== '') out.push(row)
      row = []
      if (out.length > maxRows + 1) break
    } else cell += c
  }
  if (cell !== '' || row.length) { row.push(cell); if (row.length > 1 || row[0] !== '') out.push(row) }
  const header = out[0] ?? []
  const body = out.slice(1, maxRows + 1)
  const rows = body.map((r) => {
    const o: Record<string, string | number | null> = {}
    header.forEach((h, i) => {
      const v = (r[i] ?? '').trim()
      o[h] = v === '' ? null : v !== '' && !Number.isNaN(Number(v)) ? Number(v) : v
    })
    return o
  })
  return { header, rows, truncated: out.length > maxRows + 1 }
}
