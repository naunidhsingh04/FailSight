import { useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { GLOSSARY } from '../glossary'
import { useMode } from '../mode'

/** Inline "?" help: works with hover, keyboard focus and touch (tap toggles); Escape or clicking elsewhere closes it. */
export default function Term({ k, children }: { k: string; children?: ReactNode }) {
  const e = GLOSSARY[k]
  const id = useId()
  const [hover, setHover] = useState(false)
  const [pinned, setPinned] = useState(false)          // a tap or click keeps it open (touch screens have no hover)
  const [focused, setFocused] = useState(false)
  const open = hover || pinned || focused
  const setOpen = (v: boolean) => { if (!v) { setHover(false); setPinned(false); setFocused(false) } }
  const ref = useRef<HTMLSpanElement>(null)
  const { detailed } = useMode()
  useEffect(() => {
    if (!open) return
    const onKey = (ev: KeyboardEvent) => { if (ev.key === 'Escape') setOpen(false) }
    const onDoc = (ev: MouseEvent | TouchEvent) => { if (ref.current && !ref.current.contains(ev.target as Node)) setOpen(false) }
    document.addEventListener('keydown', onKey)
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('touchstart', onDoc)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('touchstart', onDoc)
    }
  }, [open])
  if (!e) return <>{children}</>
  return (
    <span className="term" ref={ref} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      {children ?? e.term}
      <button type="button" className="term-btn" aria-label={`What does “${e.term}” mean?`} aria-expanded={open} aria-describedby={open ? id : undefined}
        onClick={(ev) => { if (pinned) { setOpen(false); ev.currentTarget.blur() } else setPinned(true) }} onFocus={() => setFocused(true)} onBlur={(ev) => { if (!ref.current?.contains(ev.relatedTarget as Node)) setFocused(false) }}>?</button>
      {open && (
        <span role="tooltip" id={id} className="term-pop">
          <strong>{e.term}</strong>
          {e.plain}
          {detailed && e.more ? <><br /><br />{e.more}</> : null}
        </span>
      )}
    </span>
  )
}
