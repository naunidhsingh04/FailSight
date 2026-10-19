import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type Mode = 'simple' | 'detailed'
const KEY = 'failsight.mode'

function read(): Mode {
  try {
    const v = window.localStorage.getItem(KEY)
    return v === 'detailed' ? 'detailed' : 'simple'
  } catch {
    return 'simple' // storage can be blocked (private windows, strict privacy settings)
  }
}

interface Ctx { mode: Mode; setMode: (m: Mode) => void; detailed: boolean }
const ModeContext = createContext<Ctx>({ mode: 'simple', setMode: () => undefined, detailed: false })

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<Mode>(read)
  const setMode = useCallback((m: Mode) => {
    setModeState(m)
    try {
      window.localStorage.setItem(KEY, m)
    } catch {
      /* remembering the choice is a convenience only */
    }
  }, [])
  useEffect(() => {
    document.documentElement.setAttribute('data-mode', mode)
  }, [mode])
  const value = useMemo(() => ({ mode, setMode, detailed: mode === 'detailed' }), [mode, setMode])
  return <ModeContext.Provider value={value}>{children}</ModeContext.Provider>
}

export const useMode = () => useContext(ModeContext)

/** content shown only in Detailed mode */
export function Detailed({ children }: { children: ReactNode }) {
  return useMode().detailed ? <>{children}</> : null
}
/** content shown only in Simple mode */
export function Simple({ children }: { children: ReactNode }) {
  return useMode().detailed ? null : <>{children}</>
}
