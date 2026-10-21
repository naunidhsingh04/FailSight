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

export type Theme = 'light' | 'dark'
const THEME_KEY = 'failsight.theme'
function readTheme(): Theme {
  try { return window.localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light' } catch { return 'light' }       // light unless the visitor chose dark
}

interface Ctx { mode: Mode; setMode: (m: Mode) => void; detailed: boolean; theme: Theme; setTheme: (t: Theme) => void }
const ModeContext = createContext<Ctx>({ mode: 'simple', setMode: () => undefined, detailed: false, theme: 'light', setTheme: () => undefined })

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<Mode>(read)
  const [theme, setThemeState] = useState<Theme>(readTheme)
  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try { window.localStorage.setItem(THEME_KEY, t) } catch { /* a convenience only */ }
  }, [])
  useEffect(() => { document.documentElement.setAttribute('data-theme', theme) }, [theme])
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
  const value = useMemo(() => ({ mode, setMode, detailed: mode === 'detailed', theme, setTheme }), [mode, setMode, theme, setTheme])
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
