import { useEffect } from 'react'

export function usePageTitle(title: string) {
  useEffect(() => {
    document.title = title === 'Console' ? 'FailSight - live failure monitor' : `${title} - FailSight`
  }, [title])
}
