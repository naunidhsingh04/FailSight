import { useEffect } from 'react'

export function usePageTitle(title: string) {
  useEffect(() => {
    document.title = title === 'Overview' ? 'FailSight - API failure prediction' : `${title} - FailSight`
  }, [title])
}
