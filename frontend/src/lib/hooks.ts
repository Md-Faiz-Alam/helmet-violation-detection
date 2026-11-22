import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useState } from 'react'
import { api } from './api'

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000, retry: false })
}

export function useModelInfo() {
  return useQuery({ queryKey: ['model'], queryFn: api.model, staleTime: Infinity, retry: 1 })
}

export function useJobs() {
  return useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 15_000 })
}

/** Poll a job every second while it is queued or running. */
export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ['queued', 'running'].includes(q.state.data.status) ? 1000 : false),
  })
}

export function useTracks(id: string | undefined, ready: boolean) {
  return useQuery({ queryKey: ['tracks', id], queryFn: () => api.tracks(id!), enabled: !!id && ready, staleTime: Infinity })
}

export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('ridesafe-theme', dark ? 'dark' : 'light')
    } catch {
      // storage blocked: the choice lasts for this page only
    }
  }, [dark])
  return { dark, toggle: useCallback(() => setDark((d) => !d), []) }
}

/** Keyboard shortcuts, ignored while typing in a form field. */
export function useHotkeys(map: Record<string, () => void>, enabled = true) {
  useEffect(() => {
    if (!enabled) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (e.ctrlKey || e.metaKey || e.altKey || t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName)) return
      const fn = map[e.key.toLowerCase()]
      if (fn) {
        e.preventDefault()
        fn()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [map, enabled])
}
