import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** 83.4 -> "1:23", 3725 -> "1:02:05" */
export function formatTime(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return '–'
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

/** Seconds with one decimal for precise positions, e.g. 12.4 s */
export function formatSeconds(seconds: number | null | undefined) {
  return seconds == null ? '–' : `${seconds.toFixed(1)} s`
}

/** "every frame", "every 2nd frame", "every 3rd frame" */
export function everyNth(n: number) {
  if (n <= 1) return 'every frame'
  const suffix = n % 100 >= 11 && n % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'
  return `every ${n}${suffix} frame`
}

export function formatPercent(value: number, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

export function relativeTime(iso: string) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`
  return new Date(iso).toLocaleDateString()
}
