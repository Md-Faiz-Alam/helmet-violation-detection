import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-xl border border-border bg-panel', className)} {...props} />
}

export function CardHeader({
  title,
  description,
  action,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-start justify-between gap-4 border-b border-border px-5 py-4', className)}>
      <div className="min-w-0">
        <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
        {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      </div>
      {action}
    </div>
  )
}

/** Small uppercase label above a section or value */
export function Eyebrow({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn('font-mono text-[11px] uppercase tracking-[0.14em] text-subtle', className)} {...props} />
}

const tone = {
  neutral: 'border-border-strong bg-panel-2 text-muted',
  accent: 'border-accent/35 bg-accent/10 text-accent',
  helmet: 'border-helmet/35 bg-helmet/10 text-helmet',
  violation: 'border-violation/35 bg-violation/10 text-violation',
  uncertain: 'border-uncertain/35 bg-uncertain/10 text-uncertain',
} as const

export type Tone = keyof typeof tone

export function Badge({ tone: t = 'neutral', className, ...props }: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium', tone[t], className)}
      {...props}
    />
  )
}

export function Progress({ value, className, indeterminate }: { value: number; className?: string; indeterminate?: boolean }) {
  return (
    <div
      className={cn('relative h-1.5 overflow-hidden rounded-full bg-panel-2', className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}
    >
      {indeterminate ? (
        <div className="scan-line absolute inset-y-0 w-1/2" />
      ) : (
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${Math.min(100, value * 100)}%` }} />
      )}
    </div>
  )
}

export function Kbd({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-5 items-center justify-center rounded border border-border-strong bg-panel-2 px-1 font-mono text-[10px] text-muted',
        className,
      )}
      {...props}
    />
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-panel-2', className)} />
}

/** A key number: label, big mono value, optional caption */
export function Stat({
  label,
  value,
  caption,
  tone: t,
  className,
}: {
  label: ReactNode
  value: ReactNode
  caption?: ReactNode
  tone?: 'helmet' | 'violation' | 'uncertain' | 'accent'
  className?: string
}) {
  const color = t ? { helmet: 'text-helmet', violation: 'text-violation', uncertain: 'text-uncertain', accent: 'text-accent' }[t] : ''
  return (
    <div className={cn('min-w-0', className)}>
      <Eyebrow>{label}</Eyebrow>
      <div className={cn('mt-1.5 font-mono text-2xl font-semibold tabular tracking-tight', color)}>{value}</div>
      {caption && <div className="mt-1 text-xs text-muted">{caption}</div>}
    </div>
  )
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="grid size-11 place-items-center rounded-xl border border-border bg-panel-2 text-muted [&_svg]:size-5">{icon}</div>
      <div className="text-sm font-medium">{title}</div>
      {children && <div className="max-w-sm text-sm text-muted">{children}</div>}
    </div>
  )
}

export function ErrorNotice({ error, className }: { error: unknown; className?: string }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className={cn('rounded-lg border border-violation/30 bg-violation/8 px-4 py-3 text-sm text-violation', className)} role="alert">
      {message}
    </div>
  )
}
