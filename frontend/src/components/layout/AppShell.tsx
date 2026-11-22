import { BookOpen, ClipboardCheck, ExternalLink, Gauge, Moon, ScanSearch, Sun } from 'lucide-react'
import { NavLink, Outlet } from 'react-router'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/primitives'
import { apiUrl } from '@/lib/api'
import { REPO_URL } from '@/data/evaluation'
import { useHealth, useMe, useModelInfo, useTheme } from '@/lib/hooks'
import { AuthorAccess } from './AuthorAccess'
import { cn } from '@/lib/utils'

const nav = [
  { to: '/', label: 'Overview', icon: Gauge, end: true },
  { to: '/analyze', label: 'Analyze', icon: ScanSearch },
  { to: '/review', label: 'Review', icon: ClipboardCheck },
  { to: '/model', label: 'Model card', icon: BookOpen },
]

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <rect width="32" height="32" rx="8" className="fill-panel-2" />
        <path d="M7 19a9 9 0 0 1 18 0v2H7z" fill="none" className="stroke-accent" strokeWidth="2.4" strokeLinejoin="round" />
        <path d="M16 10v5M11 21h10" className="stroke-accent" strokeWidth="2.4" strokeLinecap="round" />
        <circle cx="25" cy="8" r="3" className="fill-violation" />
      </svg>
      <div className="leading-none">
        <div className="text-[15px] font-semibold tracking-tight">RideSafe AI</div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-subtle">Helmet enforcement</div>
      </div>
    </div>
  )
}

function ApiStatus() {
  const health = useHealth()
  const model = useModelInfo()
  const me = useMe()
  const up = health.data?.status === 'ok'
  const label = health.isPending ? 'Connecting' : up ? 'API online' : 'API offline'
  return (
    <div className="flex items-center gap-2">
      {me.data?.role === 'author' ? (
        <Badge tone="accent">Author</Badge>
      ) : (
        model.data?.demo && <Badge tone="accent">Public demo</Badge>
      )}
      <span
        className="flex items-center gap-2 rounded-md border border-border bg-panel px-2.5 py-1 font-mono text-[11px] text-muted"
        title={up ? `RideSafe API ${health.data?.version}, ${model.data?.runtime ?? ''}` : 'The API is not reachable'}
      >
        <span className={cn('size-1.5 rounded-full', health.isPending ? 'bg-pending' : up ? 'bg-helmet shadow-[0_0_8px] shadow-helmet' : 'bg-violation')} />
        {label}
      </span>
    </div>
  )
}

export function AppShell() {
  const { dark, toggle } = useTheme()
  return (
    <div className="flex min-h-dvh flex-col lg:flex-row">
      <aside className="sticky top-0 z-20 flex shrink-0 flex-col border-b border-border bg-panel/85 backdrop-blur lg:h-dvh lg:w-60 lg:border-r lg:border-b-0">
        <div className="flex items-center justify-between px-4 py-3 lg:px-5 lg:py-5">
          <NavLink to="/" aria-label="RideSafe AI home">
            <Logo />
          </NavLink>
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={toggle} aria-label="Toggle colour theme">
            {dark ? <Sun /> : <Moon />}
          </Button>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2 lg:flex-col lg:px-3 lg:pb-0" aria-label="Main">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  'flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-muted transition-colors hover:bg-panel-2 hover:text-fg',
                  isActive && 'bg-panel-2 text-fg shadow-[inset_2px_0_0] shadow-accent',
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          ))}
          <a
            href={apiUrl('/docs')}
            target="_blank"
            rel="noreferrer"
            className="flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-muted transition-colors hover:bg-panel-2 hover:text-fg"
          >
            <ExternalLink className="size-4" />
            API docs
          </a>
        </nav>
        <div className="mt-auto hidden space-y-3 border-t border-border px-5 py-4 lg:block">
          <AuthorAccess />
          <Button variant="ghost" size="sm" className="w-full justify-start px-2" onClick={toggle}>
            {dark ? <Sun /> : <Moon />}
            {dark ? 'Light mode' : 'Dark mode'}
          </Button>
          <p className="text-[11px] leading-relaxed text-subtle">
            Open source under{' '}
            <a className="underline hover:text-fg" href={REPO_URL} target="_blank" rel="noreferrer">
              AGPL-3.0
            </a>
            . Built on Ultralytics YOLO, ONNX Runtime and ByteTrack.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-end gap-3 px-4 pt-4 sm:px-8">
          <ApiStatus />
        </header>
        <main className="flex-1 px-4 pb-10 pt-4 sm:px-8">
          <Outlet />
        </main>
        <footer className="border-t border-border px-4 py-4 text-xs text-subtle sm:px-8 lg:hidden">
          Open source under{' '}
          <a className="underline" href={REPO_URL} target="_blank" rel="noreferrer">
            AGPL-3.0
          </a>
          .
        </footer>
      </div>
    </div>
  )
}

export function PageHeader({ eyebrow, title, children, action }: { eyebrow: string; title: string; children?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-accent">{eyebrow}</div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
        {children && <p className="mt-2 text-sm leading-relaxed text-muted">{children}</p>}
      </div>
      {action}
    </div>
  )
}
