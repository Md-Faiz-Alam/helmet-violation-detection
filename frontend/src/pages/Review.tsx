import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, ClipboardCheck, ExternalLink, Lock, RotateCcw, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { PageHeader } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { buttonVariants } from '@/components/ui/button-variants'
import { Badge, Card, EmptyState, ErrorNotice, Eyebrow, Kbd, Skeleton } from '@/components/ui/primitives'
import { reviewText, reviewTone } from '@/lib/labels'
import { api, apiUrl, type ReviewStatus, type Violation } from '@/lib/api'
import { useHotkeys, useJobs } from '@/lib/hooks'
import { cn, formatSeconds } from '@/lib/utils'

type Filter = ReviewStatus | 'all'
const FILTERS: { key: Filter; label: string }[] = [
  { key: 'pending', label: 'To review' },
  { key: 'confirmed', label: 'Confirmed' },
  { key: 'rejected', label: 'Rejected' },
  { key: 'all', label: 'All' },
]

export default function Review() {
  const [params, setParams] = useSearchParams()
  const jobId = params.get('job') ?? undefined
  const [filter, setFilter] = useState<Filter>('pending')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const jobs = useJobs()
  const qc = useQueryClient()
  const key = ['violations', { job_id: jobId }]
  const all = useQuery({ queryKey: key, queryFn: () => api.violations({ job_id: jobId }) })

  const review = useMutation({
    mutationFn: ({ id, status }: { id: number; status: ReviewStatus }) => api.review(id, status),
    onSuccess: (updated) => {
      qc.setQueryData<Violation[]>(key, (old) => old?.map((v) => (v.id === updated.id ? updated : v)))
      void qc.invalidateQueries({ queryKey: ['violations'], refetchType: 'none' })
    },
  })

  const counts = useMemo(() => {
    const c: Record<Filter, number> = { pending: 0, confirmed: 0, rejected: 0, all: 0 }
    for (const v of all.data ?? []) {
      c[v.review_status]++
      c.all++
    }
    return c
  }, [all.data])

  const list = useMemo(() => (all.data ?? []).filter((v) => filter === 'all' || v.review_status === filter), [all.data, filter])
  // The selected item, or the first one when it is not in the current list (another filter, or just reviewed)
  const index = Math.max(0, list.findIndex((v) => v.id === selectedId))
  const selected = list[index] as Violation | undefined

  const jobName = (id: string | null) => jobs.data?.find((j) => j.id === id)?.filename ?? 'video'
  const move = (d: number) => list.length && setSelectedId(list[(index + d + list.length) % list.length].id)
  const decide = (status: ReviewStatus) => {
    if (!selected?.editable || review.isPending) return
    const next = filter === 'pending' && status !== 'pending' ? list[index + 1] ?? list[index - 1] : selected
    review.mutate({ id: selected.id, status }, { onSuccess: () => next && setSelectedId(next.id) })
  }
  useHotkeys({ j: () => move(1), k: () => move(-1), c: () => decide('confirmed'), r: () => decide('rejected'), u: () => decide('pending') })

  return (
    <>
      <PageHeader eyebrow="Review console" title="Confirm or reject each violation">
        The system flags; a person decides. Every logged violation waits here with its evidence until someone confirms it or
        rejects it as a false alarm.
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-lg border border-border bg-panel p-1">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={cn('flex items-center gap-2 rounded-md px-3 py-1.5 text-sm text-muted', filter === f.key && 'bg-panel-2 text-fg')}
            >
              {f.label}
              <span className="font-mono text-xs text-subtle tabular">{counts[f.key]}</span>
            </button>
          ))}
        </div>
        {jobId && (
          <Badge tone="accent" className="py-1">
            {jobName(jobId)}
            <button onClick={() => setParams({})} aria-label="Show all videos">
              <X className="size-3" />
            </button>
          </Badge>
        )}
        <div className="ml-auto hidden items-center gap-3 text-xs text-muted md:flex">
          <span className="flex items-center gap-1"><Kbd>J</Kbd><Kbd>K</Kbd> next / previous</span>
          <span className="flex items-center gap-1"><Kbd>C</Kbd> confirm</span>
          <span className="flex items-center gap-1"><Kbd>R</Kbd> reject</span>
          <span className="flex items-center gap-1"><Kbd>U</Kbd> undo</span>
        </div>
      </div>

      {all.isError && <ErrorNotice error={all.error} />}
      {all.isPending ? (
        <Skeleton className="h-[60vh] w-full" />
      ) : list.length === 0 ? (
        <Card>
          <EmptyState icon={<ClipboardCheck />} title={filter === 'pending' && counts.all ? 'All caught up' : 'Nothing here'}>
            {counts.all === 0 ? (
              <>
                No violations yet.{' '}
                <Link className="text-accent underline" to="/analyze?tab=video">
                  Analyse a video
                </Link>{' '}
                to fill the queue.
              </>
            ) : (
              'No violations with this status.'
            )}
          </EmptyState>
        </Card>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
          <Card className="overflow-hidden">
            <ul className="max-h-[72vh] divide-y divide-border overflow-y-auto" aria-label="Violations">
              {list.map((v) => (
                <li key={v.id}>
                  <button
                    onClick={() => setSelectedId(v.id)}
                    className={cn(
                      'flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-panel-2',
                      v.id === selected?.id && 'bg-panel-2 shadow-[inset_2px_0_0] shadow-accent',
                    )}
                  >
                    {v.snapshot_url ? (
                      <img src={apiUrl(v.snapshot_url)} alt="" loading="lazy" className="h-14 w-11 shrink-0 rounded border border-border object-cover object-top" />
                    ) : (
                      <div className="h-14 w-11 shrink-0 rounded bg-panel-2" />
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="font-mono font-semibold">#{v.track_id}</span>
                        <span className="font-mono text-xs text-muted">{formatSeconds(v.video_time_s)}</span>
                      </div>
                      <div className="truncate text-xs text-subtle">{jobName(v.job_id)}</div>
                    </div>
                    <span className={cn('size-2 shrink-0 rounded-full', { pending: 'bg-pending', confirmed: 'bg-violation', rejected: 'bg-helmet' }[v.review_status])} />
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          {selected && <Detail v={selected} jobName={jobName(selected.job_id)} onDecide={decide} busy={review.isPending} error={review.error} />}
        </div>
      )}
    </>
  )
}

function Detail({
  v,
  jobName,
  onDecide,
  busy,
  error,
}: {
  v: Violation
  jobName: string
  onDecide: (s: ReviewStatus) => void
  busy: boolean
  error: unknown
}) {
  const total = v.votes_without_helmet + v.votes_with_helmet
  const share = total ? v.votes_without_helmet / total : 0
  return (
    <Card className="@container overflow-hidden">
      <div className="grid @2xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1fr)]">
      <div className="grid place-items-center border-b border-border bg-black/40 p-4 @2xl:border-r @2xl:border-b-0">
        {v.snapshot_url ? (
          <img src={apiUrl(v.snapshot_url)} alt={`Rider ${v.track_id}`} className="max-h-[62vh] w-auto rounded-md border border-border" />
        ) : (
          <div className="text-sm text-muted">No snapshot</div>
        )}
      </div>
      <div className="flex flex-col gap-6 p-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Eyebrow>Rider</Eyebrow>
            <div className="mt-1 font-mono text-3xl font-semibold">#{v.track_id}</div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <Badge tone={reviewTone[v.review_status]}>{reviewText[v.review_status]}</Badge>
            {v.reviewed_by && v.reviewed_at && (
              <span className="text-[11px] text-subtle" title="Audit trail: who last changed the review status">
                by {v.reviewed_by} · {new Date(v.reviewed_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm">
          <Field label="No helmet from">{formatSeconds(v.video_time_s)}</Field>
          <Field label="In view">
            {formatSeconds(v.first_seen_s)} – {formatSeconds(v.last_seen_s)}
          </Field>
          <Field label="Highest confidence">{v.max_conf != null ? `${(v.max_conf * 100).toFixed(0)}%` : '–'}</Field>
          <Field label="Plate">
            {v.plate ? (
              <span className="rounded border border-border-strong bg-panel-2 px-1.5 py-0.5">{v.plate}</span>
            ) : (
              <span className="text-muted">not read</span>
            )}
            {v.plate_frames && <span className="ml-2 text-xs text-muted">{v.plate_frames} frames agree</span>}
          </Field>
        </dl>

        <div>
          <div className="mb-2 flex justify-between text-xs text-muted">
            <span>Frame votes</span>
            <span className="font-mono tabular">
              {v.votes_without_helmet} no helmet · {v.votes_with_helmet} helmet
            </span>
          </div>
          <div className="flex h-2 overflow-hidden rounded-full bg-panel-2">
            <div className="bg-violation" style={{ width: `${share * 100}%` }} />
            <div className="bg-helmet" style={{ width: `${(1 - share) * 100}%` }} />
          </div>
          <p className="mt-2 text-xs leading-relaxed text-subtle">
            Logged because at least 5 frames, and 60% of this rider's votes, saw no helmet, on a motorcycle.
          </p>
        </div>

        {v.job_id && (
          <Link to={`/analyze/jobs/${v.job_id}`} className={cn(buttonVariants({ variant: 'ghost', size: 'sm' }), 'self-start px-2')}>
            <ExternalLink /> Watch in {jobName}
          </Link>
        )}

        <div className="mt-auto space-y-3 border-t border-border pt-5">
          {v.editable ? (
            <div className="flex flex-wrap gap-2">
              <Button variant="violation" onClick={() => onDecide('confirmed')} disabled={busy || v.review_status === 'confirmed'}>
                <Check /> Confirm violation <Kbd className="ml-1">C</Kbd>
              </Button>
              <Button variant="helmet" onClick={() => onDecide('rejected')} disabled={busy || v.review_status === 'rejected'}>
                <X /> Not a violation <Kbd className="ml-1">R</Kbd>
              </Button>
              {v.review_status !== 'pending' && (
                <Button variant="ghost" onClick={() => onDecide('pending')} disabled={busy}>
                  <RotateCcw /> Undo
                </Button>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Lock className="size-4" />
              Read-only: only the uploader of this video or the author can review it (the sample is read-only).
            </div>
          )}
          {error != null && <ErrorNotice error={error} />}
        </div>
      </div>
      </div>
    </Card>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 font-mono">{children}</dd>
    </div>
  )
}
