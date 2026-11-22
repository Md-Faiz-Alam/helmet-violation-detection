import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Check, ClipboardCheck, Loader2, Trash2 } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router'
import { JobViewer } from '@/components/JobViewer'
import { PageHeader } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { buttonVariants } from '@/components/ui/button-variants'
import { Badge, Card, ErrorNotice, Progress, Skeleton } from '@/components/ui/primitives'
import { api, type Job } from '@/lib/api'
import { useJob, useTracks } from '@/lib/hooks'
import { cn, everyNth, formatTime } from '@/lib/utils'

const STAGES = [
  { key: 'queued', label: 'Queued' },
  { key: 'analyzing', label: 'Detecting & tracking riders' },
  { key: 'reading plates', label: 'Reading plates' },
  { key: 'finished', label: 'Done' },
]

function stageIndex(job: Job) {
  if (job.status === 'finished') return 3
  if (job.stage === 'finishing') return 2
  const i = STAGES.findIndex((s) => s.key === job.stage)
  return i < 0 ? 0 : i
}

export default function JobPage() {
  const { id } = useParams()
  const job = useJob(id)
  const finished = job.data?.status === 'finished'
  const tracks = useTracks(id, finished)
  const violations = useQuery({
    queryKey: ['violations', { job_id: id }],
    queryFn: () => api.violations({ job_id: id }),
    enabled: !!id && finished,
  })
  const navigate = useNavigate()
  const qc = useQueryClient()
  const remove = useMutation({
    mutationFn: () => api.deleteJob(id!),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['jobs'] })
      navigate('/analyze?tab=video')
    },
  })

  const back = (
    <Link to="/analyze?tab=video" className={buttonVariants({ variant: 'ghost', size: 'sm' })}>
      <ArrowLeft /> All videos
    </Link>
  )
  if (job.isPending) return <Skeleton className="h-96 w-full" />
  if (job.isError) return <div className="space-y-4">{back}<ErrorNotice error={job.error} /></div>

  const j = job.data
  return (
    <>
      <PageHeader
        eyebrow={j.readonly ? 'Sample video' : 'Video analysis'}
        title={j.filename ?? 'Video'}
        action={
          <div className="flex items-center gap-2">
            {back}
            {finished && (
              <Link to={`/review?job=${j.id}`} className={buttonVariants({ variant: 'secondary', size: 'sm' })}>
                <ClipboardCheck /> Review violations
              </Link>
            )}
            {j.editable && j.status !== 'running' && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => confirm('Delete this video and its results?') && remove.mutate()}
                disabled={remove.isPending}
                aria-label="Delete this video"
              >
                <Trash2 />
              </Button>
            )}
          </div>
        }
      >
        {j.width}×{j.height} · {j.fps.toFixed(0)} FPS · {formatTime(j.duration_s)} · {everyNth(j.vid_stride)} analysed{j.processing_fps ? ` at ${j.processing_fps.toFixed(1)} FPS on CPU` : ''}
        {j.expires_in_s != null && ` · deleted in ${Math.ceil(j.expires_in_s / 60)} min`}
      </PageHeader>

      {j.status === 'failed' && <ErrorNotice error={j.error ?? 'The analysis failed.'} />}
      {(j.status === 'queued' || j.status === 'running') && <JobProgress job={j} />}
      {finished &&
        (tracks.data && violations.data ? (
          <JobViewer job={j} tracks={tracks.data} violations={violations.data} />
        ) : tracks.isError || violations.isError ? (
          <ErrorNotice error={tracks.error ?? violations.error} />
        ) : (
          <Skeleton className="aspect-video w-full" />
        ))}
    </>
  )
}

function JobProgress({ job }: { job: Job }) {
  const current = stageIndex(job)
  const { frames_done, frames_total } = job.progress
  const fraction = frames_total ? frames_done / frames_total : 0
  const remaining =
    job.processing_fps && frames_total ? (frames_total - frames_done) / job.vid_stride / job.processing_fps : null
  return (
    <Card className="max-w-2xl space-y-6 p-6">
      <ol className="space-y-3">
        {STAGES.map((s, i) => (
          <li key={s.key} className="flex items-center gap-3 text-sm">
            <span
              className={cn(
                'grid size-6 shrink-0 place-items-center rounded-full border text-[11px] font-mono',
                i < current && 'border-helmet/50 bg-helmet/15 text-helmet',
                i === current && 'border-accent/60 bg-accent/15 text-accent',
                i > current && 'border-border text-subtle',
              )}
            >
              {i < current ? <Check className="size-3.5" /> : i === current ? <Loader2 className="size-3.5 animate-spin" /> : i + 1}
            </span>
            <span className={cn(i > current && 'text-subtle')}>{s.label}</span>
            {i === 0 && job.status === 'queued' && job.queue_position > 1 && (
              <Badge className="ml-auto">{job.queue_position - 1} ahead</Badge>
            )}
          </li>
        ))}
      </ol>
      {job.stage === 'analyzing' && (
        <div className="space-y-2">
          <Progress value={fraction} />
          <div className="flex justify-between font-mono text-xs text-muted tabular">
            <span>
              frame {frames_done} / {frames_total}
            </span>
            <span>
              {job.processing_fps ? `${job.processing_fps.toFixed(1)} FPS` : ''}
              {remaining != null ? ` · about ${formatTime(remaining)} left` : ''}
            </span>
          </div>
        </div>
      )}
      {job.stage !== 'analyzing' && job.status === 'running' && <Progress value={0} indeterminate />}
    </Card>
  )
}
