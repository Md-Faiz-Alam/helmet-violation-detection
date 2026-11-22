import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Clock, Film, Image as ImageIcon, Loader2, RotateCcw, ShieldAlert, ShieldCheck, Video } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { Dropzone } from '@/components/Dropzone'
import { ImageResult } from '@/components/ImageResult'
import { PageHeader } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { Badge, Card, CardHeader, EmptyState, ErrorNotice, Progress, Stat } from '@/components/ui/primitives'
import { api, uploadVideo, type Job } from '@/lib/api'
import { useJobs, useModelInfo } from '@/lib/hooks'
import { cn, formatBytes, formatTime, relativeTime } from '@/lib/utils'

const IMAGE_ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp'
const VIDEO_ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.webm,.m4v'

export default function Analyze() {
  const [params, setParams] = useSearchParams()
  const tab = params.get('tab') === 'video' ? 'video' : 'image'
  return (
    <>
      <PageHeader eyebrow="Analyze" title="Check a photo or a video">
        Photos are checked frame by frame. Videos are tracked: every rider gets one decision, made from all the frames
        they appear in, and every violation is logged for review.
      </PageHeader>
      <div className="mb-5 inline-flex rounded-lg border border-border bg-panel p-1" role="tablist">
        {(
          [
            ['image', 'Image', ImageIcon],
            ['video', 'Video', Video],
          ] as const
        ).map(([key, label, Icon]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => setParams(key === 'video' ? { tab: 'video' } : {})}
            className={cn('flex items-center gap-2 rounded-md px-4 py-1.5 text-sm text-muted transition-colors', tab === key && 'bg-panel-2 text-fg')}
          >
            <Icon className="size-4" />
            {label}
          </button>
        ))}
      </div>
      {tab === 'image' ? <ImageAnalyzer /> : <VideoAnalyzer />}
    </>
  )
}

function ImageAnalyzer() {
  const model = useModelInfo()
  const [preview, setPreview] = useState<string | null>(null)
  const detect = useMutation({ mutationFn: api.detect })
  const [tooBig, setTooBig] = useState<string | null>(null)

  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview])

  const onFile = (file: File) => {
    const limit = model.data?.limits.max_image_mb ?? 10
    if (file.size > limit * 1024 ** 2) {
      setTooBig(`${file.name} is ${formatBytes(file.size)}; the limit is ${limit} MB.`)
      return
    }
    setTooBig(null)
    setPreview(URL.createObjectURL(file))
    detect.mutate(file)
  }

  const r = detect.data
  if (!preview) {
    return (
      <div className="max-w-3xl space-y-3">
        <Dropzone
          accept={IMAGE_ACCEPT}
          onFile={onFile}
          icon={<ImageIcon />}
          title="Drop a photo of riders, or click to choose"
          hint={`JPEG, PNG, WebP or BMP · up to ${model.data?.limits.max_image_mb ?? 10} MB`}
        />
        {tooBig && <ErrorNotice error={tooBig} />}
      </div>
    )
  }
  return (
    <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
      <Card className="p-3">
        {r ? (
          <ImageResult src={preview} result={r} />
        ) : (
          <div className="relative overflow-hidden rounded-lg border border-border">
            <img src={preview} alt="Uploaded" className="block w-full opacity-60" />
            {detect.isPending && <div className="scan-line absolute inset-x-0 top-1/2 h-0.5" />}
          </div>
        )}
      </Card>
      <div className="space-y-4">
        <Card className="p-5">
          {detect.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Loader2 className="size-4 animate-spin" /> Detecting heads…
            </div>
          )}
          {detect.isError && <ErrorNotice error={detect.error} />}
          {r && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <Stat label="No helmet" value={r.without_helmet} tone="violation" />
                <Stat label="Helmet" value={r.with_helmet} tone="helmet" />
              </div>
              <p className="text-xs leading-relaxed text-muted">
                {r.without_helmet > 0
                  ? `${r.without_helmet} head${r.without_helmet > 1 ? 's' : ''} without a helmet at the deployed threshold (${model.data?.thresholds['Without Helmet'] ?? '–'}). A photo is one frame: for a decision you can enforce, use video, where each rider is judged over many frames.`
                  : 'No head without a helmet passed the deployed threshold.'}
              </p>
              {r.detections.length > 0 && (
                <ul className="max-h-64 space-y-1 overflow-y-auto pr-1">
                  {r.detections
                    .slice()
                    .sort((a, b) => (a.cls === b.cls ? b.conf - a.conf : a.cls === 'Without Helmet' ? -1 : 1))
                    .map((d, i) => (
                      <li key={i} className="flex items-center justify-between rounded-md bg-panel-2 px-3 py-1.5 text-xs">
                        <span className="flex items-center gap-2">
                          {d.cls === 'Without Helmet' ? <ShieldAlert className="size-3.5 text-violation" /> : <ShieldCheck className="size-3.5 text-helmet" />}
                          {d.cls === 'Without Helmet' ? 'No helmet' : 'Helmet'}
                        </span>
                        <span className="font-mono tabular text-muted">{(d.conf * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                </ul>
              )}
            </div>
          )}
        </Card>
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => {
            setPreview(null)
            detect.reset()
          }}
        >
          <RotateCcw /> Check another photo
        </Button>
      </div>
    </div>
  )
}

function VideoAnalyzer() {
  const model = useModelInfo()
  const jobs = useJobs()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null)
  const [error, setError] = useState<unknown>(null)
  const limits = model.data?.limits

  const onFile = async (file: File) => {
    setError(null)
    if (limits && file.size > limits.max_video_mb * 1024 ** 2) {
      setError(`${file.name} is ${formatBytes(file.size)}; the limit is ${limits.max_video_mb} MB.`)
      return
    }
    setUpload({ name: file.name, progress: 0 })
    try {
      const { job } = await uploadVideo(file, (p) => setUpload({ name: file.name, progress: p }))
      await qc.invalidateQueries({ queryKey: ['jobs'] })
      navigate(`/analyze/jobs/${job.id}`)
    } catch (e) {
      setError(e)
      setUpload(null)
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="space-y-3">
        {upload ? (
          <Card className="space-y-3 p-6">
            <div className="flex items-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin text-accent" />
              Uploading <span className="font-mono">{upload.name}</span>
              <span className="ml-auto font-mono text-muted tabular">{Math.round(upload.progress * 100)}%</span>
            </div>
            <Progress value={upload.progress} />
          </Card>
        ) : (
          <Dropzone
            accept={VIDEO_ACCEPT}
            onFile={onFile}
            icon={<Film />}
            title="Drop a traffic video, or click to choose"
            hint={
              limits
                ? `MP4, MOV, AVI, MKV or WebM · up to ${limits.max_video_mb} MB and ${formatTime(limits.max_video_seconds)} min · every ${limits.vid_stride === 1 ? '' : `${limits.vid_stride}nd `}frame analysed`
                : 'MP4, MOV, AVI, MKV or WebM'
            }
          />
        )}
        {error != null && <ErrorNotice error={error} />}
        {model.data?.demo && (
          <p className="text-xs leading-relaxed text-subtle">
            Public demo: your uploads and their results are visible only in this browser and are deleted after{' '}
            {Math.round((limits?.upload_ttl_s ?? 3600) / 60)} minutes. Number plates are partly masked.
          </p>
        )}
      </div>
      <Card className="overflow-hidden">
        <CardHeader title="Videos" description={model.data?.demo ? 'The sample and your uploads' : 'All analysed videos'} />
        {jobs.isPending ? (
          <div className="p-5 text-sm text-muted">Loading…</div>
        ) : jobs.isError ? (
          <div className="p-4">
            <ErrorNotice error={jobs.error} />
          </div>
        ) : jobs.data.length === 0 ? (
          <EmptyState icon={<Film />} title="No videos yet">
            Upload a clip to see riders tracked and violations logged.
          </EmptyState>
        ) : (
          <ul className="max-h-[60vh] divide-y divide-border overflow-y-auto">
            {jobs.data.map((j) => (
              <JobRow key={j.id} job={j} />
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

const statusTone = { queued: 'neutral', running: 'accent', finished: 'helmet', failed: 'violation' } as const

function JobRow({ job }: { job: Job }) {
  return (
    <li>
      <Link to={`/analyze/jobs/${job.id}`} className="flex items-center gap-3 px-5 py-3 transition-colors hover:bg-panel-2">
        <div className="grid size-9 shrink-0 place-items-center rounded-lg border border-border bg-panel-2 text-muted">
          <Film className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{job.readonly ? 'Sample: ' : ''}{job.filename ?? job.id}</div>
          <div className="flex items-center gap-2 text-xs text-muted">
            <Clock className="size-3" />
            {relativeTime(job.created_at)} · {formatTime(job.duration_s)}
            {job.summary && <span className="text-violation">· {job.summary.violations} violations</span>}
          </div>
        </div>
        <Badge tone={statusTone[job.status]}>{job.status}</Badge>
      </Link>
    </li>
  )
}
