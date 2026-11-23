import { ArrowRight, BookOpen, Box, ClipboardCheck, Database, Film, ListChecks, ScanFace, Vote } from 'lucide-react'
import { useEffect, useMemo, useRef } from 'react'
import { Link } from 'react-router'
import { buttonVariants } from '@/components/ui/button-variants'
import { Card, Eyebrow, Stat } from '@/components/ui/primitives'
import { headline } from '@/data/evaluation'
import { jobVideoUrl, type Job, type Tracks } from '@/lib/api'
import { useJobs, useTracks } from '@/lib/hooks'
import { drawBoxes, frameAt } from '@/lib/overlay'
import { cn } from '@/lib/utils'

const steps = [
  { icon: Film, title: 'Video in', text: 'Upload, file or camera stream. Every 2nd frame is analysed on CPU.' },
  { icon: ScanFace, title: 'Detect heads', text: 'YOLO26s on ONNX Runtime finds every head: helmet or no helmet.' },
  { icon: Box, title: 'Track riders', text: 'ByteTrack links a head across frames, so each rider is one track.' },
  { icon: Vote, title: 'Vote per rider', text: '≥5 no-helmet frames, ≥60% of votes and on a motorcycle: a violation. Pedestrians and cyclists are not logged.' },
  { icon: Database, title: 'Log + plate', text: 'Logged the moment it is decided, with a snapshot. Plate read by OCR.' },
  { icon: ClipboardCheck, title: 'Human review', text: 'A reviewer confirms or rejects. Nothing is enforced automatically.' },
]

export default function Overview() {
  const jobs = useJobs()
  const sample = jobs.data?.find((j) => j.readonly && j.status === 'finished')
  const pct = (h: { hit: number; total: number }) => Math.round((h.hit / h.total) * 100)

  return (
    <div className="space-y-10">
      <section className="grid items-center gap-8 pt-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          <Eyebrow className="text-accent">Two-wheeler helmet enforcement</Eyebrow>
          <h1 className="mt-3 text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl">
            Spot riders without a helmet on real traffic video.
            <span className="text-muted"> Let a person make the call.</span>
          </h1>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-muted">
            RideSafe AI tracks every rider through a video and decides once per rider, from all the frames they appear in,
            not frame by frame. Each violation is logged with a snapshot and a plate reading, and waits for a reviewer.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link to="/analyze?tab=video" className={buttonVariants({ variant: 'primary', size: 'lg' })}>
              Analyze a video <ArrowRight />
            </Link>
            <Link to="/model" className={buttonVariants({ variant: 'secondary', size: 'lg' })}>
              <BookOpen /> How well it works
            </Link>
          </div>
        </div>
        {sample ? <HeroDemo job={sample} /> : <HeroIllustration />}
      </section>

      <section>
        <Card className="grid grid-cols-2 gap-6 p-6 lg:grid-cols-4">
          <Stat
            label="Violators caught"
            value={`${pct(headline.videoRecall)}%`}
            tone="violation"
            caption={`${headline.videoRecall.hit} of ${headline.videoRecall.total} on 4 hand-labelled test clips`}
          />
          <Stat
            label="Logs that are real"
            value={`${pct(headline.videoPrecision)}%`}
            tone="helmet"
            caption={`${headline.helmetedRidersFlagged} riders wearing a helmet were flagged`}
          />
          <Stat
            label="Caps, turbans, masks"
            value={headline.headwearMap50.toFixed(2)}
            tone="accent"
            caption={`mAP50 on 635 held-out headwear images, up from ${headline.headwearMap50Before.toFixed(2)}`}
          />
          <Stat label="On a CPU" value={`${headline.cpuFpsVideo} FPS`} caption="Video tracking with every check, no GPU, no PyTorch" />
        </Card>
        <p className="mt-3 text-xs leading-relaxed text-subtle">
          Recall is the weak spot: far-away riders are not tracked and some headwear is still read as a helmet.{' '}
          <Link to="/model" className="underline hover:text-fg">
            Limitations
          </Link>
          .
        </p>
      </section>

      <section>
        <div className="mb-4 flex items-center gap-3">
          <ListChecks className="size-4 text-accent" />
          <h2 className="text-lg font-semibold tracking-tight">How it works</h2>
        </div>
        <ol className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          {steps.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="relative">
              <Card className="h-full p-4">
                <div className="flex items-center justify-between">
                  <div className="grid size-9 place-items-center rounded-lg border border-border bg-panel-2 text-accent">
                    <Icon className="size-4" />
                  </div>
                  <span className="font-mono text-[11px] text-subtle">0{i + 1}</span>
                </div>
                <div className="mt-3 text-sm font-semibold">{title}</div>
                <p className="mt-1 text-xs leading-relaxed text-muted">{text}</p>
              </Card>
              {i < steps.length - 1 && (
                <ArrowRight className="absolute top-1/2 -right-2.5 z-10 hidden size-4 -translate-y-1/2 text-subtle xl:block" />
              )}
            </li>
          ))}
        </ol>
      </section>
    </div>
  )
}

/** The read-only sample video, looping, with the tracked boxes drawn live. */
function HeroDemo({ job }: { job: Job }) {
  const tracks = useTracks(job.id, true)
  return (
    <Card className="relative overflow-hidden p-0">
      {tracks.data ? <LoopingOverlay job={job} tracks={tracks.data} /> : <div className="aspect-video animate-pulse bg-panel-2" />}
      <div className="flex items-center justify-between border-t border-border px-4 py-2.5 text-xs text-muted">
        <span className="flex items-center gap-2">
          <span className="size-1.5 animate-pulse rounded-full bg-violation" /> Sample analysis · {job.summary?.riders ?? '–'} riders ·{' '}
          {job.summary?.violations ?? '–'} violations
        </span>
        <Link to={`/analyze/jobs/${job.id}`} className="text-accent hover:underline">
          Open
        </Link>
      </div>
      {job.credit && <div className="border-t border-border px-4 py-2 text-[11px] text-muted">{job.credit}</div>}
    </Card>
  )
}

function LoopingOverlay({ job, tracks }: { job: Job; tracks: Tracks }) {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const times = useMemo(() => tracks.frames.map(([f]) => (f - 1) / tracks.fps), [tracks])
  useEffect(() => {
    const v = video.current
    if (!v) return
    let handle = 0
    let stopped = false
    const draw = () => {
      if (stopped || !canvas.current) return
      const i = frameAt(times, v.currentTime, (tracks.vid_stride + 0.5) / tracks.fps)
      drawBoxes(canvas.current, i >= 0 ? tracks.frames[i][1] : [], tracks, 'final', false)
      handle = requestAnimationFrame(draw)
    }
    handle = requestAnimationFrame(draw)
    return () => {
      stopped = true
      cancelAnimationFrame(handle)
    }
  }, [times, tracks])
  return (
    <div className="relative bg-black" style={{ aspectRatio: `${job.width} / ${job.height}` }}>
      <video ref={video} src={jobVideoUrl(job.id)} className="absolute inset-0 size-full" autoPlay muted loop playsInline />
      <canvas ref={canvas} className="pointer-events-none absolute inset-0 size-full" aria-hidden />
    </div>
  )
}

/** Shown when no sample job is available: a schematic street scene with tracked heads. */
function HeroIllustration() {
  const heads = [
    { x: 70, y: 70, label: '#4 NO HELMET', c: 'var(--violation)' },
    { x: 230, y: 92, label: '#7 HELMET', c: 'var(--helmet)' },
    { x: 375, y: 64, label: '#9 UNCERTAIN', c: 'var(--uncertain)' },
  ]
  return (
    <Card className="overflow-hidden p-0">
      <svg viewBox="0 0 500 280" className="block w-full" role="img" aria-label="Riders tracked and labelled">
        <defs>
          <linearGradient id="road" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--panel-2)" />
            <stop offset="1" stopColor="var(--panel)" />
          </linearGradient>
        </defs>
        <rect width="500" height="280" fill="url(#road)" />
        {[0, 1, 2, 3, 4].map((i) => (
          <rect key={i} x={20 + i * 110} y={238} width={60} height={4} rx={2} fill="var(--border-strong)" />
        ))}
        {heads.map((h, i) => (
          <g key={i} className={cn(i === 0 && 'animate-pulse')}>
            <ellipse cx={h.x + 30} cy={h.y + 120} rx={42} ry={14} fill="var(--border)" />
            <rect x={h.x + 8} y={h.y + 40} width={44} height={70} rx={14} fill="var(--border-strong)" />
            <circle cx={h.x + 30} cy={h.y + 24} r={17} fill="var(--subtle)" />
            <rect x={h.x + 6} y={h.y} width={48} height={48} rx={4} fill="none" stroke={h.c} strokeWidth={2.5} />
            <rect x={h.x + 6} y={h.y - 18} width={h.label.length * 6.6 + 8} height={16} rx={3} fill={h.c} />
            <text x={h.x + 10} y={h.y - 6.5} fontSize={10} fontFamily="var(--font-mono)" fontWeight={600} fill="#0a0e13">
              {h.label}
            </text>
          </g>
        ))}
      </svg>
    </Card>
  )
}
