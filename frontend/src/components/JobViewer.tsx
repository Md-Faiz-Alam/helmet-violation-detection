import { Eye, EyeOff, Layers, Play } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiUrl, jobVideoUrl, type Job, type Tracks, type Violation } from '@/lib/api'
import { reviewText, reviewTone } from '@/lib/labels'
import { drawBoxes, frameAt, type LabelMode } from '@/lib/overlay'
import { cn, formatSeconds, formatTime } from '@/lib/utils'
import { Badge, Card, CardHeader, EmptyState } from './ui/primitives'

export function JobViewer({ job, tracks, violations }: { job: Job; tracks: Tracks; violations: Violation[] }) {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [mode, setMode] = useState<LabelMode>('final')
  const [onlyFlagged, setOnlyFlagged] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(job.duration_s ?? 0)

  const times = useMemo(() => tracks.frames.map(([f]) => (f - 1) / tracks.fps), [tracks])
  const maxGap = (tracks.vid_stride + 0.5) / tracks.fps

  const render = useCallback(
    (t: number) => {
      if (!canvas.current) return
      const i = frameAt(times, t, maxGap)
      drawBoxes(canvas.current, i >= 0 ? tracks.frames[i][1] : [], tracks, mode, onlyFlagged)
    },
    [times, maxGap, tracks, mode, onlyFlagged],
  )

  // Redraw on every presented video frame (exact sync), falling back to animation frames
  useEffect(() => {
    const v = video.current
    if (!v) return
    let handle = 0
    let stopped = false
    const hasRvfc = 'requestVideoFrameCallback' in v
    const loop = (_now: number, meta?: VideoFrameCallbackMetadata) => {
      if (stopped) return
      const t = meta?.mediaTime ?? v.currentTime
      render(t)
      setTime(t)
      handle = hasRvfc ? v.requestVideoFrameCallback(loop) : requestAnimationFrame((n) => loop(n))
    }
    handle = hasRvfc ? v.requestVideoFrameCallback(loop) : requestAnimationFrame((n) => loop(n))
    const onSeek = () => {
      render(v.currentTime)
      setTime(v.currentTime)
    }
    v.addEventListener('seeked', onSeek)
    window.addEventListener('resize', onSeek)
    onSeek()
    return () => {
      stopped = true
      if (hasRvfc) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
      v.removeEventListener('seeked', onSeek)
      window.removeEventListener('resize', onSeek)
    }
  }, [render])

  const seek = (t: number, play = true) => {
    const v = video.current
    if (!v) return
    v.currentTime = Math.max(0, t)
    if (play) void v.play().catch(() => undefined)
  }

  const lanes = useMemo(() => {
    const all = Object.entries(tracks.tracks).filter(([, t]) => t.counted && t.first_s != null && t.last_s != null)
    return (['violation', 'uncertain', 'helmet'] as const).map((label) => ({
      label,
      items: all.filter(([, t]) => t.label === label).map(([id, t]) => ({ id, ...t })),
    }))
  }, [tracks])

  const s = job.summary
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Card className="overflow-hidden">
        <div className="relative bg-black" style={{ aspectRatio: `${job.width} / ${job.height}`, maxHeight: '70vh', marginInline: 'auto' }}>
          <video
            ref={video}
            src={jobVideoUrl(job.id)}
            className="absolute inset-0 size-full"
            controls
            playsInline
            muted
            preload="auto"
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || job.duration_s || 0)}
          />
          <canvas ref={canvas} className="pointer-events-none absolute inset-0 size-full" aria-hidden />
        </div>

        <div className="flex flex-wrap items-center gap-2 border-t border-border px-4 py-3">
          <div className="flex rounded-lg border border-border p-0.5" role="group" aria-label="Box labels">
            {(['final', 'live'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn('rounded-md px-2.5 py-1 text-xs text-muted', mode === m && 'bg-panel-2 text-fg')}
                title={m === 'final' ? 'Colour each rider by the final decision' : 'Colour by what the system knew at that moment (votes still coming in)'}
              >
                {m === 'final' ? 'Final decision' : 'Live votes'}
              </button>
            ))}
          </div>
          <button
            onClick={() => setOnlyFlagged((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-muted hover:text-fg"
          >
            {onlyFlagged ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
            {onlyFlagged ? 'Flagged riders only' : 'All riders'}
          </button>
          <span className="ml-auto font-mono text-xs text-muted tabular">
            {formatTime(time)} / {formatTime(duration)}
          </span>
        </div>

        <Timeline lanes={lanes} duration={duration} time={time} onSeek={seek} />

        {s && (
          <div className="grid grid-cols-2 gap-px border-t border-border bg-border sm:grid-cols-5">
            {[
              ['Riders', s.riders, ''],
              ['Violations', s.violations, 'text-violation'],
              ['Helmet', s.with_helmet, 'text-helmet'],
              ['Uncertain', s.uncertain, 'text-uncertain'],
              ['Plates read', s.plates_read, ''],
            ].map(([k, v, c]) => (
              <div key={k as string} className="bg-panel px-4 py-3">
                <div className="text-[11px] text-muted">{k}</div>
                <div className={cn('font-mono text-lg font-semibold tabular', c as string)}>{v}</div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="flex max-h-[80vh] flex-col overflow-hidden">
        <CardHeader title="Violations" description="Click one to jump to the rider" action={<Badge tone="violation">{violations.length}</Badge>} />
        <div className="flex-1 overflow-y-auto">
          {violations.length === 0 ? (
            <EmptyState icon={<Layers />} title="No violations logged">
              Every counted rider in this video was judged as wearing a helmet, or not seen long enough to decide.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-border">
              {violations.map((v) => (
                <li key={v.id}>
                  <button
                    onClick={() => seek((v.first_seen_s ?? v.video_time_s ?? 0) - 0.3)}
                    className="flex w-full gap-3 px-4 py-3 text-left transition-colors hover:bg-panel-2"
                  >
                    {v.snapshot_url ? (
                      <img src={apiUrl(v.snapshot_url)} alt={`Rider ${v.track_id}`} loading="lazy" className="h-20 w-16 shrink-0 rounded-md border border-border object-cover object-top" />
                    ) : (
                      <div className="h-20 w-16 shrink-0 rounded-md bg-panel-2" />
                    )}
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-semibold">#{v.track_id}</span>
                        <span className="font-mono text-xs text-muted">{formatSeconds(v.video_time_s)}</span>
                        <Play className="ml-auto size-3.5 text-subtle" />
                      </div>
                      <div className="font-mono text-xs text-muted">
                        {v.votes_without_helmet}/{v.votes_without_helmet + v.votes_with_helmet} frames no helmet
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Badge tone={reviewTone[v.review_status]}>{reviewText[v.review_status]}</Badge>
                        {v.plate && <span className="rounded border border-border bg-panel-2 px-1.5 font-mono text-[11px]">{v.plate}</span>}
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Card>
    </div>
  )
}

type Lane = { label: 'violation' | 'uncertain' | 'helmet'; items: { id: string; first_s: number | null; last_s: number | null; violation_s: number | null }[] }

function Timeline({ lanes, duration, time, onSeek }: { lanes: Lane[]; duration: number; time: number; onSeek: (t: number, play?: boolean) => void }) {
  const pct = (t: number) => `${(Math.min(Math.max(t / (duration || 1), 0), 1) * 100).toFixed(3)}%`
  const names = { violation: 'No helmet', uncertain: 'Uncertain', helmet: 'Helmet' }
  const bg = { violation: 'bg-violation', uncertain: 'bg-uncertain', helmet: 'bg-helmet' }
  return (
    <div className="border-t border-border px-4 py-3">
      <div className="space-y-1.5">
        {lanes.map((lane) => (
          <div key={lane.label} className="flex items-center gap-3">
            <div className="w-20 shrink-0 text-[11px] text-muted">
              {names[lane.label]} <span className="font-mono text-subtle">{lane.items.length}</span>
            </div>
            <div
              className="relative h-4 flex-1 cursor-pointer rounded bg-panel-2"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect()
                onSeek(((e.clientX - r.left) / r.width) * duration, false)
              }}
            >
              {lane.items.map((it) => (
                <button
                  key={it.id}
                  title={`Rider #${it.id}: ${formatSeconds(it.first_s)} to ${formatSeconds(it.last_s)}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    onSeek((it.first_s ?? 0) - 0.3)
                  }}
                  className={cn('absolute inset-y-0.5 min-w-1 rounded-sm opacity-70 hover:opacity-100', bg[lane.label])}
                  style={{ left: pct(it.first_s ?? 0), width: `calc(${pct((it.last_s ?? 0) - (it.first_s ?? 0))} + 3px)` }}
                />
              ))}
              <div className="pointer-events-none absolute inset-y-[-3px] w-px bg-fg" style={{ left: pct(time) }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
