import { useState } from 'react'
import type { DetectResponse } from '@/lib/api'
import { cn } from '@/lib/utils'

const color = { 'With Helmet': 'var(--helmet)', 'Without Helmet': 'var(--violation)' } as const

/** The uploaded image with the API's boxes drawn as an SVG overlay in image coordinates. */
export function ImageResult({ src, result }: { src: string; result: DetectResponse }) {
  const [hover, setHover] = useState<number | null>(null)
  const { width: w, height: h } = result
  const stroke = Math.max(2, Math.round(Math.max(w, h) / 320))
  const font = Math.max(11, Math.round(Math.max(w, h) / 55))
  return (
    <div className="relative overflow-hidden rounded-lg border border-border bg-black">
      <img src={src} alt="Uploaded" className="block h-auto w-full" />
      <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 size-full" role="img" aria-label="Detections">
        {result.detections.map((d, i) => {
          const [x1, y1, x2, y2] = d.box
          const c = color[d.cls]
          const active = hover === i
          const text = `${d.cls === 'With Helmet' ? 'Helmet' : 'No helmet'} ${(d.conf * 100).toFixed(0)}%`
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className="cursor-default">
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill={c}
                fillOpacity={active ? 0.18 : 0.06}
                stroke={c}
                strokeWidth={active ? stroke * 1.6 : stroke}
                rx={stroke}
              />
              {(active || result.detections.length <= 12) && (
                <g transform={`translate(${x1}, ${Math.max(0, y1 - font * 1.5)})`}>
                  <rect width={text.length * font * 0.6 + font * 0.6} height={font * 1.45} fill={c} rx={stroke} />
                  <text x={font * 0.3} y={font * 1.08} fontSize={font} fill="#fff" fontFamily="var(--font-mono)" fontWeight={600}>
                    {text}
                  </text>
                </g>
              )}
            </g>
          )
        })}
      </svg>
      <div className={cn('pointer-events-none absolute bottom-2 left-2 rounded-md bg-black/70 px-2 py-1 font-mono text-[11px] text-white/80')}>
        {w}×{h} · {result.latency_ms.toFixed(0)} ms on CPU
      </div>
    </div>
  )
}
