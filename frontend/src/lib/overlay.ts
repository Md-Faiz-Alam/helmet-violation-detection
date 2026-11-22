import type { TrackBox, TrackLabel, Tracks } from './api'
import { labelText } from './labels'

export type LabelMode = 'final' | 'live'

const LABEL_COLOR: Record<TrackLabel, string> = {
  violation: '--violation',
  uncertain: '--uncertain',
  helmet: '--helmet',
  pending: '--pending',
}

/** The four status colours from the current theme (read once per draw, not once per box). */
function themeColors(): Record<TrackLabel, string> {
  const style = getComputedStyle(document.documentElement)
  const read = (label: TrackLabel) => style.getPropertyValue(LABEL_COLOR[label]).trim() || '#888888'
  return { violation: read('violation'), uncertain: read('uncertain'), helmet: read('helmet'), pending: read('pending') }
}

/** Index of the processed frame shown at time t (the last one at or before t), or -1. */
export function frameAt(times: number[], t: number, maxGap: number) {
  let lo = 0
  let hi = times.length - 1
  let best = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] <= t + 1e-3) {
      best = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return best >= 0 && t - times[best] <= maxGap ? best : -1
}

export function drawBoxes(
  canvas: HTMLCanvasElement,
  boxes: TrackBox[],
  tracks: Tracks,
  mode: LabelMode,
  onlyFlagged: boolean,
) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  const colors = themeColors()
  const sx = w / tracks.width
  const sy = h / tracks.height
  const font = Math.max(10, Math.min(13, w / 70))
  ctx.font = `600 ${font}px "JetBrains Mono Variable", monospace`
  ctx.textBaseline = 'middle'
  for (const [id, x1, y1, x2, y2, live] of boxes) {
    const label = mode === 'final' ? (tracks.tracks[id]?.label ?? live) : live
    if (onlyFlagged && label !== 'violation' && label !== 'uncertain') continue
    const color = colors[label]
    const x = x1 * sx
    const y = y1 * sy
    const bw = (x2 - x1) * sx
    const bh = (y2 - y1) * sy
    ctx.lineWidth = label === 'violation' ? 2.5 : 1.75
    ctx.strokeStyle = color
    ctx.fillStyle = color + '22'
    ctx.beginPath()
    ctx.roundRect(x, y, bw, bh, 3)
    ctx.fill()
    ctx.stroke()
    const text = `#${id} ${labelText[label].toUpperCase()}`
    const tw = ctx.measureText(text).width + 8
    const th = font + 6
    const ty = y - th - 2 < 0 ? y + bh + 2 : y - th - 2
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.roundRect(x, ty, tw, th, 3)
    ctx.fill()
    ctx.fillStyle = '#0a0e13'
    ctx.fillText(text, x + 4, ty + th / 2 + 0.5)
  }
}
