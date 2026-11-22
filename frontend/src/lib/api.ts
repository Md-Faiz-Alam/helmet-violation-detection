import { authorKey, setAuthorKey } from './author'
import { ownerToken } from './owner'

/** API base URL: VITE_API_URL in production builds, /api (Vite dev proxy) otherwise. */
export const API_URL = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '')

export const apiUrl = (path: string) => `${API_URL}${path}`

// ----------------------------------------------------------------------------- types (mirror backend/ridesafe/api.py)

export type ClassName = 'With Helmet' | 'Without Helmet'
export type TrackLabel = 'violation' | 'uncertain' | 'helmet' | 'pending'
export type ReviewStatus = 'pending' | 'confirmed' | 'rejected'
export type JobStatus = 'queued' | 'running' | 'finished' | 'failed'

export interface Health {
  status: string
  version: string
  model_loaded: boolean
}

export interface ModelInfo {
  classes: ClassName[]
  thresholds: Record<ClassName, number>
  input_size: number
  dynamic_input: boolean
  runtime: string
  plates_enabled: boolean
  demo: boolean
  author_enabled: boolean
  limits: {
    max_image_mb: number
    max_video_mb: number
    max_video_seconds: number
    vid_stride: number
    upload_ttl_s: number
    detect_per_min: number
    uploads_per_hour: number
  }
}

export interface Me {
  role: 'author' | 'visitor'
}

export interface Detection {
  cls: ClassName
  conf: number
  box: [number, number, number, number]
}

export interface DetectResponse {
  width: number
  height: number
  latency_ms: number
  with_helmet: number
  without_helmet: number
  detections: Detection[]
}

export interface JobSummary {
  riders: number
  with_helmet: number
  violations: number
  uncertain: number
  tracks_total: number
  plates_read: number
}

export interface Job {
  id: string
  status: JobStatus
  stage: string
  filename: string | null
  created_at: string
  fps: number
  width: number
  height: number
  duration_s: number | null
  vid_stride: number
  progress: { frames_done: number; frames_total: number }
  queue_position: number
  processing_fps: number | null
  summary: JobSummary | null
  error: string | null
  readonly: boolean
  editable: boolean
  expires_in_s: number | null
}

/** [track id, x1, y1, x2, y2, label at that frame, confidence] */
export type TrackBox = [number, number, number, number, number, TrackLabel, number]

export interface TrackInfo {
  label: TrackLabel
  counted: boolean
  violation: boolean
  violation_id: number | null
  first_s: number | null
  last_s: number | null
  violation_s: number | null
  votes: { without_helmet: number; with_helmet: number }
}

export interface Tracks {
  fps: number
  vid_stride: number
  width: number
  height: number
  /** [1-based source frame index, boxes] for every processed frame */
  frames: [number, TrackBox[]][]
  tracks: Record<string, TrackInfo>
}

export interface Violation {
  id: number
  job_id: string | null
  track_id: number
  video_time_s: number | null
  first_seen_s: number | null
  last_seen_s: number | null
  votes_without_helmet: number
  votes_with_helmet: number
  max_conf: number | null
  plate: string | null
  plate_conf: number | null
  plate_frames: string | null
  review_status: ReviewStatus
  reviewed_by: 'author' | 'uploader' | null
  reviewed_at: string | null
  snapshot_url: string | null
  plate_image_url: string | null
  editable: boolean
}

// ----------------------------------------------------------------------------- requests

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function errorMessage(res: Response) {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join('; ')
  } catch {
    // not JSON
  }
  return `${res.status} ${res.statusText}`
}

/** Identity headers for every request: the browser's owner token, plus the author's key when signed in. */
function identityHeaders(key = authorKey()): Record<string, string> {
  const h: Record<string, string> = { 'X-Owner-Token': ownerToken() }
  if (key) h.Authorization = `Bearer ${key}`
  return h
}

async function request<T>(path: string, init: RequestInit = {}, key?: string | null): Promise<T> {
  const headers = new Headers(init.headers)
  for (const [k, v] of Object.entries(identityHeaders(key === undefined ? authorKey() : key))) headers.set(k, v)
  let res: Response
  try {
    res = await fetch(apiUrl(path), { ...init, headers })
  } catch {
    throw new ApiError(0, 'The API is not reachable. Is it running?')
  }
  if (res.status === 401 && key === undefined && authorKey()) {
    setAuthorKey(null) // the saved author key is no longer valid (changed on the server): continue as a visitor
    window.dispatchEvent(new Event('ridesafe-author-changed'))
    return request<T>(path, init)
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),
  /** Who the API thinks we are; pass a key to check it before saving it. */
  me: (key?: string | null) => request<Me>('/me', {}, key),
  model: () => request<ModelInfo>('/model'),
  jobs: () => request<Job[]>('/jobs'),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  tracks: (id: string) => request<Tracks>(`/jobs/${id}/tracks`),
  deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: 'DELETE' }),
  violations: (params: { job_id?: string; review_status?: ReviewStatus } = {}) => {
    const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v) as [string, string][])
    return request<Violation[]>(`/violations${q.size ? `?${q}` : ''}`)
  },
  review: (id: number, review_status: ReviewStatus) =>
    request<Violation>(`/violations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status }),
    }),
  detect: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<DetectResponse>('/detect', { method: 'POST', body: form })
  },
}

/** Upload a video with progress events (fetch cannot report upload progress). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void, signal?: AbortSignal) {
  return new Promise<{ job: Job }>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', apiUrl('/jobs/video'))
    for (const [k, v] of Object.entries(identityHeaders())) xhr.setRequestHeader(k, v)
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onload = () => {
      let body: { detail?: string; job?: Job } = {}
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        // not JSON
      }
      if (xhr.status === 202 && body.job) resolve({ job: body.job })
      else reject(new ApiError(xhr.status, body.detail ?? `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new ApiError(0, 'The API is not reachable. Is it running?'))
    xhr.onabort = () => reject(new ApiError(0, 'Upload cancelled'))
    signal?.addEventListener('abort', () => xhr.abort())
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

export const jobVideoUrl = (id: string) => apiUrl(`/jobs/${id}/video`)
