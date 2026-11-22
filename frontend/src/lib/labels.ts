import type { Tone } from '@/components/ui/primitives'
import type { ReviewStatus, TrackLabel } from './api'

export const labelTone: Record<TrackLabel, Tone> = {
  violation: 'violation',
  uncertain: 'uncertain',
  helmet: 'helmet',
  pending: 'neutral',
}

export const labelText: Record<TrackLabel, string> = {
  violation: 'No helmet',
  uncertain: 'Uncertain',
  helmet: 'Helmet',
  pending: 'Deciding',
}

export const reviewTone: Record<ReviewStatus, Tone> = { pending: 'neutral', confirmed: 'violation', rejected: 'helmet' }
export const reviewText: Record<ReviewStatus, string> = {
  pending: 'Awaiting review',
  confirmed: 'Confirmed',
  rejected: 'Rejected',
}
