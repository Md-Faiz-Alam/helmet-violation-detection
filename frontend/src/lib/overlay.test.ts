import { frameAt } from './overlay'

// Processed frames of a 10 FPS video analysed every 2nd frame: source frames 1, 3, 5 -> 0.0, 0.2, 0.4 s
const times = [0, 0.2, 0.4]
const maxGap = (2 + 0.5) / 10

describe('frameAt', () => {
  it('shows the last processed frame at or before the playback time', () => {
    expect(frameAt(times, 0, maxGap)).toBe(0)
    expect(frameAt(times, 0.19, maxGap)).toBe(0)
    expect(frameAt(times, 0.2, maxGap)).toBe(1)
    expect(frameAt(times, 0.45, maxGap)).toBe(2)
  })

  it('shows nothing before the first frame or long after the last one', () => {
    expect(frameAt(times, -0.1, maxGap)).toBe(-1)
    expect(frameAt(times, 0.4 + maxGap + 0.01, maxGap)).toBe(-1)
    expect(frameAt([], 1, maxGap)).toBe(-1)
  })
})
