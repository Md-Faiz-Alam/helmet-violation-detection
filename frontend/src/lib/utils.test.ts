import { formatBytes, formatSeconds, formatTime } from './utils'

describe('formatters', () => {
  it('formats durations', () => {
    expect(formatTime(0)).toBe('0:00')
    expect(formatTime(83.4)).toBe('1:23')
    expect(formatTime(3725)).toBe('1:02:05')
    expect(formatTime(null)).toBe('–')
  })
  it('formats positions and sizes', () => {
    expect(formatSeconds(12.44)).toBe('12.4 s')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(5 * 1024 ** 2)).toBe('5.0 MB')
  })
})
