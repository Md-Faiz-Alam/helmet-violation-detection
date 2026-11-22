import { everyNth, formatBytes, formatSeconds, formatTime } from './utils'

describe('formatters', () => {
  it('formats durations', () => {
    expect(formatTime(0)).toBe('0:00')
    expect(formatTime(83.4)).toBe('1:23')
    expect(formatTime(3725)).toBe('1:02:05')
    expect(formatTime(null)).toBe('–')
  })
  it('writes frame strides as words', () => {
    expect(everyNth(1)).toBe('every frame')
    expect(everyNth(2)).toBe('every 2nd frame')
    expect(everyNth(3)).toBe('every 3rd frame')
    expect(everyNth(11)).toBe('every 11th frame')
  })
  it('formats positions and sizes', () => {
    expect(formatSeconds(12.44)).toBe('12.4 s')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(5 * 1024 ** 2)).toBe('5.0 MB')
  })
})
