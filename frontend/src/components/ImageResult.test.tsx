import { render, screen } from '@testing-library/react'
import { ImageResult } from './ImageResult'

describe('ImageResult', () => {
  it('draws one labelled box per detection in image coordinates', () => {
    const { container } = render(
      <ImageResult
        src="data:image/png;base64,"
        result={{
          width: 640,
          height: 480,
          latency_ms: 42,
          with_helmet: 1,
          without_helmet: 1,
          detections: [
            { cls: 'Without Helmet', conf: 0.81, box: [10, 20, 60, 80] },
            { cls: 'With Helmet', conf: 0.66, box: [200, 40, 260, 110] },
          ],
        }}
      />,
    )
    expect(container.querySelector('svg')?.getAttribute('viewBox')).toBe('0 0 640 480')
    expect(screen.getByText('No helmet 81%')).toBeInTheDocument()
    expect(screen.getByText('Helmet 66%')).toBeInTheDocument()
    expect(screen.getByText(/640×480 · 42 ms on CPU/)).toBeInTheDocument()
  })
})
