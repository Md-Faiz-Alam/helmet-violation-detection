/**
 * Numbers shown on the site, copied from the generated reports (update them when a report changes):
 * reports/model_comparison.md, reports/model_comparison_headwear.md, reports/video_eval.md,
 * reports/onnx_benchmark.md, reports/runtime_parity.md and README.md (Results).
 */

export const REPO_URL = 'https://github.com/Md-Faiz-Alam/helmet-violation-detection'
export const reportUrl = (file: string) => `${REPO_URL}/blob/main/reports/${file}`

export const headline = {
  videoRecall: { hit: 18, total: 40 },
  videoPrecision: { hit: 18, total: 24 },
  helmetedRidersFlagged: 0,
  testMap50: 0.937,
  headwearMap50: 0.81,
  headwearMap50Before: 0.32,
  cpuFpsVideo: 16,
  imageMb: 620,
  imageMbTorch: 2270,
}

/** The 9 trained candidates (val = used for choosing, test = reported once). */
export const models = [
  { name: 'YOLO26s + oversampling + headwear', short: 'Y26s+OS+HW', params: 9.9, cpuMs: 108, valMap50: 0.918, valRecall: 0.887, testMap50: 0.937, testRecall: 0.852, chosen: true },
  { name: 'YOLO26s + oversampling + headwear, 100 epochs', short: 'Y26s+OS+HW 100ep', params: 9.9, cpuMs: 108, valMap50: 0.916, valRecall: 0.896, testMap50: 0.933, testRecall: 0.934 },
  { name: 'YOLO26s + oversampling', short: 'Y26s+OS', params: 9.9, cpuMs: 108, valMap50: 0.925, valRecall: 0.843, testMap50: 0.917, testRecall: 0.869 },
  { name: 'YOLO26s', short: 'Y26s', params: 9.9, cpuMs: 106, valMap50: 0.893, valRecall: 0.809, testMap50: 0.915, testRecall: 0.885 },
  { name: 'YOLO11s', short: 'Y11s', params: 9.4, cpuMs: 103, valMap50: 0.881, valRecall: 0.774, testMap50: 0.914, testRecall: 0.869 },
  { name: 'YOLOv8s @ 416 px', short: 'v8s@416', params: 11.1, cpuMs: 55, valMap50: 0.859, valRecall: 0.73, testMap50: 0.893, testRecall: 0.77 },
  { name: 'YOLOv8n', short: 'v8n', params: 3.1, cpuMs: 49, valMap50: 0.854, valRecall: 0.774, testMap50: 0.871, testRecall: 0.803 },
  { name: 'YOLO26n', short: 'Y26n', params: 2.5, cpuMs: 51, valMap50: 0.824, valRecall: 0.713, testMap50: 0.909, testRecall: 0.787 },
  { name: 'YOLOv8s (baseline recipe)', short: 'v8s base', params: 11.1, cpuMs: 110, valMap50: 0.818, valRecall: 0.67, testMap50: 0.869, testRecall: 0.77 },
]

/** Held-out headwear test (635 images: caps, turbans, hijabs, masks, rear views). */
export const headwear = [
  { metric: 'mAP50', before: 0.32, after: 0.81 },
  { metric: 'No-helmet AP50', before: 0.31, after: 0.781 },
  { metric: 'No-helmet recall', before: 0.285, after: 0.763 },
]

/** Video benchmark: 4 hand-labelled test clips, scored per rider. */
export const videoClips = [
  { clip: 'Highway riders', conditions: 'Daylight, close traffic', caught: 8, violators: 12 },
  { clip: 'Pune junction', conditions: 'Dense junction, 1080p', caught: 9, violators: 26 },
  { clip: 'Night fog intersection', conditions: 'Night, fog', caught: 1, violators: 2 },
  { clip: 'High-angle junction', conditions: 'CCTV-like angle', caught: 0, violators: 0 },
]

export const runtime = [
  { label: 'Image, 640x640', pytorch: 121, onnx: 97 },
  { label: 'Video frame, 1280x720 / 1080p', pytorch: 79, onnx: 52 },
]

export const limitations = [
  'Far-away riders (heads under about 36 px) are not tracked, so they are never judged.',
  'People standing among riders can be logged as violators: all 6 false logs on the test clips were non-riders.',
  'Headscarves, face scarves and woolly caps are still sometimes read as helmets on video.',
  'Plate reading is best effort: look-alike digits (4 read as 2) and the plate of a closer bike can be returned.',
  'One rider can be logged twice after a long occlusion (7 duplicate logs on the test clips).',
  'Night, rain and CCTV angles are thinly covered; the video benchmark is 4 short clips, so percentages have wide error bars.',
]
