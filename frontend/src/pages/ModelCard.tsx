import { ExternalLink } from 'lucide-react'
import type { ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PageHeader } from '@/components/layout/AppShell'
import { Badge, Card, CardHeader, Eyebrow, Stat } from '@/components/ui/primitives'
import { REPO_URL, headline, headwear, limitations, models, reportUrl, runtime, videoClips } from '@/data/evaluation'
import { useModelInfo } from '@/lib/hooks'
import { cn } from '@/lib/utils'

const axis = { stroke: 'var(--subtle)', fontSize: 11, tickLine: false, axisLine: false }
const tooltip = {
  contentStyle: { background: 'var(--panel)', border: '1px solid var(--border-strong)', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: 'var(--fg)', fontWeight: 600 },
  itemStyle: { color: 'var(--muted)' },
  cursor: { fill: 'var(--panel-2)' },
}

function Section({ title, description, source, children }: { title: string; description?: ReactNode; source?: string; children: ReactNode }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title={title}
        description={description}
        action={
          source && (
            <a href={reportUrl(source)} target="_blank" rel="noreferrer" className="flex shrink-0 items-center gap-1 text-xs text-muted hover:text-accent">
              Report <ExternalLink className="size-3" />
            </a>
          )
        }
      />
      <div className="p-5">{children}</div>
    </Card>
  )
}

export default function ModelCard() {
  const model = useModelInfo()
  return (
    <>
      <PageHeader eyebrow="Model card" title="What the model is, how it was tested, where it fails">
        Every number here comes from a generated report in the repository, measured on data that was never used for
        training or for choosing the model.
      </PageHeader>

      <div className="space-y-5">
        <Card className="grid gap-6 p-6 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Detector" value="YOLO26s" caption="9.9M parameters, 2 classes: helmet / no helmet" />
          <Stat
            label="Runtime"
            value="ONNX · CPU"
            caption={model.data ? `Input ${model.data.input_size} px (dynamic), ${model.data.runtime}` : 'ONNX Runtime, no PyTorch'}
          />
          <Stat
            label="No-helmet threshold"
            value={model.data?.thresholds['Without Helmet'].toFixed(2) ?? '0.31'}
            caption="Tuned on validation for F2 (recall weighted 2×)"
            tone="accent"
          />
          <Stat label="Decision rule" value="≥5 · 60%" caption="No-helmet frames and share of a rider's votes" />
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Section
            title="Choosing the model: 9 experiments"
            description="Chosen on validation only. Bars: validation mAP50 and validation no-helmet recall."
            source="model_comparison.md"
          >
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={models} layout="vertical" margin={{ left: 10, right: 30 }} barGap={2}>
                  <CartesianGrid horizontal={false} stroke="var(--border)" />
                  <XAxis type="number" domain={[0, 1]} {...axis} />
                  <YAxis type="category" dataKey="short" width={110} {...axis} />
                  <Tooltip {...tooltip} formatter={(v) => Number(v).toFixed(3)} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="valMap50" name="val mAP50" fill="var(--accent)" radius={[0, 3, 3, 0]} />
                  <Bar dataKey="valRecall" name="val no-helmet recall" fill="var(--violation)" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <Section
            title="The headwear fix"
            description="Held-out test of 635 images: caps, turbans, hijabs, masks, rear views. Before vs after adding 2,537 mined images."
            source="model_comparison_headwear.md"
          >
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={headwear} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="metric" {...axis} />
                  <YAxis domain={[0, 1]} {...axis} />
                  <Tooltip {...tooltip} formatter={(v) => Number(v).toFixed(3)} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="before" name="Before" fill="var(--subtle)" radius={[3, 3, 0, 0]}>
                    <LabelList dataKey="before" position="top" fontSize={11} fill="var(--muted)" formatter={(v) => Number(v).toFixed(2)} />
                  </Bar>
                  <Bar dataKey="after" name="After" fill="var(--helmet)" radius={[3, 3, 0, 0]}>
                    <LabelList dataKey="after" position="top" fontSize={11} fill="var(--fg)" formatter={(v) => Number(v).toFixed(2)} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>
        </div>

        <Section title="All candidates" description="Test columns are the unbiased estimate, reported once." source="model_comparison.md">
          <div className="-mx-5 overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted">
                  {['Model', 'Params (M)', 'PyTorch CPU ms', 'val mAP50', 'val recall', 'test mAP50', 'test recall'].map((h, i) => (
                    <th key={h} className={cn('px-5 py-2 font-medium', i > 0 && 'text-right')}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono tabular">
                {models.map((m) => (
                  <tr key={m.name} className={cn('border-b border-border last:border-0', m.chosen && 'bg-accent/5')}>
                    <td className="px-5 py-2 font-sans">
                      {m.name} {m.chosen && <Badge tone="accent" className="ml-1">deployed</Badge>}
                    </td>
                    <td className="px-5 py-2 text-right">{m.params}</td>
                    <td className="px-5 py-2 text-right">{m.cpuMs}</td>
                    <td className="px-5 py-2 text-right">{m.valMap50.toFixed(3)}</td>
                    <td className="px-5 py-2 text-right">{m.valRecall.toFixed(3)}</td>
                    <td className="px-5 py-2 text-right">{m.testMap50.toFixed(3)}</td>
                    <td className="px-5 py-2 text-right">{m.testRecall.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <div className="grid gap-5 xl:grid-cols-2">
          <Section
            title="Video benchmark: per rider"
            description={`4 hand-labelled test clips. ${headline.videoRecall.hit}/${headline.videoRecall.total} violators caught at ${Math.round((headline.videoPrecision.hit / headline.videoPrecision.total) * 100)}% precision; no rider wearing a helmet was flagged.`}
            source="video_eval.md"
          >
            <ul className="space-y-4">
              {videoClips.map((c) => (
                <li key={c.clip}>
                  <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
                    <span>
                      {c.clip} <span className="text-xs text-muted">· {c.conditions}</span>
                    </span>
                    <span className="font-mono text-xs text-muted tabular">{c.violators ? `${c.caught}/${c.violators}` : 'no violators'}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-panel-2">
                    <div className="h-full rounded-full bg-violation" style={{ width: c.violators ? `${(c.caught / c.violators) * 100}%` : 0 }} />
                  </div>
                </li>
              ))}
            </ul>
          </Section>

          <Section
            title="Runtime: PyTorch → ONNX Runtime"
            description="Same decisions on every test clip (the per-frame tracks are identical); ms per image or frame on an i7-7700K CPU."
            source="runtime_parity.md"
          >
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={runtime} layout="vertical" margin={{ left: 10, right: 40 }}>
                  <CartesianGrid horizontal={false} stroke="var(--border)" />
                  <XAxis type="number" {...axis} unit=" ms" />
                  <YAxis type="category" dataKey="label" width={150} {...axis} />
                  <Tooltip {...tooltip} formatter={(v) => `${v} ms`} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="pytorch" name="PyTorch" fill="var(--subtle)" radius={[0, 3, 3, 0]}>
                    <LabelList dataKey="pytorch" position="right" fontSize={11} fill="var(--muted)" />
                  </Bar>
                  <Bar dataKey="onnx" name="ONNX Runtime" fill="var(--accent)" radius={[0, 3, 3, 0]}>
                    <LabelList dataKey="onnx" position="right" fontSize={11} fill="var(--fg)" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 text-xs text-muted">
              API image without PyTorch: {headline.imageMb} MB vs {(headline.imageMbTorch / 1000).toFixed(2)} GB with it.
            </p>
          </Section>
        </div>

        <div className="grid gap-5 xl:grid-cols-3">
          <Section title="Known limitations">
            <ul className="space-y-2.5 text-sm leading-relaxed text-muted">
              {limitations.map((l) => (
                <li key={l} className="flex gap-2.5">
                  <span className="mt-2 size-1.5 shrink-0 rounded-full bg-uncertain" />
                  {l}
                </li>
              ))}
            </ul>
          </Section>
          <Section title="Data">
            <div className="space-y-3 text-sm leading-relaxed text-muted">
              <p>
                <span className="text-fg">Bike Helmet Detection</span> (Roboflow Universe, public domain): the original
                train / validation / test splits. Violator images repeated 2× against class imbalance.
              </p>
              <p>
                <span className="text-fg">Helmet, Rider, ANPR Detection v12</span> (CC BY 4.0): 2,537 images of headwear and
                rear views mined from it. 231 images that duplicated our own test data were removed first.
              </p>
              <p>
                <span className="text-fg">Video benchmark:</span> 8 Pexels clips (4 test, 4 validation); every rider in the
                4 test clips was labelled by hand and reviewed.
              </p>
            </div>
          </Section>
          <Section title="Privacy and responsible use">
            <div className="space-y-3 text-sm leading-relaxed text-muted">
              <p>
                The system <span className="text-fg">flags</span>; a person decides. Nothing is enforced without review.
              </p>
              <p>
                In the public demo, uploads are deleted after an hour, visitors see only their own uploads, and number
                plates are partly masked.
              </p>
              <p>
                Turbans are labelled "no helmet", which is visually true; Indian law exempts turban-wearing Sikhs, so that is
                a deployment policy, not a model decision.
              </p>
              <a href={REPO_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent hover:underline">
                Source code (AGPL-3.0) <ExternalLink className="size-3" />
              </a>
            </div>
          </Section>
        </div>
        <Eyebrow className="block text-center">Model card · RideSafe AI</Eyebrow>
      </div>
    </>
  )
}
