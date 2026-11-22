import { Upload } from 'lucide-react'
import { useRef, useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Dropzone({
  accept,
  onFile,
  title,
  hint,
  disabled,
  icon,
}: {
  accept: string
  onFile: (file: File) => void
  title: ReactNode
  hint: ReactNode
  disabled?: boolean
  icon?: ReactNode
}) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const pick = (files: FileList | null) => {
    const f = files?.[0]
    if (f) onFile(f)
  }
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        if (!disabled) pick(e.dataTransfer.files)
      }}
      className={cn(
        'group relative flex w-full cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border border-dashed border-border-strong bg-panel/60 px-6 py-12 text-center transition-colors hover:border-accent/60 hover:bg-panel disabled:cursor-not-allowed disabled:opacity-50',
        over && 'border-accent bg-accent/5',
      )}
    >
      <div className="grid size-12 place-items-center rounded-xl border border-border bg-panel-2 text-accent transition-transform group-hover:-translate-y-0.5 [&_svg]:size-5">
        {icon ?? <Upload />}
      </div>
      <div className="text-sm font-medium">{title}</div>
      <div className="text-xs text-muted">{hint}</div>
      <input
        ref={input}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          pick(e.target.files)
          e.target.value = ''
        }}
      />
    </button>
  )
}
