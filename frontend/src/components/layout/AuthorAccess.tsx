import { useQueryClient } from '@tanstack/react-query'
import { KeyRound, LogOut, ShieldCheck } from 'lucide-react'
import { useEffect, useState, type FormEvent } from 'react'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { setAuthorKey } from '@/lib/author'
import { useMe, useModelInfo } from '@/lib/hooks'

/** Sidebar control: the author signs in with their key to see and manage every upload. */
export function AuthorAccess() {
  const model = useModelInfo()
  const me = useMe()
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const [key, setKey] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // A stale key dropped by the API client: refresh everything as a visitor
  useEffect(() => {
    const onChange = () => void qc.invalidateQueries()
    window.addEventListener('ridesafe-author-changed', onChange)
    return () => window.removeEventListener('ridesafe-author-changed', onChange)
  }, [qc])

  if (!model.data?.author_enabled) return null
  const isAuthor = me.data?.role === 'author'

  const signIn = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const who = await api.me(key.trim())
      if (who.role !== 'author') throw new Error('Not an author key')
      setAuthorKey(key.trim())
      setKey('')
      setOpen(false)
      await qc.invalidateQueries()
    } catch {
      setError('That key was not accepted.')
    } finally {
      setBusy(false)
    }
  }

  const signOut = async () => {
    setAuthorKey(null)
    await qc.invalidateQueries()
  }

  if (isAuthor) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-accent">
          <ShieldCheck className="size-4" /> Signed in as the author
        </div>
        <Button variant="ghost" size="sm" className="w-full justify-start px-2" onClick={signOut}>
          <LogOut /> Sign out
        </Button>
      </div>
    )
  }

  if (!open) {
    return (
      <Button variant="ghost" size="sm" className="w-full justify-start px-2" onClick={() => setOpen(true)}>
        <KeyRound /> Author sign-in
      </Button>
    )
  }

  return (
    <form onSubmit={signIn} className="space-y-2">
      <label className="block text-xs text-muted" htmlFor="author-key">
        Author key
      </label>
      <input
        id="author-key"
        type="password"
        autoComplete="current-password"
        autoFocus
        value={key}
        onChange={(e) => setKey(e.target.value)}
        className="h-8 w-full rounded-md border border-border-strong bg-panel-2 px-2 font-mono text-xs text-fg outline-none focus:border-accent"
      />
      {error && <p className="text-xs text-violation">{error}</p>}
      <div className="flex gap-2">
        <Button type="submit" variant="primary" size="sm" disabled={busy || key.trim().length < 32} className="flex-1">
          Sign in
        </Button>
        <Button size="sm" variant="ghost" onClick={() => (setOpen(false), setError(null), setKey(''))}>
          Cancel
        </Button>
      </div>
      <p className="text-[11px] leading-relaxed text-subtle">Kept only until this tab closes.</p>
    </form>
  )
}
