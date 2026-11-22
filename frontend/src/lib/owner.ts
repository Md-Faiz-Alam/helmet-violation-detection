/**
 * One random owner token per browser. It is sent as X-Owner-Token with every request, so the API knows
 * which uploads are this visitor's (in the public demo, visitors see and review only their own uploads).
 * It is not a login: clearing site data starts a new identity.
 */
const KEY = 'ridesafe-owner-token'
let memoryToken: string | null = null

function newToken() {
  const bytes = new Uint8Array(24)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export function ownerToken(): string {
  if (memoryToken) return memoryToken
  try {
    const saved = localStorage.getItem(KEY)
    if (saved && saved.length >= 16) return (memoryToken = saved)
    memoryToken = newToken()
    localStorage.setItem(KEY, memoryToken)
  } catch {
    memoryToken ??= newToken() // storage blocked (private mode): keep the token for this tab only
  }
  return memoryToken
}
