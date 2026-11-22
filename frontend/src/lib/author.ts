/**
 * The author's key (RIDESAFE_ADMIN_TOKEN on the API). Kept in sessionStorage: it is forgotten when the tab
 * closes, so it is not left on a shared device. Sent as "Authorization: Bearer <key>" with every request.
 */
const KEY = 'ridesafe-author-key'
let memoryKey: string | null = null

export function authorKey(): string | null {
  if (memoryKey) return memoryKey
  try {
    memoryKey = sessionStorage.getItem(KEY)
  } catch {
    // storage blocked: the key lives in memory only
  }
  return memoryKey
}

export function setAuthorKey(key: string | null) {
  memoryKey = key
  try {
    if (key) sessionStorage.setItem(KEY, key)
    else sessionStorage.removeItem(KEY)
  } catch {
    // storage blocked
  }
}
