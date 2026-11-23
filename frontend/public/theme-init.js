// Apply the saved theme before the first paint (no flash of the wrong theme). A file rather than an inline
// script, so the Content-Security-Policy can forbid inline scripts.
try {
  if (localStorage.getItem('ridesafe-theme') === 'light') document.documentElement.classList.remove('dark')
} catch {}
