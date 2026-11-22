import { cva } from 'class-variance-authority'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-45 [&_svg]:size-4 [&_svg]:shrink-0 cursor-pointer',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-accent-fg hover:brightness-110',
        secondary: 'border border-border-strong bg-panel-2 text-fg hover:bg-border',
        ghost: 'text-muted hover:bg-panel-2 hover:text-fg',
        helmet: 'border border-helmet/40 bg-helmet/12 text-helmet hover:bg-helmet/20',
        violation: 'border border-violation/40 bg-violation/12 text-violation hover:bg-violation/20',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-9 px-4',
        lg: 'h-11 px-5 text-base',
        icon: 'size-9',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  },
)
