import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { createBrowserRouter, Link, RouterProvider } from 'react-router'
import { AppShell } from '@/components/layout/AppShell'
import { buttonVariants } from '@/components/ui/button-variants'
import { Skeleton } from '@/components/ui/primitives'

const Overview = lazy(() => import('@/pages/Overview'))
const Analyze = lazy(() => import('@/pages/Analyze'))
const JobPage = lazy(() => import('@/pages/JobPage'))
const Review = lazy(() => import('@/pages/Review'))
const ModelCard = lazy(() => import('@/pages/ModelCard'))

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

const page = (el: React.ReactNode) => <Suspense fallback={<Skeleton className="h-96 w-full" />}>{el}</Suspense>

function NotFound() {
  return (
    <div className="flex flex-col items-start gap-4 py-16">
      <div className="font-mono text-sm text-accent">404</div>
      <h1 className="text-2xl font-semibold">This page does not exist</h1>
      <Link to="/" className={buttonVariants({ variant: 'secondary' })}>
        Back to the overview
      </Link>
    </div>
  )
}

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: '/', element: page(<Overview />) },
      { path: '/analyze', element: page(<Analyze />) },
      { path: '/analyze/jobs/:id', element: page(<JobPage />) },
      { path: '/review', element: page(<Review />) },
      { path: '/model', element: page(<ModelCard />) },
      { path: '*', element: <NotFound /> },
    ],
  },
])

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  )
}
