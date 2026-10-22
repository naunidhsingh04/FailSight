import { Navigate, Route, Routes, useSearchParams } from 'react-router-dom'
import Layout from './components/Layout'
import Console from './pages/Console'
import LiveMonitor from './pages/LiveMonitor'
import Examiners, { ExaminerLayout } from './pages/Examiners'
import HowItWorks from './pages/HowItWorks'
import Proofs from './pages/Proofs'
import Results from './pages/Results'
import Limits from './pages/Limits'
import Testbed from './pages/Testbed'
import ConnectApp from './pages/ConnectApp'
import Glossary from './pages/Glossary'
import { usePageTitle } from './usePageTitle'

function NotFound() {
  usePageTitle('Not found')
  return (
    <>
      <h1>Page not found</h1>
      <p className="lede">There is no page at this address. Use the menu at the top.</p>
    </>
  )
}

/** The home page is the live monitor. The old address "/?tab=replay" (recorded data) now lives under For examiners; "/?tab=live" is the live monitor itself. */
function Home() {
  const [params] = useSearchParams()
  if (params.get('tab') === 'replay') return <Navigate to="/examiners/replay" replace />
  return <LiveMonitor />
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="how-it-works" element={<HowItWorks />} />
        <Route path="proofs" element={<Proofs />} />
        <Route element={<ExaminerLayout />}>
          <Route path="examiners" element={<Examiners />} />
          <Route path="examiners/replay" element={<Console />} />
          <Route path="examiners/results" element={<Results />} />
          <Route path="examiners/connect" element={<ConnectApp />} />
          <Route path="examiners/testbed" element={<Testbed />} />
          <Route path="examiners/limits" element={<Limits />} />
          <Route path="examiners/glossary" element={<Glossary />} />
          {/* the addresses used before the redesign keep working */}
          <Route path="console" element={<Console />} />
          <Route path="results" element={<Results />} />
          <Route path="connect" element={<ConnectApp />} />
          <Route path="testbed" element={<Testbed />} />
          <Route path="limits" element={<Limits />} />
          <Route path="glossary" element={<Glossary />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
