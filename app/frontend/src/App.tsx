import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Console from './pages/Console'
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
      <p className="lede">There is no page at this address. Use the navigation above.</p>
    </>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Console />} />
        <Route path="how-it-works" element={<HowItWorks />} />
        <Route path="proofs" element={<Proofs />} />
        <Route path="results" element={<Results />} />
        <Route path="connect" element={<ConnectApp />} />
        <Route path="testbed" element={<Testbed />} />
        <Route path="limits" element={<Limits />} />
        <Route path="glossary" element={<Glossary />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
