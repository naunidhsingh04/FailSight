import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import Predict from './pages/Predict'
import Models from './pages/Models'
import Explain from './pages/Explain'
import DataMethod from './pages/DataMethod'
import Limits from './pages/Limits'
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
        <Route index element={<Overview />} />
        <Route path="predict" element={<Predict />} />
        <Route path="models" element={<Models />} />
        <Route path="explain" element={<Explain />} />
        <Route path="data" element={<DataMethod />} />
        <Route path="limits" element={<Limits />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
