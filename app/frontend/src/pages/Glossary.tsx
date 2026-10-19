import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { GLOSSARY, GLOSSARY_KEYS } from '../glossary'
import { Detailed } from '../mode'
import { usePageTitle } from '../usePageTitle'

export default function Glossary() {
  usePageTitle('Glossary')
  const { hash } = useLocation()
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView()
  }, [hash])
  return (
    <>
      <h1>Glossary</h1>
      <p className="lede">Every technical word used in the app, in plain language. The same definitions appear in the small “?” help buttons.</p>
      <dl className="gloss">
        {GLOSSARY_KEYS.map((k) => (
          <div key={k}>
            <dt id={k}>{GLOSSARY[k].term}</dt>
            <dd>{GLOSSARY[k].plain}</dd>
            <Detailed>{GLOSSARY[k].more && <dd className="small muted" style={{ marginTop: 4 }}>Technical: {GLOSSARY[k].more}</dd>}</Detailed>
          </div>
        ))}
      </dl>
    </>
  )
}
