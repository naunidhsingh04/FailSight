import type { Lo2Response, Missing } from '../types'
import { GROUP_LABEL, pct } from '../format'

const TIER: Record<Missing['reliability'], { badge: string; text: string }> = {
  normal: { badge: 'accent', text: 'Most expected features are present.' },
  degraded: { badge: '', text: 'A noticeable share of expected features is missing. The model imputes them with training medians, so treat the verdict with care.' },
  unreliable: { badge: 'solid', text: 'Most expected features are missing. Do not rely on this verdict.' },
}

export default function MissingPanel({ m, lo2 }: { m: Missing; lo2?: Lo2Response | null }) {
  const t = TIER[m.reliability]
  const a = lo2?.feature_mapping?.A_logs_only
  const evidence = m.reliability === 'unreliable' && a && lo2
    ? ` Evidence: in the LO2 external test only ${a.used_features_built} of ${a.used_features_total} model features could be built, and every saved model scored balanced accuracy ${Math.max(...lo2.models.map((x) => x.balanced_accuracy)).toFixed(3)}.`
    : ''
  return (
    <section aria-labelledby="missing-h" className={m.reliability === 'normal' ? 'card' : 'callout'}>
      <h3 id="missing-h" style={{ marginTop: 0 }}>
        Input completeness <span className={`badge ${t.badge}`}>{m.reliability}</span>
      </h3>
      <p>
        {m.missing_features} of {m.expected_features} features this system should have are missing ({pct(m.missing_fraction, 0)}). {t.text}{evidence}
      </p>
      <div className="table-wrap">
        <table className="rt">
          <caption className="visually-hidden">Missing features by group</caption>
          <thead><tr><th scope="col">Group</th><th scope="col">Expected</th><th scope="col">Missing</th><th scope="col">Share missing</th></tr></thead>
          <tbody>
            {Object.entries(m.by_group).map(([g, v]) => (
              <tr key={g}>
                <td data-label="Group" className="name-cell">{GROUP_LABEL[g] ?? g}</td>
                <td data-label="Expected" className="num-cell">{v.expected}</td>
                <td data-label="Missing" className="num-cell">{v.missing}</td>
                <td data-label="Share missing" className="num-cell">{v.expected ? pct(v.missing / v.expected, 0) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 'var(--space-3)' }}>
        {m.structurally_absent_for_system} further features exist only on the other system and are set to 0 by design; they are not counted as missing. Reliability bands (10% / 50%) are heuristics, not measured on held-out data.
      </p>
    </section>
  )
}
