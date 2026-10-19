import type { Label, ModelInfo, Sample } from '../types'
import { LABEL_NAME } from '../types'
import { FAMILY_LABEL } from '../format'

export function LabelToggle({ value, onChange }: { value: Label; onChange: (l: Label) => void }) {
  return (
    <div className="field">
      <span className="label" id="label-toggle">Label version</span>
      <div className="segmented" role="group" aria-labelledby="label-toggle">
        {(['label_binary', 'label_level'] as Label[]).map((l) => (
          <button key={l} type="button" aria-pressed={value === l} onClick={() => onChange(l)} title={LABEL_NAME[l]}>
            {l === 'label_binary' ? 'Binary' : 'Five-class'}
          </button>
        ))}
      </div>
    </div>
  )
}

export function ModelSelect({ id, models, value, onChange, onlyExplainable = false }: { id: string; models: ModelInfo[]; value: string; onChange: (v: string) => void; onlyExplainable?: boolean }) {
  const list = onlyExplainable ? models.filter((m) => m.supports_explain) : models
  return (
    <div className="field">
      <label htmlFor={id}>Model</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {list.map((m) => (
          <option key={m.name} value={m.name} disabled={!m.available}>
            {m.name} ({FAMILY_LABEL[m.family]}{m.corrected_variant ? ', corrected' : ''}){m.available ? '' : ' - unavailable'}
          </option>
        ))}
      </select>
    </div>
  )
}

export function SampleSelect({ id, samples, value, onChange }: { id: string; samples: Sample[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="field">
      <label htmlFor={id}>Real test window</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {samples.map((s) => (
          <option key={s.id} value={s.id}>
            {s.id.replace('sample-', '#')} &middot; {s.system} &middot; true: {s.label_level}
          </option>
        ))}
      </select>
    </div>
  )
}
