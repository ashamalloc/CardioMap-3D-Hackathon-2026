import React, { useMemo } from 'react'

const groupFor = (name) => {
  const s = name.toLowerCase()
  if (/(age|sex|weight|height|bmi)/.test(s)) return 'Demographic'
  if (/(bp|pulse|pressure|dm|diabetes|hypertension|smoker|family|obesity|renal|cva|airway|thyroid|chest|angina|pain|fh)/.test(s)) return 'Clinical examination'
  if (/(st|ecg|t inversion|lvh|rhythm|q wave|progression)/.test(s)) return 'ECG'
  if (/(fbs|creat|cr|tg|ldl|hdl|bun|esr|hb|potassium|sodium|wbc|lymph|neut|plt|chol)/.test(s)) return 'Laboratory'
  if (/(ef|rwma|vhd|echo)/.test(s)) return 'Echocardiography'
  return 'Other clinical'
}

export default function ClinicalForm({ schema, values, setValues }) {
  const groups = useMemo(() => {
    const g = {}
    for (const f of schema?.fields || []) {
      const group = groupFor(f.name)
      if (!g[group]) g[group] = []
      g[group].push(f)
    }
    return g
  }, [schema])

  const update = (name, value) => setValues(v => ({ ...v, [name]: value }))
  return <div className="form-wrap">
    {Object.entries(groups).map(([group, fields]) => <section key={group} className="form-section">
      <div className="section-label">{group}</div>
      <div className="field-grid">
        {fields.map(field => <label key={field.name} className="field">
          <span>{field.name}</span>
          {field.kind === 'select' ? (
            <select value={values[field.name] ?? field.default ?? ''} onChange={e => update(field.name, e.target.value)}>
              {(field.options || []).map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          ) : (
            <input type="number" value={values[field.name] ?? field.default ?? ''} min={field.min ?? undefined} max={field.max ?? undefined} step={field.step ?? 'any'} onChange={e => update(field.name, e.target.value)} />
          )}
        </label>)}
      </div>
    </section>)}
  </div>
}
