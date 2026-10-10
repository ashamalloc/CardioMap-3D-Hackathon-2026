import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { getMetrics, getSchema, getStatus, predict } from './api'
import ClinicalForm from './components/ClinicalForm'
import HeartViewer from './components/HeartViewer'
import MetricPanel from './components/MetricPanel'

const vesselNames = { LAD: 'Left Anterior Descending', LCX: 'Left Circumflex', RCA: 'Right Coronary Artery' }

export default function App() {
  const [status, setStatus] = useState(null)
  const [schema, setSchema] = useState({ fields: [] })
  const [metrics, setMetrics] = useState({})
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [selected, setSelected] = useState('LAD')
  const [tab, setTab] = useState('dashboard')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getStatus(), getSchema(), getMetrics()])
      .then(([s, sc, m]) => {
        setStatus(s)
        setSchema(sc)
        setMetrics(m)
        const defaults = {}
        for (const f of sc.fields || []) defaults[f.name] = f.default
        setValues(defaults)
      })
      .catch(e => setError(e.message))
  }, [])

  const runPrediction = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await predict(values)
      setResult(data)
      setSelected(data.highest_vessel)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [values])

  const vesselProbabilities = useMemo(() => result ? ({
    LAD: result.vessels.LAD.probability,
    LCX: result.vessels.LCX.probability,
    RCA: result.vessels.RCA.probability,
  }) : ({}), [result])

  const selectedResult = result?.vessels?.[selected]
  const statusReady = status?.ready

  return <div className="app-shell">
    <header className="topbar">
      <div>
        <div className="brand">Cardio<span>Map</span> 3D</div>
        <div className="subbrand">Multimodal AI Hackathon 2026 · Track A</div>
      </div>
      <div className={`status-pill ${statusReady ? 'ready' : 'not-ready'}`}>
        <i></i>{statusReady ? 'Models ready' : 'Train models first'}
      </div>
    </header>

    <div className="safety-banner">
      <strong>Clinical safety:</strong> Educational / decision-support prototype only. Not a medical device and not a substitute for formal diagnostic imaging, angiography or clinician judgment.
    </div>

    <nav className="tabs">
      <button className={tab === 'dashboard' ? 'active' : ''} onClick={() => setTab('dashboard')}>Risk dashboard</button>
      <button className={tab === 'metrics' ? 'active' : ''} onClick={() => setTab('metrics')}>Model evaluation</button>
      <button className={tab === 'about' ? 'active' : ''} onClick={() => setTab('about')}>Method & safety</button>
    </nav>

    {error && <div className="error-banner">{error}</div>}

    {tab === 'dashboard' && <main className="dashboard">
      <section className="panel form-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">PATIENT INPUT</div>
            <h2>Clinical profile</h2>
          </div>
          <span className="counter">{schema.count || 0} supported features</span>
        </div>
        {!statusReady && <div className="notice">Run <code>python -m ml_pipeline.train</code> from the project root. The form can still be reviewed before models are trained.</div>}
        <ClinicalForm schema={schema} values={values} setValues={setValues} />
        <button className="predict-btn" disabled={!statusReady || loading} onClick={runPrediction}>{loading ? 'Running models…' : 'Predict & Map Risk'}</button>
      </section>

      <section className="center-column">
        <div className="panel viewer-panel">
          <div className="panel-heading compact">
            <div><div className="eyebrow">ANATOMICAL RISK MAP</div><h2>Heart + coronary vessels</h2></div>
            <div className="legend"><span><i className="low"></i>Low</span><span><i className="mid"></i>Intermediate</span><span><i className="high"></i>Higher</span></div>
          </div>
          <HeartViewer probabilities={vesselProbabilities} selected={selected} onSelect={setSelected} />
        </div>
        <div className="disclaimer-note">The coronary geometry is a visualization abstraction. Clinical features do not provide direct pixel-level or 3D anatomical lesion coordinates.</div>
      </section>

      <aside className="panel insights-panel">
        <div className="panel-heading compact"><div><div className="eyebrow">MODEL OUTPUT</div><h2>Risk summary</h2></div></div>
        <div className="cad-card">
          <div className="cad-label">Overall CAD likelihood</div>
          <div className="cad-value">{result ? `${result.overall_cad.percent.toFixed(1)}%` : '—'}</div>
          <div className="cad-band">{result?.overall_cad?.band || 'Run prediction'}</div>
        </div>
        <div className="vessel-list">
          {['LAD', 'LCX', 'RCA'].map(v => <button key={v} className={`vessel-card ${selected === v ? 'selected' : ''}`} onClick={() => setSelected(v)}>
            <div className="vessel-top"><span className="vessel-dot" style={{ background: colorFor(result ? vesselProbabilities[v] : null) }}></span><strong>{v}</strong><span className="vessel-pct">{result ? `${(vesselProbabilities[v] * 100).toFixed(1)}%` : '—'}</span></div>
            <div className="vessel-name">{vesselNames[v]}</div>
            <div className="bar"><span style={{ width: `${result ? vesselProbabilities[v] * 100 : 0}%`, background: colorFor(result ? vesselProbabilities[v] : null) }}></span></div>
          </button>)}
        </div>
        <div className="selected-detail">
          <div className="eyebrow">{selected} DETAIL</div>
          <h3>{vesselNames[selected]}</h3>
          {!selectedResult && <p>Run a prediction to see vessel-specific explanations.</p>}
          {selectedResult && <>
            <p className="detail-status"><b>{selectedResult.percent.toFixed(1)}%</b> · {selectedResult.band}</p>
            <div className="explanation-list">
              {(selectedResult.explanations || []).slice(0, 6).map(x => <div className="explain-row" key={x.feature}>
                <span>{x.feature}</span><b className={x.contribution >= 0 ? 'up' : 'down'}>{x.contribution >= 0 ? '+' : ''}{x.contribution.toFixed(3)}</b>
              </div>)}
              {!selectedResult.explanations?.length && <p>Local SHAP explanation unavailable for this model runtime.</p>}
            </div>
          </>}
        </div>
      </aside>
    </main>}

    {tab === 'metrics' && <main className="single-panel"><div className="panel"><div className="panel-heading"><div><div className="eyebrow">VALIDATION</div><h2>Predictive performance</h2></div><span className="counter">Hold-out 20% + stratified 5-fold CV</span></div><MetricPanel metrics={metrics} /><div className="metrics-footnote">Metrics are generated from the actual training run and stored in <code>models/metrics.json</code>. They are not clinical validation.</div></div></main>}

    {tab === 'about' && <main className="single-panel"><div className="panel about"><div className="panel-heading"><div><div className="eyebrow">METHOD</div><h2>How the prototype works</h2></div></div><div className="architecture"><div>React + Three.js</div><span>→</span><div>FastAPI</div><span>→</span><div>Leakage-safe ML</div><span>→</span><div>UCI clinical data</div></div><h3>Prediction design</h3><p>Four independent Random Forest classifiers estimate CAD, LAD, LCX and RCA status. Every task removes LAD, LCX, RCA and Cath from the feature matrix, plus obvious patient identifiers. Numeric variables use median imputation; categorical variables use most-frequent imputation and one-hot encoding inside the pipeline.</p><h3>Explainability</h3><p>SHAP TreeExplainer generates local attribution values on the fitted Random Forest. One-hot category contributions are aggregated back to the originating clinical feature before they reach the dashboard.</p><h3>3D mapping</h3><p>The viewer renders authentic 3D anatomical meshes (GLTF/GLB) for the heart myocardium and segmented coronary branches (LAD, LCX, RCA). Stable vessel IDs connect model risk probabilities to anatomical artery geometry with customizable translucency and camera angles. Geometry is used as an intuitive decision-support visualization surface; the tabular dataset does not contain image-derived lesion coordinates.</p><h3>Safety boundary</h3><p><b>This is not a diagnostic system.</b> Do not use the output to diagnose, triage, or treat patients. Formal clinical evaluation would require external validation, calibration analysis, larger and more representative cohorts, clinical oversight, and regulatory review.</p></div></main>}

    <footer>CardioMap 3D · Hackathon prototype · Research/education only</footer>
  </div>
}

function colorFor(p) {
  if (p == null) return '#66798c'
  const hue = p < 0.5 ? 172 : 35 - ((p - 0.5) * 250)
  return `hsl(${hue}, 78%, 53%)`
}
