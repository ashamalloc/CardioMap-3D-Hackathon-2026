import React from 'react'

export default function MetricPanel({ metrics }) {
  const rows = ['CAD', 'LAD', 'LCX', 'RCA']
  const names = { CAD: 'Overall CAD', LAD: 'LAD', LCX: 'LCX', RCA: 'RCA' }
  return (
    <div className="metric-grid">
      {rows.map(key => {
        const m = metrics?.[key]
        const h = m?.holdout || {}
        return <div className="metric-card" key={key}>
          <div className="metric-title">{names[key]}</div>
          <div className="metric-row"><span>Accuracy</span><b>{pct(h.accuracy)}</b></div>
          <div className="metric-row"><span>Precision</span><b>{pct(h.precision)}</b></div>
          <div className="metric-row"><span>Recall</span><b>{pct(h.recall)}</b></div>
          <div className="metric-row"><span>F1</span><b>{pct(h.f1)}</b></div>
          <div className="metric-row"><span>ROC-AUC</span><b>{pct(h.roc_auc)}</b></div>
        </div>
      })}
    </div>
  )
}

function pct(v) { return typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—' }
