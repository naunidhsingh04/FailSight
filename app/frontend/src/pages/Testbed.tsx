import { useApi } from '../api'
import ResponsiveTable from '../components/ResponsiveTable'
import type { Column } from '../components/ResponsiveTable'
import { Async } from '../components/States'
import { Detailed, Simple } from '../mode'
import { TESTBED_LIMITS } from '../testbed'
import { usePageTitle } from '../usePageTitle'
import { Link } from 'react-router-dom'

type Row = Record<string, any>
interface Results { limits: string[]; frozen: Row | null; assignment: Row[]; overall: Row[]; by_fault: Row[]; false_alarms: Row[]; prefailure: Row[]; runid: Row[]; shuffled: Row[]; live: Row[]; cv_by_fault: Row[] }

const f3 = (x: any) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(3))
const f1 = (x: any) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(1))
const ci = (r: Row, k: string) => `${f3(r[k])} (${f3(r[`${k}_ci_low`])}–${f3(r[`${k}_ci_high`])})`

export default function Testbed() {
  usePageTitle('Testbed')
  const res = useApi<Results>('/api/testbed/results')
  return (
    <>
      <h1>Testbed</h1>
      <p className="lede">
        <Simple>AnoMod could not tell us if failures are really detected, because it has no times for when each fault starts. So we built a small shop, broke it on purpose at recorded times, and tested on whole recordings the models had never seen.</Simple>
        <Detailed>Our own app (FastAPI + SQLite, monitored only through the <code>failsight_agent</code> package), 40 runs of 10 minutes with 46 injected faults and exact fault schedules; validation by whole runs; frozen detector; three scripted 12-minute live demos.</Detailed>
      </p>
      <div className="callout" role="note">
        <span className="label">Read this first</span>
        <ul>{TESTBED_LIMITS.map((t) => <li key={t}>{t}</li>)}</ul>
      </div>
      <Async state={res} rows={6} label="Loading the testbed results" level={2}>
        {(r) => <Body r={r} />}
      </Async>
    </>
  )
}

function Body({ r }: { r: Results }) {
  const supervised = r.overall.filter((m) => m.kind === 'supervised')
  const best = [...supervised].sort((a, b) => b.balanced_accuracy - a.balanced_accuracy)[0]
  const free = r.false_alarms.find((x) => x.method === best?.method && x.rule === '2 of last 3')
  const live = r.live
  const overallCols: Column<Row>[] = [
    { key: 'm', label: 'Method', nameCell: true, render: (x) => x.method, sortValue: (x) => x.method },
    { key: 'ba', label: 'Balanced accuracy (95% CI)', numeric: true, render: (x) => ci(x, 'balanced_accuracy'), sortValue: (x) => x.balanced_accuracy },
    { key: 'f1', label: 'Macro-F1', numeric: true, render: (x) => f3(x.macro_f1), sortValue: (x) => x.macro_f1 },
    { key: 'nr', label: 'Normal recall (95% CI)', numeric: true, render: (x) => ci(x, 'normal_recall'), sortValue: (x) => x.normal_recall },
    { key: 'auc', label: 'ROC-AUC', numeric: true, render: (x) => f3(x.roc_auc), sortValue: (x) => x.roc_auc },
  ]
  const liveCols: Column<Row>[] = [
    { key: 's', label: 'Live run', nameCell: true, render: (x) => `${x.scenario.replace('_', ' ')} (seed ${x.seed})`, sortValue: (x) => x.seed },
    { key: 'f', label: 'Fault injected', render: (x) => (x.actual_fault === 'none' ? 'none' : `${x.actual_fault} (${x.planned_severity})`) },
    { key: 'd', label: 'Detected', render: (x) => (x.actual_fault === 'none' ? '–' : x.detected ? 'yes' : 'NO') },
    { key: 'dl', label: 'Delay (s)', numeric: true, render: (x) => f1(x.detection_delay_s), sortValue: (x) => x.detection_delay_s },
    { key: 'fa', label: 'False alarms', numeric: true, render: (x) => `${x.false_alarms} (${f1(x.false_alarms_per_hour)}/h)`, sortValue: (x) => x.false_alarms },
    { key: 'mi', label: 'Fault windows missed', numeric: true, render: (x) => `${x.missed_fault_windows} of ${x.fault_windows}`, sortValue: (x) => x.missed_fault_windows },
  ]
  const faultCols: Column<Row>[] = [
    { key: 'm', label: 'Method', nameCell: true, render: (x) => x.method, sortValue: (x) => x.method },
    { key: 'g', label: 'Group', render: (x) => x.group },
    { key: 'v', label: 'Value', render: (x) => x.value },
    { key: 'i', label: 'Instances', numeric: true, render: (x) => x.instances, sortValue: (x) => x.instances },
    { key: 'r', label: 'Recall (95% CI)', numeric: true, render: (x) => ci(x, 'recall').replace(/\(.*\)/, `(${f3(x.ci_low)}–${f3(x.ci_high)})`), sortValue: (x) => x.recall },
    { key: 'ok', label: 'Reliable?', render: (x) => (x.reliable ? 'yes' : 'too few instances') },
  ]
  return (
    <>
      <h2>The result in plain words</h2>
      <Simple>
        <ul>
          {best && <li>On 12 recordings the models had never seen, the best model ({best.method}) told normal from faulty windows with a balanced accuracy of {f3(best.balanced_accuracy)} (range {f3(best.balanced_accuracy_ci_low)}–{f3(best.balanced_accuracy_ci_high)}).</li>}
          {free && <li>On the 2 recordings without any fault it raised {free.alert_episodes_fault_free} false alarm{free.alert_episodes_fault_free === 1 ? '' : 's'} in {f1(free.fault_free_hours * 60)} minutes. That is far too little data to quote a rate.</li>}
          <li>Mild faults and slowly growing faults are caught less reliably than severe, sudden ones, and most fault types have only one to three examples in the test, so a number per fault type should not be trusted.</li>
          <li>Because every recording contains both normal and faulty periods, recognising a recording cannot explain the result. (On AnoMod it could.)</li>
        </ul>
      </Simple>
      <Detailed><p>Held-out test: 12 runs, scored windows only (after each run&apos;s 150 s learning period, transition windows excluded). Intervals are run-bootstrap 95% CIs.</p></Detailed>

      <h2>Live demos (three new runs)</h2>
      {live.length === 0 ? <div className="state"><h3>No live demo has been run yet</h3><p>Run <code>python -m testbed.live_demo --scenario all</code>.</p></div> : (
        <>
          <p>Each demo is a new 12-minute run with a seed that was never used for training. The fault schedule was written down before the run. The numbers below are exactly what came out; nothing was tuned afterwards.</p>
          <ResponsiveTable<Row> caption="Live demo results (monitor predictions against the recorded fault schedule)" columns={liveCols} rows={live} rowKey={(x) => String(x.seed)} />
        </>
      )}

      <Detailed>
        <h2>Held-out results by method</h2>
        <ResponsiveTable<Row> caption="Held-out test runs: detection of faulty windows" columns={overallCols} rows={r.overall} rowKey={(x) => x.method} initialSort={{ key: 'ba', dir: 'desc' }} />
        <h2>Recall by fault type, severity and kind</h2>
        <p className="small muted">A cell is called reliable only with at least 5 fault instances. Most per-type cells in the test runs are not.</p>
        <ResponsiveTable<Row> caption="Share of fault windows that were flagged, XGBoost and the unsupervised detector" columns={faultCols} rows={r.by_fault.filter((x) => (x.method === 'XGB (raw)' || x.method.startsWith('RobustZ')) && x.group !== 'Normal windows')} rowKey={(x) => `${x.method}|${x.group}|${x.value}`} />
        <h2>Every run held out once (more instances per cell)</h2>
        <ResponsiveTable<Row> caption="XGBoost, all 40 runs, each run held out once" columns={faultCols} rows={r.cv_by_fault} rowKey={(x) => `${x.group}|${x.value}`} />
        <h2>Run assignment</h2>
        <ResponsiveTable<Row> caption="Which runs were held out for testing and which were used for training (folds by whole run)" columns={[
          { key: 'r', label: 'Run', nameCell: true, render: (x) => x.run_id, sortValue: (x) => x.run_id }, { key: 's', label: 'Split', render: (x) => x.split },
          { key: 'f', label: 'Fold', numeric: true, render: (x) => (x.fold === null || x.fold === '' ? '–' : x.fold) }, { key: 'x', label: 'Faults', render: (x) => x.faults }]} rows={r.assignment} rowKey={(x) => x.run_id} />
        <h2>Controls</h2>
        <ResponsiveTable<Row> caption="Shuffled-label control: balanced accuracy with the training labels shuffled" columns={[
          { key: 'm', label: 'Method', nameCell: true, render: (x) => x.method }, { key: 'a', label: 'Shuffled labels', numeric: true, render: (x) => f3(x.shuffled_labels_mean_ba) }, { key: 'b', label: 'Real labels', numeric: true, render: (x) => f3(x.real_labels_ba) }]} rows={r.shuffled} rowKey={(x) => x.method} />
        <ResponsiveTable<Row> caption="Run identification check" columns={[{ key: 'c', label: 'Check', nameCell: true, render: (x) => x.check }, { key: 'v', label: 'Value', numeric: true, render: (x) => f3(x.value) }, { key: 'ch', label: 'Chance', numeric: true, render: (x) => f3(x.chance) }]} rows={r.runid} rowKey={(x) => x.check} />
      </Detailed>
      <p className="small muted">All numbers come from <code>report_outputs/testbed_*.csv</code> and are recomputed on the <Link to="/proofs">Proofs</Link> page (P12, P16, P17). Dataset description: <code>docs/TESTBED_DATASET.md</code>.</p>
    </>
  )
}
