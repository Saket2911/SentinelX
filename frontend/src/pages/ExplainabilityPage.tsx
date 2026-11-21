import { useState } from "react";

import { api } from "../api/client";
import { BarList } from "../components/BarList";
import { ResidualBarChart } from "../components/Charts";
import { SelectField } from "../components/Filters";
import { PageHeader, Panel } from "../components/Panel";
import { Pill } from "../components/Status";
import {
  AvailabilityNotice,
  EmptyState,
  ErrorState,
  Limitations,
  Loading,
  SourceLine,
} from "../components/States";
import { decimal, timestamp } from "../lib/format";
import { useApi } from "../hooks/useApi";

export function ExplainabilityPage() {
  const [label, setLabel] = useState("");
  const [output, setOutput] = useState("");
  const [sample, setSample] = useState(0);

  const { data, error, loading, reload } = useApi(
    () =>
      api.explainability({
        label: label || undefined,
        output: output || undefined,
        sample_index: sample,
        top_n: 12,
      }),
    `explainability-${label}-${output}-${sample}`,
  );

  if (loading && !data) {
    return (
      <Panel title="Loading explanations">
        <Loading rows={4} />
      </Panel>
    );
  }
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!data) return <EmptyState title="No explanation data returned." />;

  const method = data.method;
  const local = data.local;

  return (
    <>
      <PageHeader
        title="Explainability Centre"
        subtitle="SHAP attributions for the LSTM world model: global feature importance, temporal importance, Benign/Bot comparison and local per-observation contributions."
        actions={
          <button type="button" className="btn" onClick={reload}>
            Refresh
          </button>
        }
      />

      <AvailabilityNotice availability={data.availability} title="Explanation tables unavailable" />

      <div className="notice info">
        <span className="notice-icon">i</span>
        <div>
          <strong>Attribution, not causality.</strong> SHAP values explain how input features moved
          the model's prediction relative to a benign background. They are not attack probabilities
          and do not establish why traffic occurred.
        </div>
      </div>

      <div className="grid two">
        <Panel title="Explainability method">
          <dl className="kv">
            <dt>Method</dt>
            <dd>{method.method ?? "—"}</dd>
            <dt>Experiment</dt>
            <dd>{method.experiment ?? "—"}</dd>
            <dt>Checkpoint epoch</dt>
            <dd>{method.checkpoint_epoch ?? "—"}</dd>
            <dt>Background</dt>
            <dd>
              {method.background_source ?? "—"} ({method.background_size ?? "—"} samples)
            </dd>
            <dt>Explanation source</dt>
            <dd>{method.explanation_source ?? "—"}</dd>
            <dt>Samples</dt>
            <dd>
              {method.benign_samples ?? 0} benign · {method.bot_samples ?? 0} bot ·{" "}
              {method.nsamples ?? "—"} nsamples
            </dd>
            <dt>Local explanations</dt>
            <dd>
              {method.local_available ? (
                <Pill tone="good" dot>
                  available
                </Pill>
              ) : (
                <Pill tone="warning" dot>
                  shap_values.npz missing
                </Pill>
              )}
            </dd>
          </dl>
          <div className="divider" style={{ margin: "10px 0" }} />
          <SourceLine source={data.source} />
        </Panel>

        <Panel title="Explained outputs" hint="The five predicted state features explained by the recorded SHAP run.">
          <div className="tag-row">
            {method.outputs.map((name) => (
              <Pill key={name} tone="accent">
                {name}
              </Pill>
            ))}
          </div>
          <div className="divider" style={{ margin: "12px 0" }} />
          <p className="small muted">{method.interpretation}</p>
        </Panel>
      </div>

      <Panel
        title="Global feature importance"
        hint="Mean absolute SHAP attribution aggregated across explained outputs (top 12)."
      >
        <div className="toolbar" style={{ marginBottom: 14 }}>
          <SelectField
            label="Dataset label"
            value={label}
            allLabel="All labels"
            onChange={setLabel}
            options={method.available_labels.map((value) => ({ value, label: value }))}
          />
          <SelectField
            label="Predicted output"
            value={output}
            allLabel="All outputs"
            onChange={setOutput}
            options={method.available_outputs.map((value) => ({ value, label: value }))}
          />
        </div>
        {data.global_importance.length ? (
          <BarList
            items={data.global_importance.map((entry) => ({
              label: entry.feature,
              value: entry.contribution ?? 0,
              tone: entry.direction === "increase" ? "accent" : "info",
              title: entry.direction === "increase" ? "Pushes prediction up" : "Pushes prediction down",
            }))}
          />
        ) : (
          <EmptyState
            title="Aggregated importance table unavailable"
            detail="Only the recorded SHAP experiment summary is present. Generate shap_feature_importance_by_label.csv to enable this chart."
          />
        )}
      </Panel>

      <div className="grid two">
        <Panel title="Temporal importance" hint="Where in the 10-state window the attribution comes from.">
          {data.temporal_importance.length ? (
            <ResidualBarChart
              data={data.temporal_importance.map((entry) => ({
                label: `t${entry.lag_seconds >= 0 ? `-${entry.lag_seconds}` : `+${Math.abs(entry.lag_seconds)}`}`,
                value: entry.mean_absolute,
              }))}
              height={220}
              color="#5aa9f0"
            />
          ) : (
            <EmptyState title="Temporal importance unavailable" />
          )}
        </Panel>

        <Panel title="Benign vs Bot attribution" hint="Largest differences in mean attribution by feature." flush>
          {data.label_comparison.length ? (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th className="num">Benign</th>
                    <th className="num">Bot</th>
                    <th className="num">Difference</th>
                  </tr>
                </thead>
                <tbody>
                  {data.label_comparison.map((row) => (
                    <tr key={row.feature}>
                      <td className="mono">{row.feature}</td>
                      <td className="num">{decimal(row.benign)}</td>
                      <td className="num">{decimal(row.bot)}</td>
                      <td className="num">{decimal(row.difference)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="Comparison table unavailable" />
          )}
        </Panel>
      </div>

      {data.stability.length ? (
        <Panel title="V2 / V3 stability" hint="Agreement of top-10 attributions between feature versions." flush>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Output</th>
                  <th className="num">Top-10 overlap</th>
                  <th className="num">Jaccard</th>
                  <th className="num">Spearman</th>
                </tr>
              </thead>
              <tbody>
                {data.stability.map((row) => (
                  <tr key={`${row.Label}-${row.Output}`}>
                    <td>{String(row.Label)}</td>
                    <td className="mono">{String(row.Output)}</td>
                    <td className="num">{decimal(Number(row["Top10 Overlap"]), 1)}/10</td>
                    <td className="num">{decimal(Number(row.Jaccard), 3)}</td>
                    <td className="num">{decimal(Number(row.Spearman), 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <Panel
        title="Local explanation"
        hint="Per-observation SHAP contributions summed across the 10-state input window."
      >
        {local ? (
          <div className="stack">
            <div className="toolbar">
              <label className="field" style={{ minWidth: 240 }}>
                <span className="field-label">Explanation sample</span>
                <select
                  className="select"
                  value={sample}
                  onChange={(event) => setSample(Number(event.target.value))}
                >
                  {data.samples.map((entry) => (
                    <option key={entry.sample_index} value={entry.sample_index}>
                      #{entry.sample_index} · {entry.label} · sequence {entry.explanation_index}
                    </option>
                  ))}
                </select>
              </label>
              <div className="inline" style={{ alignSelf: "flex-end", paddingBottom: 4 }}>
                <Pill tone={local.label === "Benign" ? "good" : "warning"}>{local.label}</Pill>
                <Pill tone="neutral">output: {local.output}</Pill>
                {local.timestamp ? <span className="small muted">{timestamp(local.timestamp)}</span> : null}
              </div>
            </div>
            <div className="grid two">
              <BarList
                items={local.contributions.slice(0, 12).map((entry) => ({
                  label: entry.feature,
                  value: Math.abs(entry.contribution ?? 0),
                  display: decimal(entry.contribution),
                  tone: (entry.contribution ?? 0) >= 0 ? "accent" : "info",
                }))}
              />
              <ResidualBarChart
                data={local.temporal.map((entry) => ({
                  label: `t${entry.position - 9}`,
                  value: entry.contribution,
                }))}
                height={220}
              />
            </div>
            <Limitations
              items={[
                "Contributions are relative to a benign background sample, not an absence of activity.",
                `Source archive: ${local.source}`,
              ]}
            />
          </div>
        ) : (
          <EmptyState
            title="Local explanations unavailable"
            detail="Per-observation attributions need shap_values.npz from the SHAP experiment. Global and temporal summaries remain available above."
          />
        )}
      </Panel>

      <Panel title="Interpretation limits">
        <Limitations items={data.limitations} />
      </Panel>
    </>
  );
}
