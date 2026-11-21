import { Link, NavLink, Outlet, useLocation } from "react-router-dom";

import { api } from "../api/client";
import { useApi } from "../hooks/useApi";
import { ErrorBoundary } from "./ErrorBoundary";
import { Pill } from "./Status";

interface NavItem {
  to: string;
  label: string;
  icon: string;
  end?: boolean;
}

const NAV: Array<{ group: string; items: NavItem[] }> = [
  {
    group: "Overview",
    items: [{ to: "/", label: "SOC Dashboard", icon: "▤", end: true }],
  },
  {
    group: "Network Intelligence",
    items: [
      { to: "/network", label: "Network Monitoring", icon: "⇄" },
      { to: "/replay", label: "Network Playback", icon: "▶" },
    ],
  },
  {
    group: "Predictive Intelligence",
    items: [
      { to: "/predictions", label: "Future State Prediction", icon: "◈" },
      { to: "/explorer", label: "Feature Explorer", icon: "⌗" },
      { to: "/explainability", label: "Explainability", icon: "◎" },
    ],
  },
  {
    group: "Security Operations",
    items: [
      { to: "/threats", label: "Threat Triage", icon: "▲" },
      { to: "/timeline", label: "Attack Timeline", icon: "≡" },
    ],
  },
  {
    group: "Model & Data",
    items: [
      { to: "/models", label: "Model Management", icon: "⬡" },
      { to: "/datasets", label: "Dataset Management", icon: "▦" },
    ],
  },
  {
    group: "System",
    items: [
      { to: "/reports", label: "Reports", icon: "▣" },
      { to: "/settings", label: "Settings", icon: "⚙" },
    ],
  },
];

const TITLES: Record<string, string> = {
  "/": "SOC Dashboard",
  "/network": "Network Monitoring",
  "/replay": "Network State Playback",
  "/predictions": "Future State Prediction",
  "/explorer": "Feature Explorer",
  "/threats": "Threat Triage",
  "/timeline": "Attack Timeline",
  "/explainability": "Explainability Centre",
  "/models": "Model Management",
  "/datasets": "Dataset Management",
  "/reports": "Reports & Analytics",
  "/settings": "System Settings",
};

const GROUP_FOR: Record<string, string> = {};
NAV.forEach((group) => group.items.forEach((item) => (GROUP_FOR[item.to] = group.group)));

export function AppShell() {
  const location = useLocation();
  const health = useApi(() => api.health(), "shell-health");
  const runs = useApi(() => api.runs(), "shell-runs");
  const settings = useApi(() => api.settings(), "shell-settings");
  const candidates = useApi(() => api.events({ page_size: 1 }), "shell-candidates");

  const title = TITLES[location.pathname] ?? "SentinelX";
  const group = GROUP_FOR[location.pathname] ?? "Overview";

  const healthTone = health.error
    ? "critical"
    : health.data?.status === "ok"
      ? "good"
      : health.data?.status === "degraded"
        ? "warning"
        : "neutral";

  const mode = settings.data?.settings.data_mode ?? "dataset";
  const reviewCount = candidates.data?.total ?? 0;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link to="/" className="brand" aria-label="SentinelX dashboard">
          <span className="brand-mark" aria-hidden="true">SX</span>
          <span>
            <span className="brand-name">SENTINELX</span>
            <span className="brand-sub">Predictive Cyber Defence</span>
          </span>
        </Link>
        <nav className="nav" aria-label="Main navigation">
          {NAV.map((section) => (
            <div key={section.group} className="nav-group">
              <span className="nav-group-label">{section.group}</span>
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`.trim()}
                >
                  <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span>LSTM World Model · 209 features</span>
          <span>Artifact-driven · no synthetic telemetry</span>
          <a href="/api/docs" target="_blank" rel="noreferrer">API reference</a>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <span className="crumbs">
              <span>{group}</span>
              <span aria-hidden="true">/</span>
              <span className="current">{title}</span>
            </span>
          </div>
          <div className="topbar-right">
            <Pill tone={mode === "live" ? "warning" : "info"} title="Data source configured in Settings">
              {mode === "live" ? "Live mode · no capture" : "Dataset mode"}
            </Pill>
            <span className="run-select" title="Active model run served by the API">
              model
              <code>{runs.data?.active_run_id ?? "—"}</code>
            </span>
            <Link
              to="/threats"
              className="notification"
              title={`${reviewCount.toLocaleString()} residual review candidates`}
            >
              <span aria-hidden="true">◆</span>
              <span className="notification-count">{reviewCount > 999 ? "999+" : reviewCount}</span>
              <span className="sr-only">review candidates</span>
            </Link>
            <Pill tone={healthTone} dot title={health.error ?? undefined}>
              {health.error ? "API unreachable" : health.data ? `API ${health.data.status}` : "Checking API"}
            </Pill>
          </div>
        </header>
        <main className="page">
          <ErrorBoundary key={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
