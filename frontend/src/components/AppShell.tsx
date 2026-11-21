import { NavLink, Outlet, useLocation } from "react-router-dom";

import { api } from "../api/client";
import { useApi } from "../hooks/useApi";
import { Pill } from "./Status";

interface NavItem {
  to: string;
  label: string;
  icon: string;
  end?: boolean;
}

const NAV: Array<{ group: string; items: NavItem[] }> = [
  {
    group: "Operations",
    items: [
      { to: "/", label: "SOC Overview", icon: "▤", end: true },
      { to: "/network", label: "Network", icon: "⇄" },
      { to: "/predictions", label: "Predictive Intelligence", icon: "◈" },
    ],
  },
  {
    group: "Analysis",
    items: [
      { to: "/threats", label: "Threat Triage", icon: "▲" },
      { to: "/timeline", label: "Attack Timeline", icon: "≡" },
      { to: "/explainability", label: "Explainability", icon: "◎" },
    ],
  },
  {
    group: "Governance",
    items: [
      { to: "/models", label: "Models", icon: "⬡" },
      { to: "/datasets", label: "Datasets", icon: "▦" },
      { to: "/reports", label: "Reports", icon: "▣" },
      { to: "/settings", label: "Settings", icon: "⚙" },
    ],
  },
];

const TITLES: Record<string, string> = {
  "/": "SOC Overview",
  "/network": "Network Monitoring",
  "/predictions": "Predictive Intelligence",
  "/threats": "Threat Triage",
  "/timeline": "Attack Timeline",
  "/explainability": "Explainability Centre",
  "/models": "Model Management",
  "/datasets": "Dataset Management",
  "/reports": "Reports & Analytics",
  "/settings": "System Settings",
};

export function AppShell() {
  const location = useLocation();
  const health = useApi(() => api.health(), "health");
  const runs = useApi(() => api.runs(), "runs");
  const title = TITLES[location.pathname] ?? "SentinelX";

  const healthTone = health.error
    ? "critical"
    : health.data?.status === "ok"
      ? "good"
      : health.data?.status === "degraded"
        ? "warning"
        : "neutral";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            SX
          </span>
          <div>
            <div className="brand-name">SENTINELX</div>
            <div className="brand-sub">Predictive SOC</div>
          </div>
        </div>
        <nav className="nav" aria-label="Main navigation">
          {NAV.map((group) => (
            <div key={group.group} className="nav-group">
              <span className="nav-group-label">{group.group}</span>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`.trim()}
                >
                  <span className="nav-icon" aria-hidden="true">
                    {item.icon}
                  </span>
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span>LSTM World Model · 209 features</span>
          <span>Artifact-driven · no synthetic telemetry</span>
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API reference
          </a>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-left">
            <span className="crumbs">
              <span>SentinelX</span>
              <span aria-hidden="true">/</span>
              <span className="current">{title}</span>
            </span>
          </div>
          <div className="topbar-right">
            <span className="run-select" title="Active model run served by the API">
              run
              <code>{runs.data?.active_run_id ?? "—"}</code>
            </span>
            <Pill tone={healthTone} dot title={health.error ?? undefined}>
              {health.error
                ? "API unreachable"
                : health.data
                  ? `API ${health.data.status} · v${health.data.version}`
                  : "Checking API"}
            </Pill>
          </div>
        </header>
        <main className="page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
