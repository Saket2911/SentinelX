import { createContext, useContext } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Icon } from "./components/icons";
import { Tag, Tip } from "./components/ui";
import { api, type Health } from "./lib/api";
import { useApi } from "./lib/useApi";
import Dashboard from "./pages/Dashboard";
import NetworkMonitoring from "./pages/NetworkMonitoring";
import Predictive from "./pages/Predictive";
import ThreatTriage from "./pages/ThreatTriage";
import Explainability from "./pages/Explainability";
import AttackTimeline from "./pages/AttackTimeline";
import Datasets from "./pages/Datasets";
import Models from "./pages/Models";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";

export const HealthCtx = createContext<{ health: Health | null; reload: () => void }>({
  health: null,
  reload: () => {},
});

export const useHealth = () => useContext(HealthCtx);

const NAV: { group: string; items: { to: string; label: string; icon: keyof typeof Icon }[] }[] = [
  {
    group: "Operations",
    items: [
      { to: "/", label: "SOC Dashboard", icon: "Dashboard" },
      { to: "/network", label: "Network Monitoring", icon: "Network" },
      { to: "/predictive", label: "Predictive Intelligence", icon: "Predict" },
      { to: "/triage", label: "Threat Triage", icon: "Shield" },
      { to: "/timeline", label: "Attack Timeline", icon: "Timeline" },
    ],
  },
  {
    group: "Analysis",
    items: [
      { to: "/explainability", label: "Explainability", icon: "Explain" },
      { to: "/reports", label: "Reports & Analytics", icon: "Report" },
    ],
  },
  {
    group: "Platform",
    items: [
      { to: "/datasets", label: "Datasets", icon: "Database" },
      { to: "/models", label: "Models", icon: "Model" },
      { to: "/settings", label: "Settings", icon: "Settings" },
    ],
  },
];

const TITLES: Record<string, string> = {
  "/": "SOC Dashboard",
  "/network": "Network Monitoring",
  "/predictive": "Predictive Intelligence",
  "/triage": "Threat Triage",
  "/timeline": "Attack Timeline",
  "/explainability": "Explainability Centre",
  "/reports": "Reports & Analytics",
  "/datasets": "Dataset Management",
  "/models": "Model Management",
  "/settings": "System Settings",
};

export default function App() {
  const { data: health, reload } = useApi<Health>(() => api.health(), []);
  const location = useLocation();
  const here = TITLES[location.pathname] ?? "SentinelX";

  return (
    <HealthCtx.Provider value={{ health, reload }}>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">
              <Icon.Shield size={15} />
            </div>
            <div>
              <div className="brand-name">SentinelX</div>
              <div className="brand-sub">Predictive SOC</div>
            </div>
          </div>

          <nav className="nav">
            {NAV.map((g) => (
              <div key={g.group}>
                <div className="nav-group-label">{g.group}</div>
                {g.items.map((item) => {
                  const Ico = Icon[item.icon];
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === "/"}
                      className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
                    >
                      <span className="ico">
                        <Ico size={15} />
                      </span>
                      {item.label}
                    </NavLink>
                  );
                })}
              </div>
            ))}
          </nav>

          <div className="sidebar-foot">
            <div className="row" style={{ gap: 6 }}>
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: health ? "var(--ok)" : "var(--med)",
                  display: "inline-block",
                }}
              />
              <span>{health ? "API connected" : "Connecting…"}</span>
            </div>
            <div className="row" style={{ gap: 6 }}>
              <Tag tone="neutral">Offline dataset mode</Tag>
            </div>
            <div style={{ fontSize: 10 }}>v{health?.version ?? "4.0.0"} · no live telemetry</div>
          </div>
        </aside>

        <div className="main">
          <header className="topbar">
            <div className="crumbs">
              <span>SentinelX</span>
              <span className="sep">/</span>
              <span className="here">{here}</span>
            </div>
            <div className="topbar-right">
              <Tip text="All figures come from recorded research artifacts. No live network telemetry is ingested.">
                <Tag tone="derived">Offline dataset mode</Tag>
              </Tip>
              <Tip
                text={
                  health?.inference_available
                    ? "A trained checkpoint is present and loadable."
                    : "No trained checkpoint in this checkout, so on-demand inference is disabled. Recorded results remain available."
                }
              >
                <Tag tone={health?.inference_available ? "ok" : "unavailable"} dot>
                  {health?.inference_available ? "Inference ready" : "Inference offline"}
                </Tag>
              </Tip>
            </div>
          </header>

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/network" element={<NetworkMonitoring />} />
            <Route path="/predictive" element={<Predictive />} />
            <Route path="/triage" element={<ThreatTriage />} />
            <Route path="/timeline" element={<AttackTimeline />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<Models />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>
    </HealthCtx.Provider>
  );
}
