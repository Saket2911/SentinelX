import { Link, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { ExplainabilityPage } from "./pages/ExplainabilityPage";
import { FeatureExplorerPage } from "./pages/FeatureExplorerPage";
import { ModelsPage } from "./pages/ModelsPage";
import { NetworkPage } from "./pages/NetworkPage";
import { PredictionsPage } from "./pages/PredictionsPage";
import { ReplayPage } from "./pages/ReplayPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ThreatsPage } from "./pages/ThreatsPage";
import { TimelinePage } from "./pages/TimelinePage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="network" element={<NetworkPage />} />
        <Route path="predictions" element={<PredictionsPage />} />
        <Route path="replay" element={<ReplayPage />} />
        <Route path="explorer" element={<FeatureExplorerPage />} />
        <Route path="threats" element={<ThreatsPage />} />
        <Route path="timeline" element={<TimelinePage />} />
        <Route path="explainability" element={<ExplainabilityPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route
          path="*"
          element={
            <div className="state">
              <span className="state-title">Page not found</span>
              <Link className="btn" to="/">
                Return to overview
              </Link>
            </div>
          }
        />
      </Route>
    </Routes>
  );
}
