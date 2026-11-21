import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Keeps one failing page from unmounting the whole console. The boundary is
 * keyed by route, so navigating elsewhere clears the error automatically.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Page render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="state" role="alert">
        <span className="notice-icon" style={{ color: "var(--danger)" }}>
          ERR
        </span>
        <span className="state-title">This page could not be rendered</span>
        <div className="state-detail">
          The rest of the console is unaffected. The underlying data was not changed.
        </div>
        <code className="mono small" style={{ overflowWrap: "anywhere" }}>
          {this.state.error.message}
        </code>
        <div className="inline" style={{ marginTop: 10 }}>
          <button type="button" className="btn" onClick={() => this.setState({ error: null })}>
            Try again
          </button>
          <button type="button" className="btn primary" onClick={() => window.location.reload()}>
            Reload console
          </button>
        </div>
      </div>
    );
  }
}
