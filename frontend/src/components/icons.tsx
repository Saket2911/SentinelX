interface IconProps {
  size?: number;
  className?: string;
}

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export const Icon = {
  Dashboard: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <rect x="3" y="3" width="7" height="9" rx="1.5" />
      <rect x="14" y="3" width="7" height="5" rx="1.5" />
      <rect x="14" y="12" width="7" height="9" rx="1.5" />
      <rect x="3" y="16" width="7" height="5" rx="1.5" />
    </svg>
  ),
  Network: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <circle cx="12" cy="4.5" r="2.2" />
      <circle cx="4.5" cy="19" r="2.2" />
      <circle cx="19.5" cy="19" r="2.2" />
      <path d="M12 6.7v4.6M12 11.3 5.6 16.9M12 11.3l6.4 5.6" />
    </svg>
  ),
  Predict: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M3 17.5 8.5 11l4 3.6L21 5.5" />
      <path d="M21 10V5.5h-4.5" />
    </svg>
  ),
  Shield: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M12 3 4.5 6v6.2c0 4.3 3.1 7.4 7.5 8.8 4.4-1.4 7.5-4.5 7.5-8.8V6L12 3Z" />
      <path d="m9 12 2.2 2.2L15.4 10" />
    </svg>
  ),
  Explain: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 8.2v.2M10.6 11.6h1.6v5M9.6 16.6h4.8" />
    </svg>
  ),
  Timeline: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M5 3v18" />
      <circle cx="5" cy="7" r="1.8" />
      <circle cx="5" cy="14.5" r="1.8" />
      <path d="M9 7h10M9 14.5h7" />
    </svg>
  ),
  Database: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <ellipse cx="12" cy="6" rx="7.5" ry="3" />
      <path d="M4.5 6v12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6" />
      <path d="M4.5 12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3" />
    </svg>
  ),
  Model: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1" />
    </svg>
  ),
  Report: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M6 2.8h7.5L19 8.3V21a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3.8a1 1 0 0 1 1-1Z" />
      <path d="M13.2 3v5.4H19M9 13h6M9 17h6" />
    </svg>
  ),
  Settings: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7.5 19l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 3 13.9H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.7 7.5l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9.5A1.6 1.6 0 0 0 10.5 3.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8v.1a1.6 1.6 0 0 0 1.5 1h.1a2 2 0 1 1 0 4H21a1.6 1.6 0 0 0-1.5 1Z" />
    </svg>
  ),
  Search: ({ size = 14, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </svg>
  ),
  Alert: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M12 3.5 2.8 20h18.4L12 3.5Z" />
      <path d="M12 9.5v4.2M12 17h.01" />
    </svg>
  ),
  Info: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.5M12 7.8h.01" />
    </svg>
  ),
  Check: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="m4.5 12.5 5 5 10-11" />
    </svg>
  ),
  X: ({ size = 16, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M5.5 5.5l13 13M18.5 5.5l-13 13" />
    </svg>
  ),
  Chevron: ({ size = 14, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="m9 6 6 6-6 6" />
    </svg>
  ),
  Download: ({ size = 14, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M12 3.5v11M7.5 10 12 14.5 16.5 10" />
      <path d="M4 17.5V20a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2.5" />
    </svg>
  ),
  Link: ({ size = 14, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M10 13.8a4 4 0 0 0 5.7 0l2.7-2.7a4 4 0 0 0-5.7-5.7l-1.3 1.3" />
      <path d="M14 10.2a4 4 0 0 0-5.7 0l-2.7 2.7a4 4 0 0 0 5.7 5.7l1.3-1.3" />
    </svg>
  ),
  Refresh: ({ size = 14, className }: IconProps) => (
    <svg {...base(size)} className={className} aria-hidden="true">
      <path d="M20 11a8 8 0 1 0-2.3 6.3" />
      <path d="M20 4.5V11h-6.3" />
    </svg>
  ),
};
