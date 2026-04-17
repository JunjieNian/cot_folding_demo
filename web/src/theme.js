// Design tokens for COT Folding Map visualization
export const theme = {
  colors: {
    explore: "#5B8DEF",
    exploit: "#E05A47",
    bond: "rgba(220,180,60,0.25)",
    primary: "#1A73E8",
    bg: "#F2F3F5",
    surface: "#FFFFFF",
    surfaceAlt: "#F7F8FA",
    border: "#E5E7EB",
    borderLight: "#F0F1F3",
    text: "#1A1A2E",
    textSecondary: "#555",
    textMuted: "#888",
    textFaint: "#999",
    error: "#C5221F",
    errorBg: "#FDECEA",
    infoBg: "#E8F0FE",
    success: "#4CAF50",
    warning: "#FF9800",
    highlight: "#FFF59D",
    highlightBorder: "#F9A825",
    // Flow colors
    arterial: "#FF8A65",
    venous: "#7E57C2",
    capillary: "#FFD54F",
    shunt: "#9E9E9E",
    // Functional colors
    core: "#4CAF50",
    closure: "#FF9800",
    drift: "#F44336",
    returnSite: "#AB47BC",
    productive: "#90A4AE",
    catalyticBond: "#FFD700",
  },
  spacing: { xs: 2, sm: 4, md: 8, lg: 12, xl: 16, xxl: 24 },
  fontSize: { xs: 9, sm: 10, md: 11, lg: 12, xl: 13, xxl: 14, heading: 16 },
  radius: { sm: 4, md: 6, lg: 8, pill: 20 },
  sidebar: { width: 210, collapsedWidth: 48 },
  textPanel: { width: 340 },
};

export const darkTheme = {
  ...theme,
  colors: {
    ...theme.colors,
    bg: "#1A1A2E",
    surface: "#16213E",
    surfaceAlt: "#1A1A2E",
    border: "#2A2A4A",
    borderLight: "#2A2A4A",
    text: "#E0E0E0",
    textSecondary: "#B0B0B0",
    textMuted: "#888",
    textFaint: "#666",
    infoBg: "#1A2744",
    errorBg: "#3C1515",
    highlight: "#5C5300",
    highlightBorder: "#8C7A00",
  },
};

export const FLOW_COLORS = {
  arterial: theme.colors.arterial,
  venous: theme.colors.venous,
  capillary: theme.colors.capillary,
  shunt: theme.colors.shunt,
};

export const FUNC_COLORS = {
  core: theme.colors.core,
  closure: theme.colors.closure,
  drift: theme.colors.drift,
  return_site: theme.colors.returnSite,
  productive: theme.colors.productive,
  catalytic_bond: theme.colors.catalyticBond,
};
