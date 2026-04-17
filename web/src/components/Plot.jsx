import React, { useCallback, useEffect, useMemo, useRef } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js/lib/core";
import scatter from "plotly.js/lib/scatter";
import bar from "plotly.js/lib/bar";
import heatmap from "plotly.js/lib/heatmap";
import histogram from "plotly.js/lib/histogram";
import scatter3d from "plotly.js/lib/scatter3d";

Plotly.register([scatter, bar, heatmap, histogram, scatter3d]);

const PlotlyComponent = createPlotlyComponent(Plotly);

export default function Plot({
  layout,
  style,
  onInitialized,
  onUpdate,
  onPurge,
  ...props
}) {
  const containerRef = useRef(null);
  const graphDivRef = useRef(null);
  const frameRef = useRef(0);

  const scheduleResize = useCallback(() => {
    if (!graphDivRef.current) return;

    const resize = () => {
      if (!graphDivRef.current) return;
      Promise.resolve(Plotly.Plots.resize(graphDivRef.current)).catch(() => {});
    };

    if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
      if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
      frameRef.current = window.requestAnimationFrame(resize);
    } else {
      resize();
    }
  }, []);

  const normalizedLayout = useMemo(() => {
    const nextLayout = layout ? { ...layout } : {};
    if (nextLayout.autosize == null && nextLayout.width == null && nextLayout.height == null) {
      nextLayout.autosize = true;
    }
    return nextLayout;
  }, [layout]);

  const containerStyle = useMemo(() => ({
    width: "100%",
    height: "100%",
    minWidth: 0,
    minHeight: 0,
    ...style,
  }), [style]);

  const handleInitialized = useCallback((figure, graphDiv) => {
    graphDivRef.current = graphDiv;
    scheduleResize();
    if (onInitialized) onInitialized(figure, graphDiv);
  }, [onInitialized, scheduleResize]);

  const handleUpdate = useCallback((figure, graphDiv) => {
    graphDivRef.current = graphDiv;
    scheduleResize();
    if (onUpdate) onUpdate(figure, graphDiv);
  }, [onUpdate, scheduleResize]);

  const handlePurge = useCallback((figure, graphDiv) => {
    graphDivRef.current = null;
    if (onPurge) onPurge(figure, graphDiv);
  }, [onPurge]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return undefined;

    const observer = new ResizeObserver(() => {
      scheduleResize();
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
      if (typeof window !== "undefined" && typeof window.cancelAnimationFrame === "function" && frameRef.current) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, [scheduleResize]);

  return (
    <div ref={containerRef} style={containerStyle}>
      <PlotlyComponent
        {...props}
        layout={normalizedLayout}
        style={{ width: "100%", height: "100%" }}
        onInitialized={handleInitialized}
        onUpdate={handleUpdate}
        onPurge={handlePurge}
      />
    </div>
  );
}
