import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function StepCountChart({
  exploitHistory = [],
  exploreHistory = [],
  currentEpisode,
  hoveredEpisode = null,
  onHoverEpisode,
  width = 720,
  height = 180,
}) {
  const svgRef = useRef(null);

  const margin = { top: 10, right: 20, bottom: 30, left: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Create root group
    const g = svg
      .attr("width", width)
      .attr("height", height)
      .attr("role", "img")
      .attr("aria-label", "Greedy vs Random Step Counts per Episode")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xMaxDomain = Math.max(
      100,
      currentEpisode || 0,
      exploitHistory?.length || 0,
      exploreHistory?.length || 0
    );
    const xScale = d3
      .scaleLinear()
      .domain([0, xMaxDomain])
      .range([0, innerWidth]);

    // Y axis centered at 0, ranging from -80 to 80 steps
    const yScale = d3.scaleLinear().domain([-80, 80]).range([innerHeight, 0]);

    // Axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(6)
      .tickFormat((d) => (Number.isInteger(d) ? d : d.toFixed(0)));
    const yAxis = d3.axisLeft(yScale).ticks(5);

    g.append("g")
      .attr("transform", `translate(0, ${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .attr("fill", "#cbd5e1");

    g.append("g")
      .call(yAxis)
      .call((g) =>
        g
          .selectAll(".tick line")
          .clone()
          .attr("x2", innerWidth)
          .attr("stroke-opacity", 0.1)
          .attr("stroke", "#cbd5e1")
      )
      .selectAll("text")
      .attr("fill", "#cbd5e1");

    // Axis labels
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + margin.bottom - 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#94a3b8")
      .attr("font-size", 12)
      .text("Episode");

    g.append("text")
      .attr("transform", `rotate(-90)`)
      .attr("x", -innerHeight / 2)
      .attr("y", -margin.left + 12)
      .attr("text-anchor", "middle")
      .attr("fill", "#94a3b8")
      .attr("font-size", 12)
      .text("Step Count");


    const pathGroup = g.append("g").attr("clip-path", "url(#clip-steps)");

    const policyColor = "#22c55e";
    const exploreColor = "#a78bfa";
    const step = innerWidth / xMaxDomain;
    const barWidth = Math.max(1, step * 0.8);
    const baselineY = yScale(0);

    const bars = pathGroup
      .selectAll(".episode-bar")
      .data(
        d3.range(Math.max(exploitHistory.length, exploreHistory.length))
      )
      .join("g")
      .attr("class", "episode-bar")
      .attr("opacity", (d) => (hoveredEpisode == null ? 0.9 : d === hoveredEpisode ? 1 : 0.25))
      .on("mouseenter", (event, d) => {
        if (onHoverEpisode) onHoverEpisode(d);
      })
      .on("mouseleave", () => {
        if (onHoverEpisode) onHoverEpisode(null);
      })
      .each(function (i) {
        const gBar = d3.select(this);
        const x = xScale(i + 1) - barWidth / 2;
        const exploitSteps = exploitHistory[i] || 0;
        const exploreSteps = exploreHistory[i] || 0;
        
        const upH = Math.abs(baselineY - yScale(exploitSteps));
        const downH = Math.abs(baselineY - yScale(-exploreSteps));
        
        if (exploitSteps > 0) {
          gBar
            .append("rect")
            .attr("x", x)
            .attr("y", yScale(exploitSteps))
            .attr("width", barWidth)
            .attr("height", upH)
            .attr("fill", policyColor)
            .attr("opacity", 0.6);
        }
        if (exploreSteps > 0) {
          gBar
            .append("rect")
            .attr("x", x)
            .attr("y", baselineY)
            .attr("width", barWidth)
            .attr("height", downH)
            .attr("fill", exploreColor)
            .attr("opacity", 0.6);
        }
      });

    if (hoveredEpisode != null) {
      const i = hoveredEpisode;
      const policy = exploitHistory[i] || 0;
      const explore = exploreHistory[i] || 0;
      const total = policy + explore;
      const hx = xScale(i + 1);
      const hy = 12;
      const tt = g.append("g").attr("class", "tooltip-steps");
      const lines = [
        `Episode ${i + 1}`,
        `Total Steps Count: ${total}`,
        `Policy Steps Count: ${policy}`,
        `Random Steps Count: ${explore}`,
      ];
      const boxW = 150;
      const boxH = 18 * lines.length + 8;
      tt.append("rect")
        .attr("x", hx + 8)
        .attr("y", hy)
        .attr("rx", 6)
        .attr("ry", 6)
        .attr("width", boxW)
        .attr("height", boxH)
        .attr("fill", "#0f172a")
        .attr("stroke", "#334155");
      lines.forEach((t, idx) => {
        tt.append("text")
          .attr("x", hx + 16)
          .attr("y", hy + 18 + idx * 16)
          .attr("fill", "#e5e7eb")
          .attr("font-size", 11)
          .text(t);
      });
    }
  }, [exploitHistory, exploreHistory, currentEpisode, hoveredEpisode, width, height]);

  return (
    <div className="step-count-chart flex flex-col">
      <h4 className="font-medium">Exploitation vs Exploration Step Counts per Episode</h4>
      <div className="flex items-center gap-5 text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: '#22c55e', opacity: 0.7 }}></div>
          <span className="text-slate-300 text-xs">Policy (exploitation)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: '#a78bfa', opacity: 0.7 }}></div>
          <span className="text-slate-300 text-xs">Random (exploration)</span>
        </div>
      </div>
      <svg
        ref={svgRef}
        style={{ background: "transparent", borderRadius: 6 }}
      />
    </div>
  );
}