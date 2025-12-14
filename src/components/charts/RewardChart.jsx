import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function RewardChart({
  rewards,
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
      .attr("aria-label", "Cumulative Reward per Episode")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xMaxDomain = Math.max(
      100,
      currentEpisode || rewards.length || 0
    );
    const xScale = d3
      .scaleLinear()
      .domain([0, xMaxDomain])
      .range([0, innerWidth]);

    // Y axis is fixed to [-1.8, 1]
    const yScale = d3.scaleLinear().domain([-1.8, 1]).range([innerHeight, 0]);

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
      .text("Cumulative Reward");

    // Line generator
    const line = d3
      .line()
      .x((d, i) => xScale(i + 1))
      .y((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    const pathGroup = g.append("g").attr("clip-path", "url(#clip-reward)");

    pathGroup
      .append("path")
      .datum(rewards)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#60a5fa")
      .attr("stroke-width", 2);

    const points = pathGroup
      .selectAll(".point")
      .data(rewards)
      .join("circle")
      .attr("class", "point")
      .attr("cx", (d, i) => xScale(i + 1))
      .attr("cy", (d) => yScale(d))
      .attr("r", (d, i) => (hoveredEpisode === i ? 4 : 2.2))
      .attr("fill", (d, i) => (hoveredEpisode === i ? "#f97316" : "#93c5fd"))
      .attr("stroke", (d, i) => (hoveredEpisode === i ? "#fff" : "none"))
      .attr("stroke-width", (d, i) => (hoveredEpisode === i ? 0.6 : 0))
      .attr("opacity", (d, i) => (hoveredEpisode == null ? 0.95 : hoveredEpisode === i ? 1 : 0.2))
      .attr("data-idx", (d, i) => i)
      .on("mouseenter", (event) => {
        const i = parseInt(event.target.getAttribute("data-idx"), 10);
        if (onHoverEpisode) onHoverEpisode(i);
      })
      .on("mouseleave", () => {
        if (onHoverEpisode) onHoverEpisode(null);
      });

    if (hoveredEpisode != null && rewards[hoveredEpisode] != null) {
      const hx = xScale(hoveredEpisode + 1);
      const hy = yScale(rewards[hoveredEpisode]);
      const tt = g.append("g").attr("class", "tooltip-reward");
      const text = `Episode ${hoveredEpisode + 1} Reward: ${rewards[hoveredEpisode].toFixed(3)}`;
      tt.append("rect")
        .attr("x", hx + 8)
        .attr("y", hy - 22)
        .attr("rx", 4)
        .attr("ry", 4)
        .attr("width", 140)
        .attr("height", 18)
        .attr("fill", "#0f172a")
        .attr("stroke", "#334155");
      tt.append("text")
        .attr("x", hx + 14)
        .attr("y", hy - 10)
        .attr("fill", "#e5e7eb")
        .attr("font-size", 11)
        .text(text);
    }
  }, [rewards, currentEpisode, hoveredEpisode, width, height]);

  return (
    <div className="reward-chart flex flex-col gap-2">
      <h4 className="font-medium">Cumulative Reward per Episode</h4>
      <svg
        ref={svgRef}
        style={{ background: "transparent", borderRadius: 6 }}
      />
    </div>
  );
}