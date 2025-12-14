import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

import * as d3 from "d3";

import GridWorldCanvas from "./components/GridWorldCanvas.jsx";
import useGridWorld from "./hooks/useGridWorld.js";
import useDQN, { useNewDQN } from "./hooks/useDQN.js";
import useTableQLearning from "./hooks/useTableQLearning.js";
import CartPoleCanvas from "./components/CartPoleCanvas.jsx";
import useCartPoleDQN from "./hooks/useCartPoleDQN.js";
import NNVisual from "./components/NNVisual.jsx";
import DQNExplain from "./components/DQNExplain.jsx";
function RewardChart({
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

    g.append("defs")
      .append("clipPath")
      .attr("id", "clip-reward")
      .append("rect")
      .attr("width", innerWidth)
      .attr("height", innerHeight);

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



function StepCountChart({
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

    g.append("defs")
      .append("clipPath")
      .attr("id", "clip-steps")
      .append("rect")
      .attr("width", innerWidth)
      .attr("height", innerHeight);

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
      .attr("opacity", (d) => (hoveredEpisode == null ? 0.9 : d === hoveredEpisode ? 1 : 0.35))
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
        `Explore Steps Count: ${explore}`,
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
      <h4 className="font-medium">Greedy vs Random Step Counts per Episode</h4>
      <div className="flex items-center gap-5 text-sm">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: '#22c55e', opacity: 0.7 }}></div>
          <span className="text-slate-300 text-xs">Policy (greedy)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: '#a78bfa', opacity: 0.7 }}></div>
          <span className="text-slate-300 text-xs">Explore (random)</span>
        </div>
      </div>
      <svg
        ref={svgRef}
        style={{ background: "transparent", borderRadius: 6 }}
      />
    </div>
  );
}

// Tooltip component for hyperparameters
function HyperparamLabel({ children, tooltip, targetId }) {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleClick = () => {
    const element = document.getElementById(targetId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });

      element.classList.add("highlight-flash");
      setTimeout(() => element.classList.remove("highlight-flash"), 2000);
    }
  };

  return (
    <span className="relative inline-block">
      <span
        className="cursor-pointer hover:text-blue-400 transition-colors underline decoration-dotted"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={handleClick}
      >
        {children}
      </span>
      {showTooltip && (
        <span className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-950 text-slate-200 text-xs rounded border border-slate-700 shadow-lg whitespace-nowrap">
          {tooltip}
          <span className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-950"></span>
        </span>
      )}
    </span>
  );
}

// Interactive view pane
function InteractivePane() {
  const { state, reset, step } = useGridWorld({ gridSize: 6, cellSize: 72 });
  // Hidden layer config state
  const [numHidden, setNumHidden] = useState(2);
  const [hiddenSizes, setHiddenSizes] = useState([64, 64]);
  // Hyperparameter config state
  const [learningRate, setLearningRate] = useState(5e-4);
  const [epsilonDecay, setEpsilonDecay] = useState(1000);
  const [gamma, setGamma] = useState(0.99);

  const [diagonalSensors, setDiagonalSensors] = useState(true);
  const [straightSensors, setStraightSensors] = useState(true);
  const [goalLocalication, setGoalLocalication] = useState(true);
  const [pitDistance, setPitDistance] = useState(true);

  const [sensorError, setSensorError] = useState("");

  const [stopped, setStopped] = useState(false);

  // Update hiddenSizes when numHidden changes
  useEffect(() => {
    setHiddenSizes((prev) => {
      if (numHidden > prev.length) {
        return [...prev, ...Array(numHidden - prev.length).fill(64)];
      } else if (numHidden < prev.length) {
        return prev.slice(0, numHidden);
      }
      return prev;
    });
  }, [numHidden]);

  const {
    startTraining,
    stopTraining,
    resumeTraining,
    training,
    episode,
    totalSteps,
    lastReward,
    epsilon,
    avgReward,
    getQValues,
    exploreCount,
    exploitCount,
    model,
  } = useNewDQN({
    gridState: state,
    envStep: step,
    envReset: reset,
    hiddenLayers: hiddenSizes,
    learningRate,
    epsilonDecay,
    gamma,
    sensorFlags: {
      diagonalSensors,
      straightSensors,
      goalLocalication,
      pitDistance,
    },
  });

  const [cellQGrid, setCellQGrid] = useState(null);

  // compute Q-values for every valid cell whenever grid state or model changes
  useEffect(() => {
    if (!state || !getQValues) {
      setCellQGrid(null);
      return;
    }

    const gs = state.gridSize;
    const grid = Array.from({ length: gs }, () =>
      Array.from({ length: gs }, () => null)
    );

    for (let r = 0; r < gs; r++) {
      for (let c = 0; c < gs; c++) {
        const isBlock = state.blocks.some((b) => b[0] === r && b[1] === c);
        const isPit = state.pit[0] === r && state.pit[1] === c;
        const isGoal = state.goalPos[0] === r && state.goalPos[1] === c;
        if (isBlock || isPit || isGoal) {
          grid[r][c] = null;
          continue;
        }
        try {
          const q = getQValues([r, c]);
          grid[r][c] = Array.isArray(q) ? q : null;
        } catch (e) {
          grid[r][c] = null;
        }
      }
    }

    setCellQGrid(grid);
  }, [state, getQValues]);

  const resetTrainParams = () => {
    reset();
    setStopped(false);
  };

  const SensorPreview = ({ type }) => {
    const size = 60;
    const grid = 3;
    const cell = size / grid;
    const cx = 1.5 * cell;
    const cy = 1.5 * cell;
    const lines = (() => {
      if (type === "diagonal") {
        const dirs = [
          [-1, -1],
          [1, -1],
          [1, 1],
          [-1, 1],
        ];
        return dirs.map((d, i) => (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={cx + d[0] * cell}
            y2={cy + d[1] * cell}
            stroke="rgb(255,255,255)"
            strokeWidth={2}
          />
        ));
      }
      if (type === "straight") {
        const dirs = [
          [0, -1],
          [1, 0],
          [0, 1],
          [-1, 0],
        ];
        return dirs.map((d, i) => (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={cx + d[0] * cell}
            y2={cy + d[1] * cell}
            stroke="rgb(255,255,255)"
            strokeWidth={2}
          />
        ));
      }
      if (type === "goal") {
        return (
          <line
            x1={cx}
            y1={cy}
            x2={2.5 * cell}
            y2={0.5 * cell}
            stroke="rgba(90,200,90,0.7)"
            strokeWidth={1}
          />
        );
      }
      if (type === "pit") {
        return (
          <circle
            cx={cx}
            cy={cy}
            r={Math.sqrt(2) * cell}
            fill="none"
            stroke="rgba(255,0,0,0.7)"
            strokeWidth={2}
          />
        );
      }
      return null;
    })();
    return (
      <svg
        width={size}
        height={size}
        className="rounded border border-slate-700 w-[60px] h-[60px]"
      >
        <rect x={0} y={0} width={size} height={size} fill="#0f172a" />
        {Array.from({ length: grid + 1 }).map((_, i) => (
          <line
            key={`h-${i}`}
            x1={0}
            y1={i * cell}
            x2={size}
            y2={i * cell}
            stroke="rgba(255,255,255,0.15)"
          />
        ))}
        {Array.from({ length: grid + 1 }).map((_, i) => (
          <line
            key={`v-${i}`}
            x1={i * cell}
            y1={0}
            x2={i * cell}
            y2={size}
            stroke="rgba(255,255,255,0.15)"
          />
        ))}
        <circle cx={cx} cy={cy} r={cell * 0.3} fill="rgb(66,135,245)" />
        {lines}
        {type === "goal" && (
          <rect
            x={2 * cell + cell * 0.25}
            y={0 * cell + cell * 0.25}
            width={cell * 0.5}
            height={cell * 0.5}
            fill="rgba(90,200,90,0.6)"
          />
        )}
        {type === "pit" && (
          <rect
            x={0 * cell + cell * 0.25}
            y={2 * cell + cell * 0.25}
            width={cell * 0.5}
            height={cell * 0.5}
            fill="rgba(255,0,0,0.6)"
          />
        )}
      </svg>
    );
  };

  const SensorButton = ({ title, selected, onClick, type }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const tooltipText = (() => {
      switch (type) {
        case "diagonal":
          return "Detects diagonal boundaries and obstacles (does not detect pits)";
        case "straight":
          return "Detects cardinal boundaries and obstacles (does not detect pits)";
        case "goal":
          return "Provides the agent's x and y offset from the goal";
        case "pit":
          return "Measures the Euclidean distance to the nearest pit";
        default:
          return title;
      }
    })();

    return (
      <span className="relative inline-block">
        <button
          onClick={() => {
            onClick();
            setSensorError("");
          }}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          className={`cursor-pointer flex items-center gap-2 px-3 py-2 rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 transition ${
            selected ? "" : "opacity-40"
          }`}
        >
          <SensorPreview type={type} />
          <span className="text-xs">{title}</span>
        </button>
        {showTooltip && (
          <span className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-950 text-slate-200 text-xs rounded border border-slate-700 shadow-lg whitespace-nowrap">
            {tooltipText}
            <span className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-950"></span>
          </span>
        )}
      </span>
    );
  };

  // ---- reward history state and episode change handling ----
  const [rewards, setRewards] = useState([]);
  const [exploitHistory, setExploitHistory] = useState([]);
  const [exploreHistory, setExploreHistory] = useState([]);
  const [hoveredEpisode, setHoveredEpisode] = useState(null);
  const prevEpisodeRef = useRef(0);

  useEffect(() => {
    if (episode < prevEpisodeRef.current) {
      setRewards([]);
      setExploitHistory([]);
      setExploreHistory([]);
    }
    if (episode > prevEpisodeRef.current) {
      if (typeof lastReward === "number" && Number.isFinite(lastReward)) {
        setRewards((r) => [...r, lastReward]);
      } else {
        setRewards((r) => [...r, 0]);
      }
      setExploitHistory((h) => [...h, exploitCount || 0]);
      setExploreHistory((h) => [...h, exploreCount || 0]);
    }

    prevEpisodeRef.current = episode;
  }, [episode, lastReward, exploitCount, exploreCount]);

  return (
    <div className="flex flex-col overflow-hidden w-full h-full items-center p-8">
      {/* Top Control Panel */}
      <header className="flex flex-col gap-2 mb-4">
        <h2 className="text-3xl font-bold">
          Deep Q-Network Playground: Gridworld
        </h2>
      </header>

      <div
        className={`flex flex-col items-center mb-4 mt-4 p-5 pt-2 pb-0 rounded-lg bg-slate-900 border border-slate-700 ${
          training ? "opacity-50 pointer-events-none" : ""
        }`}
        aria-disabled={training}
      >
        <header className="flex flex-col gap-2">
          <h4 className="text-2xl font-bold w-full text-start">
            Control Panel
          </h4>
        </header>
        <div className="w-full my-2">
          <h4 className="text-lg font-medium">Neural Network Architecture:</h4>
        </div>
        {/* Hidden layer controls */}
        <div className="flex gap-2 items-center flex-wrap">
          <label className="text-gray-300 text-sm">Number of Layers:</label>
          <select
            value={numHidden}
            onChange={(e) => {
              setNumHidden(Number(e.target.value));
              if (stopped) resetTrainParams();
            }}
            disabled={training}
            className="cursor-pointer bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[1, 2, 3].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          {hiddenSizes.map((size, idx) => {
            const sizeOptions = [8, 16, 32, 64, 128];
            return (
              <div key={idx} className="flex items-center gap-2">
                <label className="text-gray-300 text-xs">
                  Layer {idx + 1}:
                </label>
                <select
                  value={size}
                  disabled={training}
                  onChange={(e) => {
                    setHiddenSizes((sizes) =>
                      sizes.map((s, i) =>
                        i === idx ? Number(e.target.value) : s
                      )
                    );
                    if (stopped) resetTrainParams();
                  }}
                  className="cursor-pointer bg-slate-800 text-white px-2 py-1 rounded border border-slate-700"
                >
                  {sizeOptions.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>

        {/* Hyperparameter controls */}
        <div className="w-full my-2">
          <h4 className="text-lg font-medium">Hyperparameters:</h4>
        </div>
        <div className="flex gap-4 items-center flex-wrap">
          <label className="text-gray-300 text-sm">
            <HyperparamLabel
              tooltip="Step size for network weight updates"
              targetId="learning-rate-section"
            >
              Learning Rate
            </HyperparamLabel>
            :
          </label>
          <select
            value={learningRate}
            onChange={(e) => {
              setLearningRate(Number(e.target.value));
              if (stopped) resetTrainParams();
            }}
            disabled={training}
            className="cursor-pointer bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[0.00001, 0.00005, 0.0001, 0.0005, 0.001].map((val) => (
              <option key={val} value={val}>
                {val}
              </option>
            ))}
          </select>
          <label className="text-gray-300 text-sm">
            <HyperparamLabel
              tooltip="How quickly exploration decreases"
              targetId="epsilon-decay-section"
            >
              Epsilon Decay
            </HyperparamLabel>
            :
          </label>
          <select
            value={epsilonDecay}
            onChange={(e) => {
              setEpsilonDecay(Number(e.target.value));
              if (stopped) resetTrainParams();
            }}
            disabled={training}
            className="cursor-pointer bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[500, 1000, 2000, 5000, 10000].map((val) => (
              <option key={val} value={val}>
                {val}
              </option>
            ))}
          </select>
          <label className="text-gray-300 text-sm">
            <HyperparamLabel
              tooltip="Discount factor for future rewards"
              targetId="gamma-section"
            >
              Gamma
            </HyperparamLabel>
            :
          </label>
          <select
            value={gamma}
            onChange={(e) => {
              setGamma(Number(e.target.value));
              if (stopped) resetTrainParams();
            }}
            disabled={training}
            className="cursor-pointer bg-slate-800 text-white px-2 py-1 rounded"
          >
            {[0.9, 0.95, 0.99, 0.999].map((val) => (
              <option key={val} value={val}>
                {val}
              </option>
            ))}
          </select>
        </div>

        <div className="w-full my-2">
          <h4 className="text-lg font-medium">Agent Observables:</h4>
        </div>
        <div className="flex gap-3">
          <SensorButton
            title="Diagonal LiDAR"
            selected={diagonalSensors}
            onClick={() => {
              setDiagonalSensors((s) => !s);
              if (stopped) resetTrainParams();
            }}
            type="diagonal"
          />
          <SensorButton
            title="Cardinal LiDAR"
            selected={straightSensors}
            onClick={() => {
              setStraightSensors((s) => !s);
              if (stopped) resetTrainParams();
            }}
            type="straight"
          />
          <SensorButton
            title="Goal Localization"
            selected={goalLocalication}
            onClick={() => {
              setGoalLocalication((s) => !s);
              if (stopped) resetTrainParams();
            }}
            type="goal"
          />
          <SensorButton
            title="Pit Distance"
            selected={pitDistance}
            onClick={() => {
              setPitDistance((s) => !s);
              if (stopped) resetTrainParams();
            }}
            type="pit"
          />
        </div>

        <div className="w-full mt-4">
          <h4 className="text-lg font-medium">Q-Network Preview:</h4>
        </div>
        {/* Neural Network Panel - Below the grids */}
        <div className="flex flex-col items-center w-[960px]">
          <div className="w-full flex justify-center">
            <NNVisual
              hiddenSizes={hiddenSizes}
              numHidden={numHidden}
              width={960}
              minHeight={600}
              sensorFlags={{
                diagonalSensors,
                straightSensors,
                goalLocalication,
                pitDistance,
              }}
            />
          </div>
        </div>
      </div>

      {sensorError && (
        <div className=" text-red-400 text-sm mb-3">{sensorError}</div>
      )}
      <div className="flex gap-3 mb-8">
        {stopped && (
          <button
            className="cursor-pointer px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
            onClick={() => {
              resetTrainParams();
            }}
            disabled={training}
          >
            Reset
          </button>
        )}
        {!training ? (
          <button
            className="cursor-pointer px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
            onClick={() => {
              if (
                !diagonalSensors &&
                !straightSensors &&
                !goalLocalication &&
                !pitDistance
              ) {
                setSensorError(
                  "Please select at least one agent observable before starting training."
                );
                return;
              }
              setSensorError("");
              setStopped(false);
              if (stopped) {
                resumeTraining({ episodes: 100 });
              } else startTraining({ episodes: 100 });
            }}
            disabled={training}
          >
            {stopped ? "Resume Training" : "Start Training"}
          </button>
        ) : (
          <button
            className="cursor-pointer px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
            onClick={() => {
              stopTraining();
              setStopped(true);
            }}
            disabled={!training}
          >
            Stop Training
          </button>
        )}
      </div>

      {/* Three Grid Layout */}
      <div className="flex justify-center gap-8 mb-4">
        {/* Left Panel - Q-value arrows only */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Q-Value</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={cellQGrid}
            showGoal={false}
            showPitDistance={false}
            showDiagonalSensors={false}
            showStraightSensors={false}
          />
        </div>

        {/* Center Panel - Base GridWorld (no arrows/lines) */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Environment</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={null}
            showGoal={false}
            showPitDistance={false}
            showDiagonalSensors={false}
            showStraightSensors={false}
          />
        </div>

        {/* Right Panel - Agent visualizations (lidar, goal line, pit circle) */}
        <div className="flex flex-col items-center">
          <h3 className="text-xl font-semibold mb-4">Agent Observations</h3>
          <GridWorldCanvas
            gridSize={state.gridSize}
            cellSize={state.cellSize}
            agentPos={state.agentPos}
            blocks={state.blocks}
            pit={state.pit}
            goal={state.goalPos}
            actionQGrid={null}
            showGoal={goalLocalication}
            showPitDistance={pitDistance}
            showDiagonalSensors={diagonalSensors}
            showStraightSensors={straightSensors}
          />
        </div>
      </div>
      <div className="flex gap-6 text-gray-300 text-sm mb-4 items-end">
        <div>Episode: {episode}</div>
        <div>Total Steps: {totalSteps}</div>
        <div>
          Last Episode Reward:{" "}
          {typeof lastReward === "number" && Number.isFinite(lastReward)
            ? lastReward.toFixed(3)
            : "—"}
        </div>
        <div>Epsilon: {epsilon.toFixed(3)}</div>
        <div>Avg Reward (10): {avgReward.toFixed(3)}</div>
      </div>

      {/* Reward Chart placed beneath the labels */}
      <div className="mb-8">
        <div className="w-full flex justify-center mb-8 mt-4">
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-700">
            <RewardChart
              rewards={rewards}
              currentEpisode={episode}
              hoveredEpisode={hoveredEpisode}
              onHoverEpisode={setHoveredEpisode}
              width={720}
              height={200}
            />
            <StepCountChart
              currentEpisode={episode}
              exploitHistory={exploitHistory}
              exploreHistory={exploreHistory}
              hoveredEpisode={hoveredEpisode}
              onHoverEpisode={setHoveredEpisode}
              width={720}
              height={200}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <InteractivePane />
      <DQNExplain />
    </div>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);
