import React, { useMemo, useState } from "react";

export default function NNVisual({ hiddenSizes = [64, 64], numHidden = 2, width = 960, minHeight = 400, minNodeGap = 16, sensorFlags = { diagonalSensors: true, straightSensors: true, goalLocalization: true, pitDistance: true } }) {
  const data = useMemo(() => {
    const flags = sensorFlags || {};
    const diag = flags.diagonalSensors !== undefined ? flags.diagonalSensors : true;
    const straight = flags.straightSensors !== undefined ? flags.straightSensors : true;
    const goalActive = (flags.goalLocalization ?? flags.goalLocalication ?? true);
    const pitActive = flags.pitDistance !== undefined ? flags.pitDistance : true;

    const inputLabels = [];
    if (goalActive) {
      inputLabels.push("Observation: Goal dx", "Observation: Goal dy");
    }
    const dirEntries = [
      { name: "NW", type: "diag" },
      { name: "N", type: "straight" },
      { name: "NE", type: "diag" },
      { name: "E", type: "straight" },
      { name: "SE", type: "diag" },
      { name: "S", type: "straight" },
      { name: "SW", type: "diag" },
      { name: "W", type: "straight" },
    ];
    dirEntries.forEach(d => {
      if ((d.type === "diag" && diag) || (d.type === "straight" && straight)) {
        inputLabels.push(`Observation: ${d.name} LiDAR`);
      }
    });
    if (pitActive) {
      inputLabels.push("Observation: Pit Proximity");
    }

    const inputDim = inputLabels.length;
    const outputDim = 4;
    const hs = Array.isArray(hiddenSizes) ? hiddenSizes.slice(0, numHidden) : [];
    const effectiveHidden = hs.length ? hs : [64];
    const layerNodeCounts = [inputDim, ...effectiveHidden, outputDim];

    const kernels = [];
    const biases = [];
    let maxAbs = 0;
    const rand = () => (Math.random() * 2 - 1);

    for (let li = 0; li < layerNodeCounts.length - 1; li++) {
      const rows = layerNodeCounts[li];
      const cols = layerNodeCounts[li + 1];
      const mat = Array.from({ length: rows }, () => Array.from({ length: cols }, () => rand()));
      kernels.push(mat);
      const b = Array.from({ length: cols }, () => rand());
      biases.push(b);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = Math.abs(mat[i][j]);
          if (v > maxAbs) maxAbs = v;
        }
      }
      for (let j = 0; j < cols; j++) {
        const v = Math.abs(b[j]);
        if (v > maxAbs) maxAbs = v;
      }
    }

    return { kernels, biases, maxAbs: maxAbs || 1e-6, layerNodeCounts, inputLabels };
  }, [hiddenSizes, numHidden, sensorFlags?.diagonalSensors, sensorFlags?.straightSensors, sensorFlags?.pitDistance, sensorFlags?.goalLocalization, sensorFlags?.goalLocalication]);

  const [hover, setHover] = useState(null);

  if (!data) return <div className="text-sm text-gray-400">No model</div>;

  const marginX = 40;
  const marginY = 30;
  const layerCount = data.layerNodeCounts.length;
  const usableWidth = width - marginX * 2;
  const maxNodes = Math.max(...data.layerNodeCounts);
  const requiredHeight = (maxNodes + 1) * minNodeGap + marginY * 2;
  const height = Math.max(minHeight, requiredHeight);
  const usableHeight = height - marginY * 2;
  const layerSpacing = usableWidth / (layerCount - 1);

  const positions = data.layerNodeCounts.map((count, li) => {
    const x = marginX + li * layerSpacing;
    const stepY = usableHeight / (count + 1);
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({ x, y: marginY + (i + 1) * stepY });
    }
    return arr;
  });

  const colorFor = (v) => {
    const t = Math.min(1, Math.abs(v) / data.maxAbs);
    const a = 0.2 + 0.8 * t;
    return v >= 0 ? `rgba(239, 68, 68, ${a})` : `rgba(59, 130, 246, ${a})`;
  };

  const strokeW = (v) => 0.5 + 2.5 * Math.min(1, Math.abs(v) / data.maxAbs);

  const getTooltip = (li, idx) => {
    if (li === 0) {
      return data.inputLabels[idx] ?? "Input Node";
    }
    if (li === layerCount - 1) {
      const actions = ["Up", "Down", "Left", "Right"];
      return `Action: ${actions[idx]}`;
    }
    return "Hidden Node";
  };

  const hoveredPos = hover ? positions[hover.li][hover.idx] : null;
  const tooltipText = hover ? getTooltip(hover.li, hover.idx) : "";

  return (
    <div className="relative inline-block">
      <svg width={width} height={height}>
        {/* Layer Labels */}
        {positions.map((_, li) => {
          let label = `Hidden Layer ${li}`;
          if (li === 0) label = "Input Layer";
          else if (li === layerCount - 1) label = "Output Layer";
          
          return (
            <text
              key={`label-${li}`}
              x={marginX + li * layerSpacing}
              y={20}
              textAnchor="middle"
              className="text-xs fill-slate-400 font-mono"
            >
              {label}
            </text>
          );
        })}

        {/* Edges (Low Highlight) */}
        {data.kernels.map((mat, li) =>
        mat.map((row, i) =>
          row.map((w, j) => {
            const p = positions[li][i];
            const q = positions[li + 1][j];
            const isHi = hover && hover.li === li + 1 && hover.idx === j;
            if (isHi) return null;
            return (
              <line
                key={`e-${li}-${i}-${j}`}
                x1={p.x}
                y1={p.y}
                x2={q.x}
                y2={q.y}
                stroke={colorFor(w)}
                strokeWidth={strokeW(w)}
              />
            );
          })
        )
      )}
      {data.kernels.map((mat, li) =>
        mat.map((row, i) =>
          row.map((w, j) => {
            const p = positions[li][i];
            const q = positions[li + 1][j];
            const isHi = hover && hover.li === li + 1 && hover.idx === j;
            if (!isHi) return null;
            return (
              <line
                key={`eh-${li}-${i}-${j}`}
                x1={p.x}
                y1={p.y}
                x2={q.x}
                y2={q.y}
                stroke="#facc15"
                strokeWidth={strokeW(w) + 1.2}
              />
            );
          })
        )
      )}
      {positions.map((nodes, li) =>
        nodes.map((n, idx) => {
          const r = 6;
          const bias = li === 0 ? 0 : (data.biases[li - 1][idx] ?? 0);
          const isOutput = li === layerCount - 1;
          const isInput = li === 0;
          
          // Output arrows: 0=Up, 1=Down, 2=Left, 3=Right
          let arrow = null;
          if (isOutput) {
            if (idx === 0) arrow = "🡹";
            if (idx === 1) arrow = "🡻";
            if (idx === 2) arrow = "🡸";
            if (idx === 3) arrow = "🡺";
          }

          // Input icons based on label
          let inputIcon = null;
          if (isInput) {
            const lbl = data.inputLabels[idx] || "";
            if (lbl.includes("Goal dx")) inputIcon = "↔";
            else if (lbl.includes("Goal dy")) inputIcon = "↕";
            else if (lbl.includes("NW")) inputIcon = "🡤";
            else if (lbl.includes("NE")) inputIcon = "🡥";
            else if (lbl.includes("SE")) inputIcon = "🡦";
            else if (lbl.includes("SW")) inputIcon = "🡧";
            else if (lbl.includes(" N ")) inputIcon = "🡡";
            else if (lbl.includes(" E ")) inputIcon = "🡢";
            else if (lbl.includes(" S ")) inputIcon = "🡣";
            else if (lbl.includes(" W ")) inputIcon = "🡠";
            else if (lbl.toLowerCase().includes("pit")) inputIcon = "◎";
          }

          return (
            <g key={`n-${li}-${idx}`}>
              <circle
                cx={n.x}
                cy={n.y}
                r={r}
                fill={colorFor(bias)}
                stroke="#ddd"
                strokeWidth="1"
                onMouseEnter={() => setHover({ li, idx })}
                onMouseLeave={() => setHover(null)}
              />
              {inputIcon && (
                <text
                  x={n.x-24}
                  y={n.y-3}
                  dy={1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={24}
                  fill="white"
                  fontWeight="bold"
                  className="pointer-events-none"
                >
                  {inputIcon}
                </text>
              )}
              {arrow && (
                <text
                  x={n.x+24}
                  y={n.y-2}
                  dy={1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={24}
                  fill="white"
                  fontWeight="bold"
                  className="pointer-events-none"
                >
                  {arrow}
                </text>
              )}
            </g>
          );
        })
      )}
      </svg>
      {hover && hoveredPos && tooltipText != "Hidden Node" && (
        <div
          className="absolute z-50 px-3 py-2 bg-slate-950 text-slate-200 text-xs rounded border border-slate-700 shadow-lg whitespace-nowrap pointer-events-none"
          style={{
            left: hoveredPos.x,
            top: hoveredPos.y,
            transform: 'translate(-50%, -140%)'
          }}
        >
          {tooltipText}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-950"></div>
        </div>
      )}
    </div>
  );
}