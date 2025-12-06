import React, { useMemo } from "react";
import * as tf from "@tensorflow/tfjs";

export default function NNVisual({ model, width = 720, height = 1300, refreshKey }) {
  const data = useMemo(() => {
    if (!model || !model.layers || model.layers.length === 0) return null;
    const denseLayers = model.layers.filter((l) => l.getClassName && l.getClassName() === "Dense");
    if (denseLayers.length === 0) return null;

    const kernels = [];
    const biases = [];
    let maxAbs = 0;

    denseLayers.forEach((l) => {
      const ws = l.getWeights();
      const k = ws[0];
      const b = ws[1];
      const kArr = k.arraySync();
      const bArr = b ? b.arraySync() : [];
      kernels.push(kArr);
      biases.push(bArr);
      for (let i = 0; i < kArr.length; i++) {
        for (let j = 0; j < kArr[i].length; j++) {
          const v = Math.abs(kArr[i][j]);
          if (v > maxAbs) maxAbs = v;
        }
      }
      for (let j = 0; j < bArr.length; j++) {
        const v = Math.abs(bArr[j]);
        if (v > maxAbs) maxAbs = v;
      }
    });

    const inDim = kernels[0].length;
    const layerNodeCounts = [inDim];
    kernels.forEach((mat) => layerNodeCounts.push(mat[0].length));

    return { kernels, biases, maxAbs: maxAbs || 1e-6, layerNodeCounts };
  }, [model, refreshKey]);

  if (!data) return <div className="text-sm text-gray-400">No model</div>;

  const marginX = 40;
  const marginY = 20;
  const layerCount = data.layerNodeCounts.length;
  const usableWidth = width - marginX * 2;
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

  return (
    <svg width={width} height={height}>
      {data.kernels.map((mat, li) =>
        mat.map((row, i) =>
          row.map((w, j) => {
            const p = positions[li][i];
            const q = positions[li + 1][j];
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
      {positions.map((nodes, li) =>
        nodes.map((n, idx) => {
          const r = 8;
          const bias = li === 0 ? 0 : (data.biases[li - 1][idx] ?? 0);
          return (
            <circle
              key={`n-${li}-${idx}`}
              cx={n.x}
              cy={n.y}
              r={r}
              fill={colorFor(bias)}
              stroke="#ddd"
              strokeWidth="1"
            />
          );
        })
      )}
    </svg>
  );
}