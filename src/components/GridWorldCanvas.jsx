import React, { useRef, useEffect } from "react";

export default function GridWorldCanvas({
  gridSize,
  cellSize,
  agentPos,
  blocks,
  pit,
  goal,
  actionQGrid, // 2D array of q-values for each cell or null
}) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = gridSize * cellSize;
    canvas.height = gridSize * cellSize;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "rgb(18,18,20)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "rgb(60,60,70)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(gridSize * cellSize, i * cellSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, gridSize * cellSize);
      ctx.stroke();
    }

    ctx.fillStyle = "rgb(120,120,130)";
    blocks.forEach(([r, c]) => {
      ctx.beginPath();
      ctx.roundRect(c * cellSize + 2, r * cellSize + 2, cellSize - 4, cellSize - 4, 6);
      ctx.fill();
    });

    ctx.fillStyle = "rgb(200,70,70)";
    const [pr, pc] = pit;
    ctx.beginPath();
    ctx.roundRect(pc * cellSize + 6, pr * cellSize + 6, cellSize - 12, cellSize - 12, 10);
    ctx.fill();

    ctx.fillStyle = "rgb(90,200,90)";
    const [gr, gc] = goal;
    ctx.beginPath();
    ctx.roundRect(gc * cellSize + 2, gr * cellSize + 2, cellSize - 4, cellSize - 4, 8);
    ctx.fill();

    ctx.fillStyle = "rgb(66,135,245)";
    const [ar, ac] = agentPos;
    const cx = ac * cellSize + cellSize / 2;
    const cy = ar * cellSize + cellSize / 2;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.floor(cellSize / 3), 0, Math.PI * 2);
    ctx.fill();

    // draw action arrows for every valid cell if provided
    if (Array.isArray(actionQGrid)) {
      const gs = gridSize;
      const minAlpha = 0.12;

      const drawTriCell = (r, c, dir, alpha) => {
        const cx = c * cellSize + cellSize / 2;
        const cy = r * cellSize + cellSize / 2;
        const s = Math.floor(cellSize * 0.12);
        const offset = Math.floor(cellSize * 0.24);
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.beginPath();
        if (dir === 0) {
          const nx = cx;
          const ny = cy - offset;
          ctx.moveTo(nx, ny - s);
          ctx.lineTo(nx - s * 0.6, ny + s * 0.6);
          ctx.lineTo(nx + s * 0.6, ny + s * 0.6);
        } else if (dir === 1) {
          const nx = cx;
          const ny = cy + offset;
          ctx.moveTo(nx, ny + s);
          ctx.lineTo(nx - s * 0.6, ny - s * 0.6);
          ctx.lineTo(nx + s * 0.6, ny - s * 0.6);
        } else if (dir === 2) {
          const nx = cx - offset;
          const ny = cy;
          ctx.moveTo(nx - s, ny);
          ctx.lineTo(nx + s * 0.6, ny - s * 0.6);
          ctx.lineTo(nx + s * 0.6, ny + s * 0.6);
        } else if (dir === 3) {
          const nx = cx + offset;
          const ny = cy;
          ctx.moveTo(nx + s, ny);
          ctx.lineTo(nx - s * 0.6, ny - s * 0.6);
          ctx.lineTo(nx - s * 0.6, ny + s * 0.6);
        }
        ctx.closePath();
        ctx.fill();
      };

      for (let r = 0; r < gs; r++) {
        for (let c = 0; c < gs; c++) {
          const qArr = actionQGrid[r] && actionQGrid[r][c];
          if (!qArr) continue;
          // skip blocks/pit/goal drawing safety
          const isBlock = blocks.some((b) => b[0] === r && b[1] === c);
          const isPit = pit && pit[0] === r && pit[1] === c;
          const isGoal = goal && goal[0] === r && goal[1] === c;
          if (isBlock || isPit || isGoal) continue;

          const vals = qArr.map((v) => Number(v));
          const min = Math.min(...vals);
          const max = Math.max(...vals);
          const range = Math.abs(max - min);
          const alphas = vals.map((v) => {
            if (range < 1e-6) return 0.5;
            const norm = (v - min) / range;
            return Math.min(1, Math.max(minAlpha, norm));
          });

          for (let a = 0; a < 4; a++) {
            drawTriCell(r, c, a, alphas[a]);
          }
        }
      }
    }
  }, [gridSize, cellSize, agentPos, blocks, pit, goal, actionQGrid]);

  return <canvas ref={canvasRef} className="border border-gray-700 rounded" />;
}