import React, { useRef, useEffect } from "react";

export default function GridWorldCanvas({
  gridSize,
  cellSize,
  agentPos,
  blocks,
  pit,
  goal,
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
  }, [gridSize, cellSize, agentPos, blocks, pit, goal]);

  return <canvas ref={canvasRef} className="border border-gray-700 rounded" />;
}