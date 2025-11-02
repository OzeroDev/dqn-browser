import React, { useEffect, useRef } from "react";

export default function CartPoleCanvas({ obs }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = "#1f2937"; // gray-800
    ctx.fillRect(0, 0, width, height);

    // Ground
    ctx.strokeStyle = "#9ca3af"; // gray-400
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, height - 60);
    ctx.lineTo(width, height - 60);
    ctx.stroke();

    // Defaults if obs not ready
    const x = obs ? obs[0] : 0.0;
    const theta = obs ? obs[2] : 0.0;

    // Map x [-2.4, 2.4] to pixels
    const X_THRESHOLD = 2.4;
    const cartX =
      width / 2 + (x / X_THRESHOLD) * (width * 0.4);
    const cartY = height - 90;
    const cartW = 100;
    const cartH = 40;

    // Cart
    ctx.fillStyle = "#3b82f6"; // blue-500
    ctx.fillRect(cartX - cartW / 2, cartY - cartH / 2, cartW, cartH);

    // Pole
    const poleLenPx = 120; // visual length
    const pivotX = cartX;
    const pivotY = cartY - cartH / 2;
    const poleEndX = pivotX + poleLenPx * Math.sin(theta);
    const poleEndY = pivotY - poleLenPx * Math.cos(theta);

    ctx.strokeStyle = "#f59e0b"; // amber-500
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();

    // Pivot circle
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 6, 0, Math.PI * 2);
    ctx.fill();
  }, [obs]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={300}
      className="bg-gray-800 rounded"
    />
  );
}