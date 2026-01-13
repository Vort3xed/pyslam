"use client";

import { useRef, useEffect, useState, useCallback } from "react";

interface Position {
  x: number;
  y: number;
  z: number;
  yaw: number;
}

interface MinimapProps {
  position: Position;
  pathHistory: Position[];
  onReset: () => void;
}

export default function Minimap({ position, pathHistory, onReset }: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(10); // pixels per unit
  const [offset, setOffset] = useState({ x: 0, y: 0 }); // pan offset
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const offsetStartRef = useRef({ x: 0, y: 0 });

  // Canvas size
  const canvasSize = 200;

  // Draw the minimap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Calculate center
    const centerX = canvasSize / 2 + offset.x;
    const centerY = canvasSize / 2 + offset.y;

    // Draw grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 1;
    const gridSpacing = zoom * 1; // 1 unit grid

    // Vertical lines
    for (let x = centerX % gridSpacing; x < canvasSize; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasSize);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = centerY % gridSpacing; y < canvasSize; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasSize, y);
      ctx.stroke();
    }

    // Draw path history as a line trace
    if (pathHistory.length > 1) {
      ctx.strokeStyle = "#4ade80"; // Green trace
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();

      for (let i = 0; i < pathHistory.length; i++) {
        const pos = pathHistory[i];
        // Using X and Z for bird's eye view (Y is up in SLAM)
        const screenX = centerX + pos.x * zoom;
        const screenY = centerY - pos.z * zoom; // Negative because canvas Y is inverted

        if (i === 0) {
          ctx.moveTo(screenX, screenY);
        } else {
          ctx.lineTo(screenX, screenY);
        }
      }

      ctx.stroke();
    }

    // Draw starting point
    if (pathHistory.length > 0) {
      const startPos = pathHistory[0];
      const startX = centerX + startPos.x * zoom;
      const startY = centerY - startPos.z * zoom;

      ctx.fillStyle = "#60a5fa"; // Blue
      ctx.beginPath();
      ctx.arc(startX, startY, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw current position as an arrow/triangle
    const curX = centerX + position.x * zoom;
    const curY = centerY - position.z * zoom;

    // Arrow pointing in yaw direction
    ctx.save();
    ctx.translate(curX, curY);
    ctx.rotate(-position.yaw); // Negative because canvas rotation is clockwise

    // Draw arrow
    ctx.fillStyle = "#f87171"; // Red
    ctx.beginPath();
    ctx.moveTo(0, -8); // Point
    ctx.lineTo(-5, 5); // Bottom left
    ctx.lineTo(5, 5); // Bottom right
    ctx.closePath();
    ctx.fill();

    // Draw center dot
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(0, 0, 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Draw scale indicator
    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${(1 / zoom * 10).toFixed(1)}m`, 5, canvasSize - 5);

  }, [position, pathHistory, zoom, offset]);

  // Handle zoom
  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev * 1.5, 100));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev / 1.5, 1));
  };

  // Handle wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.min(Math.max(prev * delta, 1), 100));
  }, []);

  // Handle drag pan
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    offsetStartRef.current = { ...offset };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;

    setOffset({
      x: offsetStartRef.current.x + dx,
      y: offsetStartRef.current.y + dy,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle touch events for mobile
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setIsDragging(true);
      dragStartRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      offsetStartRef.current = { ...offset };
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging || e.touches.length !== 1) return;

    const dx = e.touches[0].clientX - dragStartRef.current.x;
    const dy = e.touches[0].clientY - dragStartRef.current.y;

    setOffset({
      x: offsetStartRef.current.x + dx,
      y: offsetStartRef.current.y + dy,
    });
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  // Center on current position
  const handleCenter = () => {
    setOffset({
      x: -position.x * zoom,
      y: position.z * zoom,
    });
  };

  return (
    <>
      <div className="minimap-container">
        <canvas
          ref={canvasRef}
          width={canvasSize}
          height={canvasSize}
          className="minimap-canvas"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          style={{ cursor: isDragging ? "grabbing" : "grab" }}
        />
      </div>

      <div className="zoom-controls">
        <button onClick={handleZoomIn} title="Zoom In">+</button>
        <button onClick={handleZoomOut} title="Zoom Out">−</button>
        <button onClick={handleCenter} title="Center on Position">⌖</button>
        <button onClick={onReset} title="Reset Position">↺</button>
      </div>
    </>
  );
}
