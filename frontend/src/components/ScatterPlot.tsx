'use client';

import React, { useEffect, useRef, useMemo } from 'react';

interface ScatterPlotProps {
  data: {
    x: number[];
    y: number[];
    colors: string[];
    labels?: string[];
    values?: number[];
  };
  colorBy: 'cluster' | 'pseudotime' | 'value';
  width?: number;
  height?: number;
  title?: string;
}

export function ScatterPlot({ data, colorBy, width = 500, height = 400, title }: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const colorData = useMemo(() => {
    if (colorBy === 'cluster') {
      return data.colors;
    } else if (colorBy === 'pseudotime' && data.values) {
      const min = Math.min(...data.values);
      const max = Math.max(...data.values);
      return data.values.map(v => {
        const norm = (v - min) / (max - min);
        const r = Math.round(255 * norm);
        const g = Math.round(255 * (1 - norm) * 0.8);
        const b = Math.round(100 + 155 * (1 - norm));
        return `rgb(${r},${g},${b})`;
      });
    } else if (colorBy === 'value' && data.values) {
      const min = Math.min(...data.values);
      const max = Math.max(...data.values);
      return data.values.map(v => {
        const norm = (v - min) / (max - min);
        // Plasma-like colormap
        const r = Math.round(255 * Math.min(1, 0.5 + norm));
        const g = Math.round(255 * Math.max(0, norm - 0.3));
        const b = Math.round(255 * Math.max(0, 1 - norm * 1.5));
        return `rgb(${r},${g},${b})`;
      });
    }
    return data.colors;
  }, [data, colorBy]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate scale
    const padding = 40;
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;

    const xMin = Math.min(...data.x);
    const xMax = Math.max(...data.x);
    const yMin = Math.min(...data.y);
    const yMax = Math.max(...data.y);

    const xScale = (v: number) => padding + ((v - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (v: number) => height - padding - ((v - yMin) / (yMax - yMin)) * plotHeight;

    // Draw points
    for (let i = 0; i < data.x.length; i++) {
      const x = xScale(data.x[i]);
      const y = yScale(data.y[i]);
      
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = colorData[i] || '#3b82f6';
      ctx.fill();
    }

    // Draw title
    if (title) {
      ctx.font = '14px Inter, sans-serif';
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'center';
      ctx.fillText(title, width / 2, 20);
    }
  }, [data, colorData, width, height, title]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border rounded-lg"
      />
    </div>
  );
}
