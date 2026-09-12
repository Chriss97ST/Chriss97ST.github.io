import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Upload, Columns, Sparkles, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import type { RetroSettings } from '../types/retro';
import { WebGLRetroRenderer } from '../services/webglRenderer';
import { renderRetroOverlay } from '../services/canvasOverlay';
import { generateSampleImages, type SampleImage } from '../constants/sampleImages';

interface StudioViewProps {
  imageElement: HTMLImageElement | null;
  onImageLoaded: (img: HTMLImageElement) => void;
  settings: RetroSettings;
  onRandomizeSeed: () => void;
}

export const StudioView: React.FC<StudioViewProps> = ({
  imageElement,
  onImageLoaded,
  settings,
  onRandomizeSeed,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const glCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGLRetroRenderer | null>(null);

  const [compareSplit, setCompareSplit] = useState<number>(50);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);
  const [zoom, setZoom] = useState<number>(1);
  const [sampleImages, setSampleImages] = useState<SampleImage[]>([]);

  // Initialize samples on mount
  useEffect(() => {
    setSampleImages(generateSampleImages());
  }, []);

  // Initialize WebGL renderer
  useEffect(() => {
    if (!glCanvasRef.current) return;
    const renderer = new WebGLRetroRenderer(glCanvasRef.current);
    rendererRef.current = renderer;

    return () => {
      renderer.destroy();
      rendererRef.current = null;
    };
  }, []);

  // Render loop whenever image or settings change
  const renderFrame = useCallback(() => {
    if (!imageElement || !glCanvasRef.current || !overlayCanvasRef.current || !rendererRef.current) return;

    const overlayCanvas = overlayCanvasRef.current;
    const w = imageElement.naturalWidth || imageElement.width || 800;
    const h = imageElement.naturalHeight || imageElement.height || 600;

    // 1. Render base WebGL
    rendererRef.current.render(imageElement, settings, 0);

    // 2. Adjust overlay canvas dimensions
    if (overlayCanvas.width !== w || overlayCanvas.height !== h) {
      overlayCanvas.width = w;
      overlayCanvas.height = h;
    }

    const oCtx = overlayCanvas.getContext('2d');
    if (!oCtx) return;

    oCtx.clearRect(0, 0, w, h);

    if (isComparing) {
      // Split mode: Show original on left side
      const splitX = Math.round((w * compareSplit) / 100);

      // Draw original image on left side of split
      oCtx.save();
      oCtx.beginPath();
      oCtx.rect(0, 0, splitX, h);
      oCtx.clip();
      oCtx.drawImage(imageElement, 0, 0, w, h);
      oCtx.restore();

      // Draw Retro Overlays (only on the right side)
      oCtx.save();
      oCtx.beginPath();
      oCtx.rect(splitX, 0, w - splitX, h);
      oCtx.clip();
      renderRetroOverlay(oCtx, w, h, settings);
      oCtx.restore();

      // Divider line
      oCtx.strokeStyle = '#39ff14';
      oCtx.lineWidth = Math.max(2, Math.round(w * 0.003));
      oCtx.beginPath();
      oCtx.moveTo(splitX, 0);
      oCtx.lineTo(splitX, h);
      oCtx.stroke();
    } else {
      // Full Retro mode: Draw all 2D overlays
      renderRetroOverlay(oCtx, w, h, settings);
    }
  }, [imageElement, settings, isComparing, compareSplit]);

  useEffect(() => {
    renderFrame();
  }, [renderFrame]);

  // Handle Drag and Drop
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImageFromFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      loadImageFromFile(e.target.files[0]);
    }
  };

  const loadImageFromFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (ev.target?.result) {
        const img = new Image();
        img.onload = () => {
          onImageLoaded(img);
        };
        img.src = ev.target.result as string;
      }
    };
    reader.readAsDataURL(file);
  };

  const loadSample = (sample: SampleImage) => {
    const img = new Image();
    img.onload = () => {
      onImageLoaded(img);
    };
    img.src = sample.dataUrl;
  };

  // Split Drag Handlers
  const handleSplitMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingSplit(true);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingSplit || !glCanvasRef.current) return;
    const rect = glCanvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = Math.round((x / rect.width) * 100);
    setCompareSplit(percent);
  };

  const handleMouseUp = () => {
    setIsDraggingSplit(false);
  };

  return (
    <div
      className="studio-container"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Viewport Area */}
      <div
        className="canvas-viewport"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        {imageElement ? (
          <div
            className="canvas-transform-wrapper"
            style={{ transform: `scale(${zoom})` }}
          >
            <div className="canvas-frame-outer">
              {/* WebGL Canvas */}
              <canvas ref={glCanvasRef} className="main-render-canvas" />

              {/* Overlay 2D Canvas */}
              <canvas
                ref={overlayCanvasRef}
                className="main-render-canvas"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  pointerEvents: 'none',
                }}
              />

              {/* Compare Split Handle */}
              {isComparing && (
                <div
                  className="split-handle-line"
                  style={{ left: `${compareSplit}%` }}
                  onMouseDown={handleSplitMouseDown}
                >
                  <div className="split-handle-knob">
                    <span>◄ Original | Retro ►</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-upload-card">
            <div className="upload-icon-circle">
              <Upload size={36} />
            </div>
            <h3>Foto hier ablegen oder hochladen</h3>
            <p>Unterstützt JPG, PNG, WebP – die gesamte Bildentwicklung erfolgt lokal im Browser.</p>

            <label className="upload-btn">
              <span>Foto von Festplatte wählen</span>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
            </label>

            {/* Quick Samples */}
            <div className="samples-section">
              <span className="samples-title">Oder ein Beispiel-Foto testen:</span>
              <div className="samples-grid">
                {sampleImages.map((s) => (
                  <button
                    key={s.id}
                    className="sample-card-btn"
                    onClick={() => loadSample(s)}
                  >
                    <img src={s.dataUrl} alt={s.name} />
                    <span>{s.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Viewport Toolbar */}
      {imageElement && (
        <div className="viewport-toolbar">
          <label className="toolbar-btn" title="Anderes Foto hochladen">
            <Upload size={16} />
            <span>Neues Foto</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
          </label>

          <button
            className={`toolbar-btn ${isComparing ? 'active' : ''}`}
            onClick={() => {
              setIsComparing(!isComparing);
              setCompareSplit(50);
            }}
            title="Vorher / Nachher Vergleichs-Schieberegler"
          >
            <Columns size={16} />
            <span>{isComparing ? 'Vergleich An (50%)' : 'Vorher/Nachher'}</span>
          </button>

          <button
            className="toolbar-btn"
            onClick={onRandomizeSeed}
            title="Zufälligen Unikat-Seed generieren"
          >
            <Sparkles size={16} />
            <span>Seed #{settings.seed}</span>
          </button>

          <div className="zoom-controls">
            <button
              className="zoom-btn"
              onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}
              title="Verkleinern"
            >
              <ZoomOut size={15} />
            </button>
            <span className="zoom-text">{Math.round(zoom * 100)}%</span>
            <button
              className="zoom-btn"
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}
              title="Vergrößern"
            >
              <ZoomIn size={15} />
            </button>
            <button
              className="zoom-btn"
              onClick={() => setZoom(1)}
              title="Auf 100% zurücksetzen"
            >
              <Maximize2 size={15} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
