import React, { useRef, useEffect, useState, useCallback } from 'react';
import { SwitchCamera, Zap, Timer, Sparkles, X } from 'lucide-react';
import type { RetroSettings } from '../types/retro';
import { WebGLRetroRenderer } from '../services/webglRenderer';
import { renderRetroOverlay } from '../services/canvasOverlay';
import { audioEffects } from '../services/audioEffects';

interface CameraViewProps {
  settings: RetroSettings;
  onPhotoCaptured: (img: HTMLImageElement) => void;
  onClose: () => void;
  onRandomizeSeed: () => void;
}

export const CameraView: React.FC<CameraViewProps> = ({
  settings,
  onPhotoCaptured,
  onClose,
  onRandomizeSeed,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rendererRef = useRef<WebGLRetroRenderer | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  const [hasTorch, setHasTorch] = useState<boolean>(false);
  const [torchOn, setTorchOn] = useState<boolean>(false);
  const [timerSeconds, setTimerSeconds] = useState<number>(0);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [flashEffect, setFlashEffect] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initialize Camera Stream
  const initCamera = useCallback(async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      setErrorMsg(null);
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch {
          // Play was interrupted or not allowed yet
        }
      }

      // Check for torch capability
      const track = stream.getVideoTracks()[0];
      if (track) {
        const capabilities = (track.getCapabilities?.() as unknown as { torch?: boolean }) || {};
        setHasTorch(!!capabilities.torch);
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setErrorMsg('Kamerazugriff verweigert oder keine Kamera gefunden. Bitte Berechtigung im Browser erteilen.');
    }
  }, [facingMode]);

  useEffect(() => {
    initCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.destroy();
      }
    };
  }, [initCamera]);

  // Real-time render loop on Canvas
  useEffect(() => {
    if (!canvasRef.current) return;
    const renderer = new WebGLRetroRenderer(canvasRef.current);
    rendererRef.current = renderer;

    let startTime = performance.now();

    const loop = (now: number) => {
      if (videoRef.current && videoRef.current.readyState >= 2 && rendererRef.current && canvasRef.current) {
        const time = (now - startTime) / 1000;
        rendererRef.current.render(videoRef.current, settings, time);

        // Render 2D overlays on top
        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = canvasRef.current.width;
        overlayCanvas.height = canvasRef.current.height;
        const oCtx = overlayCanvas.getContext('2d')!;
        renderRetroOverlay(oCtx, overlayCanvas.width, overlayCanvas.height, settings);

        const ctx2d = canvasRef.current.getContext('2d');
        if (ctx2d) {
          ctx2d.drawImage(overlayCanvas, 0, 0);
        }
      }
      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      renderer.destroy();
    };
  }, [settings]);

  // Toggle Torch
  const toggleTorch = async () => {
    if (!streamRef.current) return;
    const track = streamRef.current.getVideoTracks()[0];
    if (track) {
      try {
        const nextState = !torchOn;
        await (track as unknown as { applyConstraints: (c: unknown) => Promise<void> }).applyConstraints({
          advanced: [{ torch: nextState }],
        });
        setTorchOn(nextState);
      } catch (err) {
        console.error('Torch error:', err);
      }
    }
  };

  // Flip Camera
  const flipCamera = () => {
    setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'));
  };

  // Trigger Shutter Capture
  const handleShutter = () => {
    if (countdown !== null) return;

    if (timerSeconds > 0) {
      let count = timerSeconds;
      setCountdown(count);
      audioEffects.playBeep();

      const timerInterval = setInterval(() => {
        count -= 1;
        if (count > 0) {
          setCountdown(count);
          audioEffects.playBeep();
        } else {
          clearInterval(timerInterval);
          setCountdown(null);
          executeCapture();
        }
      }, 1000);
    } else {
      executeCapture();
    }
  };

  const executeCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;

    // Flash animation and shutter sound
    audioEffects.playShutter();
    setFlashEffect(true);
    setTimeout(() => setFlashEffect(false), 250);

    // Grab high-res snapshot from video track
    const vid = videoRef.current;
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = vid.videoWidth || 1920;
    snapCanvas.height = vid.videoHeight || 1080;
    const sCtx = snapCanvas.getContext('2d')!;

    // If front camera, flip horizontally for natural selfie orientation
    if (facingMode === 'user') {
      sCtx.translate(snapCanvas.width, 0);
      sCtx.scale(-1, 1);
    }

    sCtx.drawImage(vid, 0, 0, snapCanvas.width, snapCanvas.height);

    const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.95);
    const img = new Image();
    img.onload = () => {
      onPhotoCaptured(img);
    };
    img.src = dataUrl;
  };

  return (
    <div className="camera-view-container">
      {/* Hidden Video Feed for Stream Sampling */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        style={{ display: 'none' }}
      />

      {/* Camera Header Bar */}
      <div className="camera-top-bar">
        <button className="cam-icon-btn" onClick={onClose} title="Schließen & zurück zum Studio">
          <X size={20} />
        </button>

        <div className="cam-top-actions">
          {hasTorch && (
            <button
              className={`cam-icon-btn ${torchOn ? 'active' : ''}`}
              onClick={toggleTorch}
              title="Blitzlicht / Taschenlampe"
            >
              <Zap size={20} />
            </button>
          )}

          <button
            className={`cam-icon-btn ${timerSeconds > 0 ? 'active' : ''}`}
            onClick={() => {
              const next = timerSeconds === 0 ? 3 : timerSeconds === 3 ? 10 : 0;
              setTimerSeconds(next);
            }}
            title={`Selbstauslöser: ${timerSeconds === 0 ? 'Aus' : timerSeconds + 's'}`}
          >
            <Timer size={20} />
            {timerSeconds > 0 && <span className="timer-badge">{timerSeconds}s</span>}
          </button>

          <button className="cam-icon-btn" onClick={onRandomizeSeed} title="Zufälliger Unikat-Seed">
            <Sparkles size={20} />
          </button>
        </div>
      </div>

      {/* Viewport Live Preview Canvas */}
      <div className="camera-canvas-wrapper">
        <canvas ref={canvasRef} className="camera-live-canvas" />

        {/* Flash Overlay */}
        {flashEffect && <div className="camera-flash-overlay" />}

        {/* Countdown Overlay */}
        {countdown !== null && (
          <div className="camera-countdown-overlay">
            <span className="countdown-number">{countdown}</span>
          </div>
        )}

        {/* Error Message */}
        {errorMsg && (
          <div className="camera-error-banner">
            <p>{errorMsg}</p>
            <button className="primary-btn" onClick={initCamera}>
              Erneut versuchen
            </button>
          </div>
        )}
      </div>

      {/* Camera Bottom Controls */}
      <div className="camera-bottom-bar">
        <div className="bottom-bar-left">
          {/* Preset hint */}
          <span className="live-filter-badge">Echtzeit-Shader: {settings.seed ? `Seed #${settings.seed}` : ''}</span>
        </div>

        {/* Shutter Button */}
        <div className="shutter-button-wrapper">
          <button
            className="shutter-outer-ring"
            onClick={handleShutter}
            title="Foto aufnehmen"
          >
            <div className="shutter-inner-button" />
          </button>
        </div>

        <div className="bottom-bar-right">
          <button className="cam-flip-btn" onClick={flipCamera} title="Kamera wechseln (Vorder-/Rückseite)">
            <SwitchCamera size={24} />
          </button>
        </div>
      </div>
    </div>
  );
};
