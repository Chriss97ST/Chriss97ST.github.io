import React, { useState, useEffect } from 'react';
import { X, Download, Share2, Copy, Check, Sparkles } from 'lucide-react';
import confetti from 'canvas-confetti';
import type { RetroSettings } from '../types/retro';
import {
  processHighResImage,
  downloadBlob,
  shareImage,
  copyImageToClipboard,
  type ExportOptions,
} from '../services/imageExporter';

interface ExportModalProps {
  imageElement: HTMLImageElement | null;
  settings: RetroSettings;
  onClose: () => void;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  imageElement,
  settings,
  onClose,
}) => {
  const [format, setFormat] = useState<'image/jpeg' | 'image/png' | 'image/webp'>('image/jpeg');
  const [quality, setQuality] = useState<number>(0.92);
  const [maxDimension, setMaxDimension] = useState<number | undefined>(undefined);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const [shareSupported, setShareSupported] = useState<boolean>(false);
  const [imageSizeInfo, setImageSizeInfo] = useState<string>('');

  useEffect(() => {
    setShareSupported(typeof navigator !== 'undefined' && !!navigator.share);
  }, []);

  // Process high-res render on parameter change
  useEffect(() => {
    if (!imageElement) return;

    let isMounted = true;
    setIsProcessing(true);

    const timer = setTimeout(async () => {
      try {
        const options: ExportOptions = {
          format,
          quality,
          maxDimension,
        };

        const result = await processHighResImage(imageElement, settings, options);

        if (isMounted) {
          setPreviewUrl(result.dataUrl);
          setCurrentBlob(result.blob);
          const sizeKb = Math.round(result.blob.size / 1024);
          const sizeMb = (result.blob.size / (1024 * 1024)).toFixed(2);
          const sizeDisplay = sizeKb > 1000 ? `${sizeMb} MB` : `${sizeKb} KB`;
          setImageSizeInfo(`${result.width} × ${result.height} px (${sizeDisplay})`);
          setIsProcessing(false);
        }
      } catch (err) {
        console.error('Export error:', err);
        if (isMounted) setIsProcessing(false);
      }
    }, 120);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [imageElement, settings, format, quality, maxDimension]);

  const handleDownload = () => {
    if (!currentBlob) return;

    const ext = format === 'image/jpeg' ? 'jpg' : format === 'image/png' ? 'png' : 'webp';
    const filename = `retrolens_${Date.now()}.${ext}`;

    downloadBlob(currentBlob, filename);

    // Fire Confetti
    confetti({
      particleCount: 80,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#ff7b00', '#ffd54f', '#39ff14', '#00f0ff', '#ff0077'],
    });
  };

  const handleShare = async () => {
    if (!currentBlob) return;
    const ext = format === 'image/jpeg' ? 'jpg' : format === 'image/png' ? 'png' : 'webp';
    const filename = `retrolens_${Date.now()}.${ext}`;
    await shareImage(currentBlob, filename);
  };

  const handleCopy = async () => {
    if (!currentBlob) return;
    const success = await copyImageToClipboard(currentBlob);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="export-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <Sparkles size={20} className="sparkle-icon" />
            <h3>Foto Entwickeln & Exportieren</h3>
          </div>
          <button className="close-btn" onClick={onClose} title="Schließen">
            <X size={20} />
          </button>
        </div>

        {/* Content Body */}
        <div className="modal-body-grid">
          {/* Left: Preview */}
          <div className="export-preview-box">
            {isProcessing ? (
              <div className="processing-spinner">
                <div className="spinner-ring" />
                <span>Foto wird in voller Auflösung entwickelt...</span>
              </div>
            ) : previewUrl ? (
              <img src={previewUrl} alt="Entwickelte Vorschau" className="export-preview-img" />
            ) : null}

            {imageSizeInfo && <div className="size-badge">{imageSizeInfo}</div>}
          </div>

          {/* Right: Settings & Actions */}
          <div className="export-settings-col">
            {/* Format Selection */}
            <div className="export-group">
              <label className="group-label">Dateiformat</label>
              <div className="format-pills">
                {[
                  { id: 'image/jpeg', label: 'JPG (Klassisch)' },
                  { id: 'image/png', label: 'PNG (Verlustfrei)' },
                  { id: 'image/webp', label: 'WebP (Kompakt)' },
                ].map((f) => (
                  <button
                    key={f.id}
                    className={`format-pill ${format === f.id ? 'active' : ''}`}
                    onClick={() => setFormat(f.id as any)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Quality Slider (for JPG/WebP) */}
            {format !== 'image/png' && (
              <div className="export-group">
                <div className="slider-header">
                  <label className="group-label">Kompression & Qualität</label>
                  <span className="val-badge">{Math.round(quality * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.4"
                  max="1.0"
                  step="0.02"
                  value={quality}
                  onChange={(e) => setQuality(parseFloat(e.target.value))}
                />
              </div>
            )}

            {/* Resolution Preset */}
            <div className="export-group">
              <label className="group-label">Auflösung</label>
              <div className="format-pills">
                {[
                  { id: undefined, label: 'Original (100%)' },
                  { id: 2160, label: '4K Ultra HD' },
                  { id: 1080, label: '1080p Web' },
                ].map((res) => (
                  <button
                    key={String(res.id)}
                    className={`format-pill ${maxDimension === res.id ? 'active' : ''}`}
                    onClick={() => setMaxDimension(res.id)}
                  >
                    {res.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="export-action-buttons">
              <button
                className="primary-btn download-action-btn"
                onClick={handleDownload}
                disabled={isProcessing || !currentBlob}
              >
                <Download size={18} />
                <span>Foto Herunterladen</span>
              </button>

              <div className="secondary-action-row">
                {shareSupported && (
                  <button
                    className="secondary-btn"
                    onClick={handleShare}
                    disabled={isProcessing || !currentBlob}
                    title="Über Systemdialog teilen (Instagram, WhatsApp, etc.)"
                  >
                    <Share2 size={16} />
                    <span>Teilen</span>
                  </button>
                )}

                <button
                  className="secondary-btn"
                  onClick={handleCopy}
                  disabled={isProcessing || !currentBlob}
                  title="In die Zwischenablage kopieren"
                >
                  {copied ? <Check size={16} color="#39ff14" /> : <Copy size={16} />}
                  <span>{copied ? 'Kopiert!' : 'Kopieren'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
