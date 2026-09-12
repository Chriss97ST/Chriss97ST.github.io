import React from 'react';
import { Camera, Image as ImageIcon, Sparkles, Volume2, VolumeX, Download, RotateCcw, DownloadCloud } from 'lucide-react';

interface HeaderProps {
  mode: 'studio' | 'camera';
  onModeChange: (mode: 'studio' | 'camera') => void;
  audioEnabled: boolean;
  onToggleAudio: () => void;
  onRandomizeSeed: () => void;
  onUndoSeed?: () => void;
  canUndoSeed?: boolean;
  onResetSettings: () => void;
  onOpenExport: () => void;
  canExport: boolean;
  canInstallPWA: boolean;
  onInstallPWA: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  mode,
  onModeChange,
  audioEnabled,
  onToggleAudio,
  onRandomizeSeed,
  onUndoSeed,
  canUndoSeed,
  onResetSettings,
  onOpenExport,
  canExport,
  canInstallPWA,
  onInstallPWA,
}) => {
  return (
    <header className="header-container">
      <div className="header-left">
        <div className="logo-badge">
          <span className="logo-lens">📷</span>
          <div>
            <h1 className="logo-title">RetroLens Studio</h1>
            <span className="logo-tagline">Analog & Vintage Photo Engine (1960–2010)</span>
          </div>
        </div>
      </div>

      <div className="header-center">
        <div className="mode-toggle-pill">
          <button
            className={`pill-btn ${mode === 'studio' ? 'active' : ''}`}
            onClick={() => onModeChange('studio')}
            title="Studio-Editor / Foto hochladen"
          >
            <ImageIcon size={16} />
            <span>Studio / Upload</span>
          </button>
          <button
            className={`pill-btn ${mode === 'camera' ? 'active' : ''}`}
            onClick={() => onModeChange('camera')}
            title="Live-Kamera mit Filter-Vorschau"
          >
            <Camera size={16} />
            <span>Live Retro-Kamera</span>
          </button>
        </div>
      </div>

      <div className="header-right">
        {canInstallPWA && (
          <button className="action-btn pwa-install-header-btn" onClick={onInstallPWA} title="Als App installieren">
            <DownloadCloud size={17} />
            <span className="btn-text">App Installieren</span>
          </button>
        )}

        <button
          className="action-btn"
          onClick={onRandomizeSeed}
          title="Neues Unikat generieren (Zufällige Lichteinfälle, Staub & Kratzer)"
        >
          <Sparkles size={17} className="sparkle-icon" />
          <span className="btn-text">Unikat (Seed)</span>
        </button>

        {canUndoSeed && onUndoSeed && (
          <button
            className="icon-btn"
            onClick={onUndoSeed}
            title="Vorherigen Seed wiederherstellen (Rückgängig)"
          >
            <RotateCcw size={18} />
          </button>
        )}

        <button
          className="icon-btn"
          onClick={onToggleAudio}
          title={audioEnabled ? 'Soundeffekte stummschalten' : 'Soundeffekte aktivieren'}
        >
          {audioEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
        </button>

        <button
          className="icon-btn"
          onClick={onResetSettings}
          title="Einstellungen auf Standard zurücksetzen"
        >
          <RotateCcw size={18} />
        </button>

        <button
          className={`action-btn primary-btn ${!canExport ? 'disabled' : ''}`}
          onClick={onOpenExport}
          disabled={!canExport}
          title="Entwickeltes Foto exportieren & teilen"
        >
          <Download size={17} />
          <span>Foto Speichern</span>
        </button>
      </div>
    </header>
  );
};
