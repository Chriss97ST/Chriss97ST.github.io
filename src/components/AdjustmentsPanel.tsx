import React, { useState } from 'react';
import { Sparkles, SunMedium, Frame, Shuffle } from 'lucide-react';
import type { RetroSettings, FrameStyle } from '../types/retro';

interface AdjustmentsPanelProps {
  settings: RetroSettings;
  onChange: (updated: RetroSettings) => void;
  onRandomizeSeed: () => void;
}

type TabType = 'effects' | 'color' | 'overlay';

export const AdjustmentsPanel: React.FC<AdjustmentsPanelProps> = ({
  settings,
  onChange,
  onRandomizeSeed,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('effects');

  const updateSetting = <K extends keyof RetroSettings>(key: K, value: RetroSettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  const updateDateStamp = <K extends keyof RetroSettings['dateStamp']>(
    key: K,
    value: RetroSettings['dateStamp'][K]
  ) => {
    onChange({
      ...settings,
      dateStamp: {
        ...settings.dateStamp,
        [key]: value,
      },
    });
  };

  return (
    <div className="adjustments-panel">
      {/* Tab Navigation */}
      <div className="adj-tabs">
        <button
          className={`adj-tab-btn ${activeTab === 'effects' ? 'active' : ''}`}
          onClick={() => setActiveTab('effects')}
        >
          <Sparkles size={16} />
          <span>Retro-Effekte</span>
        </button>

        <button
          className={`adj-tab-btn ${activeTab === 'color' ? 'active' : ''}`}
          onClick={() => setActiveTab('color')}
        >
          <SunMedium size={16} />
          <span>Farbe & Licht</span>
        </button>

        <button
          className={`adj-tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
          onClick={() => setActiveTab('overlay')}
        >
          <Frame size={16} />
          <span>Stempel & Rahmen</span>
        </button>
      </div>

      <div className="adj-content">
        {/* --- TAB 1: RETRO EFFECTS --- */}
        {activeTab === 'effects' && (
          <div className="control-group-list">
            <div className="seed-banner">
              <div className="seed-info">
                <span className="seed-title">Dynamischer Unikat-Seed #{settings.seed}</span>
                <span className="seed-desc">Erzeugt zufällige Lichteinfälle, Kratzer & Kornmuster</span>
              </div>
              <button className="seed-btn" onClick={onRandomizeSeed} title="Neu auswürfeln">
                <Shuffle size={15} />
                <span>Neu würfeln</span>
              </button>
            </div>

            {/* Film Grain */}
            <div className="control-card">
              <div className="slider-header">
                <label>Filmkorn (Analoges Rauschen)</label>
                <span className="val-badge">{Math.round(settings.filmGrain * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.filmGrain}
                onChange={(e) => updateSetting('filmGrain', parseFloat(e.target.value))}
              />

              {settings.filmGrain > 0.05 && (
                <div className="sub-slider">
                  <div className="slider-header">
                    <label>Korngröße</label>
                    <span className="val-badge">{settings.grainSize.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="3.0"
                    step="0.1"
                    value={settings.grainSize}
                    onChange={(e) => updateSetting('grainSize', parseFloat(e.target.value))}
                  />
                </div>
              )}
            </div>

            {/* Light Leak */}
            <div className="control-card">
              <div className="slider-header">
                <label>Lichteinfall (Light Leak)</label>
                <span className="val-badge">{Math.round(settings.lightLeak * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.lightLeak}
                onChange={(e) => updateSetting('lightLeak', parseFloat(e.target.value))}
              />

              {settings.lightLeak > 0.05 && (
                <div className="sub-options">
                  <div className="slider-header">
                    <label>Farbton des Lichteinfalls</label>
                    <span className="val-badge">{Math.round(settings.lightLeakHue)}°</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="360"
                    step="1"
                    value={settings.lightLeakHue}
                    onChange={(e) => updateSetting('lightLeakHue', parseFloat(e.target.value))}
                  />

                  <div className="leak-positions">
                    <span className="sub-label">Position:</span>
                    {[
                      { id: 0, label: 'Oben Rechts' },
                      { id: 1, label: 'Oben Links' },
                      { id: 2, label: 'Unten Rechts' },
                      { id: 3, label: 'Unten Links' },
                      { id: 4, label: 'Mitte Oben' },
                    ].map((pos) => (
                      <button
                        key={pos.id}
                        className={`mini-pill ${settings.lightLeakPosition === pos.id ? 'active' : ''}`}
                        onClick={() => updateSetting('lightLeakPosition', pos.id)}
                      >
                        {pos.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Dust & Scratches */}
            <div className="control-card">
              <div className="slider-header">
                <label>Staub & Filmkratzer</label>
                <span className="val-badge">{Math.round(settings.dustScratches * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.dustScratches}
                onChange={(e) => updateSetting('dustScratches', parseFloat(e.target.value))}
              />
            </div>

            {/* Chromatic Aberration */}
            <div className="control-card">
              <div className="slider-header">
                <label>Chromatische Aberration (RGB-Split)</label>
                <span className="val-badge">{Math.round(settings.chromaticAberration * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.chromaticAberration}
                onChange={(e) => updateSetting('chromaticAberration', parseFloat(e.target.value))}
              />
            </div>

            {/* Bloom & Highlight Glow */}
            <div className="control-card">
              <div className="slider-header">
                <label>Lichterglühen / Diffusions-Glow</label>
                <span className="val-badge">{Math.round(settings.bloomGlow * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.bloomGlow}
                onChange={(e) => updateSetting('bloomGlow', parseFloat(e.target.value))}
              />
            </div>

            {/* Vignette */}
            <div className="control-card">
              <div className="slider-header">
                <label>Vignettierung (Randabdunklung)</label>
                <span className="val-badge">{Math.round(settings.vignette * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.vignette}
                onChange={(e) => updateSetting('vignette', parseFloat(e.target.value))}
              />
            </div>

            {/* VHS Scanlines */}
            <div className="control-card">
              <div className="slider-header">
                <label>VHS / CRT Scanlines</label>
                <span className="val-badge">{Math.round(settings.scanlines * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.scanlines}
                onChange={(e) => updateSetting('scanlines', parseFloat(e.target.value))}
              />
            </div>

            {/* Pixelation & Dither (Y2K) */}
            <div className="control-card">
              <div className="slider-header">
                <label>Y2K Handy-Pixelation / Dithering</label>
                <span className="val-badge">{Math.round(settings.pixelation * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.pixelation}
                onChange={(e) => updateSetting('pixelation', parseFloat(e.target.value))}
              />

              {settings.pixelation > 0.05 && (
                <div className="sub-slider">
                  <div className="slider-header">
                    <label>Bayer Farbtiefe (Dither)</label>
                    <span className="val-badge">{Math.round(settings.dither * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={settings.dither}
                    onChange={(e) => updateSetting('dither', parseFloat(e.target.value))}
                  />
                </div>
              )}
            </div>

            {/* Lens Blur / Tilt Shift */}
            <div className="control-card">
              <div className="slider-header">
                <label>Rand-Unschärfe (Tilt-Shift)</label>
                <span className="val-badge">{Math.round(settings.lensBlur * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.lensBlur}
                onChange={(e) => updateSetting('lensBlur', parseFloat(e.target.value))}
              />
            </div>
          </div>
        )}

        {/* --- TAB 2: COLOR & TONE --- */}
        {activeTab === 'color' && (
          <div className="control-group-list">
            {/* Brightness */}
            <div className="control-card">
              <div className="slider-header">
                <label>Helligkeit</label>
                <span className="val-badge">{(settings.brightness * 100).toFixed(0)}</span>
              </div>
              <input
                type="range"
                min="-0.5"
                max="0.5"
                step="0.01"
                value={settings.brightness}
                onChange={(e) => updateSetting('brightness', parseFloat(e.target.value))}
              />
            </div>

            {/* Contrast */}
            <div className="control-card">
              <div className="slider-header">
                <label>Kontrast</label>
                <span className="val-badge">{(settings.contrast * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1.8"
                step="0.01"
                value={settings.contrast}
                onChange={(e) => updateSetting('contrast', parseFloat(e.target.value))}
              />
            </div>

            {/* Saturation */}
            <div className="control-card">
              <div className="slider-header">
                <label>Farbsättigung</label>
                <span className="val-badge">{(settings.saturation * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="2.0"
                step="0.01"
                value={settings.saturation}
                onChange={(e) => updateSetting('saturation', parseFloat(e.target.value))}
              />
            </div>

            {/* Warmth (Color Temperature) */}
            <div className="control-card">
              <div className="slider-header">
                <label>Farbtemperatur (Kalt ↔ Warm)</label>
                <span className="val-badge">{(settings.warmth * 100).toFixed(0)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.02"
                value={settings.warmth}
                onChange={(e) => updateSetting('warmth', parseFloat(e.target.value))}
              />
            </div>

            {/* Tint (Green ↔ Magenta) */}
            <div className="control-card">
              <div className="slider-header">
                <label>Farbton (Grün ↔ Magenta)</label>
                <span className="val-badge">{(settings.tint * 100).toFixed(0)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.02"
                value={settings.tint}
                onChange={(e) => updateSetting('tint', parseFloat(e.target.value))}
              />
            </div>

            {/* Faded Matte Blacks */}
            <div className="control-card">
              <div className="slider-header">
                <label>Matte Tiefen (Faded Film)</label>
                <span className="val-badge">{Math.round(settings.fade * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.fade}
                onChange={(e) => updateSetting('fade', parseFloat(e.target.value))}
              />
            </div>

            {/* Sepia */}
            <div className="control-card">
              <div className="slider-header">
                <label>Sepia-Tönung</label>
                <span className="val-badge">{Math.round(settings.sepia * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.sepia}
                onChange={(e) => updateSetting('sepia', parseFloat(e.target.value))}
              />
            </div>

            {/* Monochrome */}
            <div className="control-card">
              <div className="slider-header">
                <label>Schwarz-Weiß Modus</label>
                <span className="val-badge">{Math.round(settings.monochrome * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={settings.monochrome}
                onChange={(e) => updateSetting('monochrome', parseFloat(e.target.value))}
              />
            </div>
          </div>
        )}

        {/* --- TAB 3: OVERLAYS, DATE STAMP & FRAMES --- */}
        {activeTab === 'overlay' && (
          <div className="control-group-list">
            {/* Frame Selector */}
            <div className="control-card">
              <label className="section-label">Analoger Rahmen / Format</label>
              <div className="frame-grid">
                {[
                  { id: 'none', label: 'Kein Rahmen' },
                  { id: 'polaroid', label: 'SX-70 Polaroid' },
                  { id: 'film35mm', label: '35mm Filmstreifen' },
                  { id: 'slide6x6', label: '6x6 Diarahmen' },
                  { id: 'crt', label: 'CRT Röhren-TV' },
                  { id: 'minimal-white', label: 'Schlichter Rand' },
                ].map((f) => (
                  <button
                    key={f.id}
                    className={`frame-opt-btn ${settings.frame === f.id ? 'active' : ''}`}
                    onClick={() => updateSetting('frame', f.id as FrameStyle)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {/* VHS OSD Toggle */}
            <div className="control-card">
              <div className="toggle-row">
                <div>
                  <label className="toggle-title">VHS Camcorder OSD Overlay</label>
                  <span className="toggle-desc">Zeigt PLAY ▶, REC ● und Timecode an</span>
                </div>
                <input
                  type="checkbox"
                  className="toggle-checkbox"
                  checked={settings.vhsOsd}
                  onChange={(e) => updateSetting('vhsOsd', e.target.checked)}
                />
              </div>
            </div>

            {/* Date Stamp Controls */}
            <div className="control-card">
              <div className="toggle-row">
                <div>
                  <label className="toggle-title">Retro Datumsstempel</label>
                  <span className="toggle-desc">Klassischer LED/LCD Datumsdruck wie in den 90ern</span>
                </div>
                <input
                  type="checkbox"
                  className="toggle-checkbox"
                  checked={settings.dateStamp.enabled}
                  onChange={(e) => updateDateStamp('enabled', e.target.checked)}
                />
              </div>

              {settings.dateStamp.enabled && (
                <div className="datestamp-settings">
                  <div className="form-group">
                    <label>Stempel-Text</label>
                    <input
                      type="text"
                      className="text-input"
                      value={settings.dateStamp.text}
                      placeholder="z.B. '96 08 24 oder frei wählbar"
                      onChange={(e) => updateDateStamp('text', e.target.value)}
                    />
                  </div>

                  <div className="color-selector">
                    <label className="sub-label">Farbe des Datums:</label>
                    <div className="color-pills">
                      {[
                        { id: 'orange', label: 'Orange (Klassisch)', color: '#ff7b00' },
                        { id: 'yellow', label: 'Gelb', color: '#ffea00' },
                        { id: 'green', label: 'Grün', color: '#39ff14' },
                        { id: 'red', label: 'Rot', color: '#ff2b2b' },
                        { id: 'white', label: 'Weiß', color: '#ffffff' },
                      ].map((c) => (
                        <button
                          key={c.id}
                          className={`color-pill ${settings.dateStamp.color === c.id ? 'active' : ''}`}
                          style={{ borderColor: c.color }}
                          onClick={() => updateDateStamp('color', c.id as any)}
                        >
                          <span className="dot" style={{ backgroundColor: c.color }} />
                          <span>{c.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="position-selector">
                    <label className="sub-label">Position:</label>
                    <div className="mini-pills">
                      {[
                        { id: 'bottom-right', label: 'Unten Rechts' },
                        { id: 'bottom-left', label: 'Unten Links' },
                        { id: 'top-right', label: 'Oben Rechts' },
                        { id: 'top-left', label: 'Oben Links' },
                      ].map((p) => (
                        <button
                          key={p.id}
                          className={`mini-pill ${settings.dateStamp.position === p.id ? 'active' : ''}`}
                          onClick={() => updateDateStamp('position', p.id as any)}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="slider-header" style={{ marginTop: '12px' }}>
                    <label>Schriftgröße</label>
                    <span className="val-badge">{settings.dateStamp.size.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.6"
                    max="1.8"
                    step="0.1"
                    value={settings.dateStamp.size}
                    onChange={(e) => updateDateStamp('size', parseFloat(e.target.value))}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
