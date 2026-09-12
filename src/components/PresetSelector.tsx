import React, { useState } from 'react';
import { Camera, Film, Image as ImageIcon, Sun, Tv, Zap, Sparkles, Flame, Laptop, Smartphone, Layers, SunMedium, Check } from 'lucide-react';
import { RETRO_PRESETS } from '../constants/presets';
import type { Decade, RetroPreset } from '../types/retro';

interface PresetSelectorProps {
  selectedPresetId: string | null;
  onSelectPreset: (preset: RetroPreset) => void;
}

const DECADES: { id: Decade | 'all'; label: string; yearRange: string }[] = [
  { id: 'all', label: 'Alle Epochen', yearRange: '1960 - 2010' },
  { id: '1960s', label: '1960er', yearRange: 'Kodachrome & SW' },
  { id: '1970s', label: '1970er', yearRange: 'Polaroid & Disco' },
  { id: '1980s', label: '1980er', yearRange: 'VHS & Neon Wave' },
  { id: '1990s', label: '1990er', yearRange: 'Einwegkamera 35mm' },
  { id: '2000s', label: '2000er', yearRange: 'Y2K Digicam & VGA' },
  { id: '2010s', label: '2010er', yearRange: 'Early Instagram' },
];

const ICON_MAP: Record<string, React.ReactNode> = {
  Camera: <Camera size={18} />,
  Film: <Film size={18} />,
  Image: <ImageIcon size={18} />,
  Sun: <Sun size={18} />,
  Tv: <Tv size={18} />,
  Zap: <Zap size={18} />,
  Sparkles: <Sparkles size={18} />,
  Flame: <Flame size={18} />,
  Laptop: <Laptop size={18} />,
  Smartphone: <Smartphone size={18} />,
  Layers: <Layers size={18} />,
  SunMedium: <SunMedium size={18} />,
};

export const PresetSelector: React.FC<PresetSelectorProps> = ({
  selectedPresetId,
  onSelectPreset,
}) => {
  const [activeDecade, setActiveDecade] = useState<Decade | 'all'>('all');

  const filteredPresets = activeDecade === 'all'
    ? RETRO_PRESETS
    : RETRO_PRESETS.filter((p) => p.decade === activeDecade);

  return (
    <div className="preset-selector-container">
      {/* Decade Filter Tabs */}
      <div className="decade-tabs">
        {DECADES.map((d) => (
          <button
            key={d.id}
            className={`decade-tab-btn ${activeDecade === d.id ? 'active' : ''}`}
            onClick={() => setActiveDecade(d.id)}
          >
            <span className="decade-label">{d.label}</span>
            <span className="decade-sub">{d.yearRange}</span>
          </button>
        ))}
      </div>

      {/* Preset Cards Grid */}
      <div className="preset-grid">
        {filteredPresets.map((preset) => {
          const isSelected = selectedPresetId === preset.id;
          return (
            <div
              key={preset.id}
              className={`preset-card ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectPreset(preset)}
            >
              <div className="preset-card-header">
                <div className="preset-icon-badge">
                  {ICON_MAP[preset.iconName] || <Sparkles size={18} />}
                </div>
                <span className="preset-decade-badge">{preset.decade}</span>
                {isSelected && (
                  <div className="preset-active-check">
                    <Check size={14} />
                  </div>
                )}
              </div>

              <div className="preset-card-body">
                <h4 className="preset-title">{preset.name}</h4>
                <div className="preset-subtitle">{preset.subtitle}</div>
                <p className="preset-desc">{preset.description}</p>
              </div>

              <div className="preset-tags">
                {preset.tags.map((tag) => (
                  <span key={tag} className="tag-pill">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
