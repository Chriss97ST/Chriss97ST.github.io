export type Decade = '1960s' | '1970s' | '1980s' | '1990s' | '2000s' | '2010s' | 'custom';

export type FrameStyle = 'none' | 'polaroid' | 'film35mm' | 'slide6x6' | 'vhs' | 'crt' | 'minimal-white';

export interface DateStampConfig {
  enabled: boolean;
  text: string; // e.g. " '94 10 14 " or custom
  format: 'auto-decade' | 'today' | 'custom';
  customDate: string;
  color: 'orange' | 'yellow' | 'green' | 'white' | 'red';
  fontStyle: 'lcd7' | 'dotmatrix' | 'digital';
  position: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  size: number; // 0.5 to 2.0
}

export interface RetroSettings {
  // Color & Tone
  brightness: number; // -1 to 1 (default 0)
  contrast: number; // 0 to 2 (default 1)
  saturation: number; // 0 to 2 (default 1)
  warmth: number; // -1 (cool/cyan) to 1 (warm/amber)
  tint: number; // -1 (green) to 1 (magenta)
  fade: number; // 0 to 1 (lift blacks / matte curve)
  sepia: number; // 0 to 1
  monochrome: number; // 0 to 1

  // Film Emulation
  filmGrain: number; // 0 to 1
  grainSize: number; // 0.5 to 3
  vignette: number; // 0 to 1
  vignetteRoundness: number; // 0 to 1

  // Dynamic Procedural Artifacts
  seed: number; // Dynamic seed for scratches, dust, light leak angle & texture
  dustScratches: number; // 0 to 1 (density & visibility of dust specks and scratches)
  lightLeak: number; // 0 to 1 (intensity of dynamic procedural light leak)
  lightLeakHue: number; // 0 to 360 (color temperature/tint of light leak)
  lightLeakPosition: number; // 0: Top-Right, 1: Top-Left, 2: Bottom-Right, 3: Bottom-Left, 4: Edge Streak

  // Digital / Analog Optics
  chromaticAberration: number; // 0 to 1 (RGB color fringe)
  bloomGlow: number; // 0 to 1 (soft highlight diffusion)
  scanlines: number; // 0 to 1 (CRT / VHS horizontal lines)
  pixelation: number; // 0 to 1 (Early digital/VGA pixel mosaic)
  dither: number; // 0 to 1 (Retro Bayer / retro color depth limit)
  lensBlur: number; // 0 to 1 (tilt-shift miniature/peripheral blur)

  // Overlays
  dateStamp: DateStampConfig;
  frame: FrameStyle;
  vhsOsd: boolean; // "PLAY", "SP", battery indicator
}

export interface RetroPreset {
  id: string;
  name: string;
  decade: Decade;
  subtitle: string;
  description: string;
  iconName: string;
  tags: string[];
  settings: RetroSettings;
}
