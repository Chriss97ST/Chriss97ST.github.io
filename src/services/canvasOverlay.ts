import type { RetroSettings } from '../types/retro';

export function renderRetroOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  settings: RetroSettings
): void {
  // 1. Render Frame if selected
  if (settings.frame !== 'none') {
    renderFrame(ctx, width, height, settings.frame);
  }

  // 2. Render VHS OSD if enabled
  if (settings.vhsOsd) {
    renderVhsOsd(ctx, width, height, settings);
  }

  // 3. Render Date Stamp if enabled
  if (settings.dateStamp && settings.dateStamp.enabled) {
    renderDateStamp(ctx, width, height, settings);
  }
}

function renderFrame(ctx: CanvasRenderingContext2D, w: number, h: number, frame: string): void {
  ctx.save();

  if (frame === 'polaroid') {
    // Polaroid classic style: equal white border top/sides, larger bottom chin
    const sideMargin = Math.round(w * 0.055);
    const topMargin = Math.round(h * 0.055);
    const bottomMargin = Math.round(h * 0.18);

    ctx.fillStyle = '#f6f4ea'; // Slightly warm vintage photo paper off-white

    // Top
    ctx.fillRect(0, 0, w, topMargin);
    // Left
    ctx.fillRect(0, 0, sideMargin, h);
    // Right
    ctx.fillRect(w - sideMargin, 0, sideMargin, h);
    // Bottom chin
    ctx.fillRect(0, h - bottomMargin, w, bottomMargin);

    // Subtle inner shadow on photo edge
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = Math.max(1, Math.round(w * 0.002));
    ctx.strokeRect(sideMargin, topMargin, w - sideMargin * 2, h - topMargin - bottomMargin);
  } else if (frame === 'film35mm') {
    // 35mm Sprocket Film Border (Top & Bottom black bands with perforation holes)
    const bandHeight = Math.round(h * 0.12);
    ctx.fillStyle = '#0a0a0c';

    // Top band
    ctx.fillRect(0, 0, w, bandHeight);
    // Bottom band
    ctx.fillRect(0, h - bandHeight, w, bandHeight);

    // Film perforations (rounded rectangles)
    const holeWidth = Math.round(w * 0.032);
    const holeHeight = Math.round(bandHeight * 0.55);
    const holeRadius = Math.round(holeWidth * 0.25);
    const spacing = Math.round(w * 0.058);

    ctx.fillStyle = '#ffffff'; // Through-hole look

    const numHoles = Math.floor(w / spacing);
    const startX = (w - (numHoles - 1) * spacing) / 2;

    for (let i = 0; i < numHoles; i++) {
      const x = startX + i * spacing - holeWidth / 2;
      const topY = (bandHeight - holeHeight) / 2;
      const botY = h - bandHeight + (bandHeight - holeHeight) / 2;

      // Top hole
      drawRoundedRect(ctx, x, topY, holeWidth, holeHeight, holeRadius);
      ctx.fill();

      // Bottom hole
      drawRoundedRect(ctx, x, botY, holeWidth, holeHeight, holeRadius);
      ctx.fill();
    }

    // 35mm edge text
    ctx.fillStyle = 'rgba(230, 160, 40, 0.75)';
    ctx.font = `bold ${Math.max(10, Math.round(bandHeight * 0.22))}px monospace`;
    ctx.fillText('▶ KODAK SAFETY FILM 400', Math.round(w * 0.08), Math.round(bandHeight * 0.9));
    ctx.fillText('▶ 24A', Math.round(w * 0.65), Math.round(bandHeight * 0.9));
    ctx.fillText('ISO 400/27°', Math.round(w * 0.08), h - Math.round(bandHeight * 0.15));
    ctx.fillText('EXP 24', Math.round(w * 0.65), h - Math.round(bandHeight * 0.15));
  } else if (frame === 'slide6x6') {
    // 6x6 Medium format slide mount
    const border = Math.round(Math.min(w, h) * 0.06);
    ctx.fillStyle = '#1c1b18';
    ctx.fillRect(0, 0, w, border);
    ctx.fillRect(0, 0, border, h);
    ctx.fillRect(w - border, 0, border, h);
    ctx.fillRect(0, h - border, w, border);

    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 2;
    ctx.strokeRect(border, border, w - border * 2, h - border * 2);
  } else if (frame === 'crt') {
    // CRT Curved Screen Mask & Bezel
    const bezel = Math.round(Math.min(w, h) * 0.04);
    ctx.fillStyle = '#08080a';
    ctx.fillRect(0, 0, w, bezel);
    ctx.fillRect(0, 0, bezel, h);
    ctx.fillRect(w - bezel, 0, bezel, h);
    ctx.fillRect(0, h - bezel, w, bezel);

    // Curved inner corners
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 4;
    drawRoundedRect(ctx, bezel, bezel, w - bezel * 2, h - bezel * 2, bezel * 2);
    ctx.stroke();
  } else if (frame === 'minimal-white') {
    const border = Math.round(Math.min(w, h) * 0.035);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, border);
    ctx.fillRect(0, 0, border, h);
    ctx.fillRect(w - border, 0, border, h);
    ctx.fillRect(0, h - border, w, border);
  }

  ctx.restore();
}

function renderVhsOsd(ctx: CanvasRenderingContext2D, w: number, h: number, settings: RetroSettings): void {
  ctx.save();
  const fontSize = Math.max(14, Math.round(w * 0.038));
  ctx.font = `900 ${fontSize}px "Courier New", Courier, monospace`;
  ctx.fillStyle = '#39ff14'; // Retro green phosphorescent or white
  ctx.shadowColor = 'rgba(57, 255, 20, 0.8)';
  ctx.shadowBlur = 8;

  const paddingX = Math.round(w * 0.06);
  const paddingY = Math.round(h * 0.08);

  // Top Left: PLAY & SP
  ctx.fillText('PLAY ▶', paddingX, paddingY);
  ctx.font = `bold ${Math.round(fontSize * 0.75)}px monospace`;
  ctx.fillText('SP', paddingX, paddingY + fontSize * 1.2);

  // Top Right: REC ● or Battery icon
  ctx.fillStyle = '#ff2a2a';
  ctx.shadowColor = 'rgba(255, 42, 42, 0.8)';
  ctx.fillText('● REC', w - paddingX - fontSize * 3.5, paddingY);

  // Bottom Left: Timecode
  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
  ctx.font = `bold ${fontSize}px monospace`;
  const timecode = `0:${String(Math.floor(settings.seed % 60)).padStart(2, '0')}:34`;
  ctx.fillText(timecode, paddingX, h - paddingY);

  ctx.restore();
}

function renderDateStamp(ctx: CanvasRenderingContext2D, w: number, h: number, settings: RetroSettings): void {
  const ds = settings.dateStamp;
  let text = ds.text;

  if (!text || ds.format === 'today') {
    const d = new Date();
    const yy = String(d.getFullYear()).slice(2);
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    text = `'${yy} ${mm} ${dd}`;
  }

  const baseSize = Math.max(16, Math.round(w * 0.042 * (ds.size || 1.0)));

  ctx.save();

  // Color mapping
  let colorHex = '#ff7b00'; // Classic orange LCD
  let glowHex = 'rgba(255, 123, 0, 0.75)';

  if (ds.color === 'yellow') {
    colorHex = '#ffea00';
    glowHex = 'rgba(255, 234, 0, 0.75)';
  } else if (ds.color === 'green') {
    colorHex = '#39ff14';
    glowHex = 'rgba(57, 255, 20, 0.75)';
  } else if (ds.color === 'red') {
    colorHex = '#ff2b2b';
    glowHex = 'rgba(255, 43, 43, 0.75)';
  } else if (ds.color === 'white') {
    colorHex = '#ffffff';
    glowHex = 'rgba(255, 255, 255, 0.75)';
  }

  ctx.fillStyle = colorHex;
  ctx.shadowColor = glowHex;
  ctx.shadowBlur = Math.round(baseSize * 0.4);

  if (ds.fontStyle === 'lcd7' || ds.fontStyle === 'digital') {
    ctx.font = `bold ${baseSize}px "Courier New", monospace`;
  } else {
    ctx.font = `900 ${baseSize}px monospace`;
  }

  const textMetrics = ctx.measureText(text);
  const textWidth = textMetrics.width;

  const marginX = Math.round(w * 0.06);
  const marginY = Math.round(h * 0.07);

  let x = w - textWidth - marginX;
  let y = h - marginY;

  if (ds.position === 'bottom-left') {
    x = marginX;
    y = h - marginY;
  } else if (ds.position === 'top-right') {
    x = w - textWidth - marginX;
    y = marginY + baseSize;
  } else if (ds.position === 'top-left') {
    x = marginX;
    y = marginY + baseSize;
  }

  // Draw date stamp
  ctx.fillText(text, x, y);

  ctx.restore();
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}
