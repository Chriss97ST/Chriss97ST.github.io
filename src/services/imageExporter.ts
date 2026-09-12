import type { RetroSettings } from '../types/retro';
import { renderRetroOverlay } from './canvasOverlay';
import { WebGLRetroRenderer } from './webglRenderer';

export interface ExportOptions {
  format: 'image/jpeg' | 'image/png' | 'image/webp';
  quality: number; // 0.1 to 1.0
  maxDimension?: number; // e.g. 1080, 2160, or undefined for full original
  filename?: string;
}

export async function processHighResImage(
  sourceImage: HTMLImageElement | HTMLCanvasElement,
  settings: RetroSettings,
  options: ExportOptions
): Promise<{ blob: Blob; dataUrl: string; width: number; height: number }> {
  // Determine target dimensions
  let srcW = sourceImage instanceof HTMLImageElement ? sourceImage.naturalWidth : sourceImage.width;
  let srcH = sourceImage instanceof HTMLImageElement ? sourceImage.naturalHeight : sourceImage.height;

  if (srcW === 0 || srcH === 0) {
    srcW = 1200;
    srcH = 1200;
  }

  let targetW = srcW;
  let targetH = srcH;

  if (options.maxDimension && Math.max(srcW, srcH) > options.maxDimension) {
    const scale = options.maxDimension / Math.max(srcW, srcH);
    targetW = Math.round(srcW * scale);
    targetH = Math.round(srcH * scale);
  }

  // 1. Render WebGL Base Layer
  const glCanvas = document.createElement('canvas');
  glCanvas.width = targetW;
  glCanvas.height = targetH;

  const renderer = new WebGLRetroRenderer(glCanvas);
  renderer.render(sourceImage, settings, 0);

  // 2. Composite with 2D Overlays on 2D Output Canvas
  const outCanvas = document.createElement('canvas');
  outCanvas.width = targetW;
  outCanvas.height = targetH;
  const ctx = outCanvas.getContext('2d')!;

  // Draw WebGL filtered image
  ctx.drawImage(glCanvas, 0, 0, targetW, targetH);

  // Draw 2D overlays (Date Stamp, Polaroid / 35mm borders, OSD)
  renderRetroOverlay(ctx, targetW, targetH, settings);

  // Clean up WebGL
  renderer.destroy();

  // 3. Convert to Blob & DataURL
  return new Promise((resolve, reject) => {
    outCanvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Failed to create image blob'));
          return;
        }
        const dataUrl = outCanvas.toDataURL(options.format, options.quality);
        resolve({
          blob,
          dataUrl,
          width: targetW,
          height: targetH,
        });
      },
      options.format,
      options.quality
    );
  });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function shareImage(blob: Blob, filename: string, title: string = 'RetroLens Photo'): Promise<boolean> {
  if (navigator.share && navigator.canShare) {
    const file = new File([blob], filename, { type: blob.type });
    if (navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({
          title,
          text: 'Entwickelt mit RetroLens Studio PWA 📷',
          files: [file],
        });
        return true;
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Share failed:', err);
        }
      }
    }
  }
  return false;
}

export async function copyImageToClipboard(blob: Blob): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.write) {
      // PNG is required for ClipboardItem in most browsers
      let pngBlob = blob;
      if (blob.type !== 'image/png') {
        const img = new Image();
        const url = URL.createObjectURL(blob);
        await new Promise((res) => {
          img.onload = res;
          img.src = url;
        });
        const c = document.createElement('canvas');
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        pngBlob = await new Promise<Blob>((res) => c.toBlob((b) => res(b!), 'image/png'));
      }
      const item = new ClipboardItem({ 'image/png': pngBlob });
      await navigator.clipboard.write([item]);
      return true;
    }
  } catch (err) {
    console.error('Clipboard copy failed:', err);
  }
  return false;
}
