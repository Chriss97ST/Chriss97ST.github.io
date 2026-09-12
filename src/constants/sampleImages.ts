export interface SampleImage {
  id: string;
  name: string;
  category: string;
  dataUrl: string;
}

// Generate rich procedural photographic sample images using Canvas
export function generateSampleImages(): SampleImage[] {
  return [
    {
      id: 'sunset-coast',
      name: 'Kalifornische Küste (Sonnenuntergang)',
      category: 'Landschaft',
      dataUrl: createSunsetSample(),
    },
    {
      id: 'city-street',
      name: 'Retro City & Neon',
      category: 'Street',
      dataUrl: createCityNeonSample(),
    },
    {
      id: 'vintage-car',
      name: 'Klassischer Oldtimer',
      category: 'Automobil',
      dataUrl: createVintageCarSample(),
    },
    {
      id: 'portrait-sun',
      name: 'Sommer Portrait',
      category: 'Porträt',
      dataUrl: createPortraitSample(),
    },
  ];
}

function createSunsetSample(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 900;
  canvas.height = 1200;
  const ctx = canvas.getContext('2d')!;

  // Sky Gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, 800);
  skyGrad.addColorStop(0, '#1a1836'); // Deep indigo
  skyGrad.addColorStop(0.3, '#5c2c58'); // Purple
  skyGrad.addColorStop(0.6, '#bf4342'); // Coral red
  skyGrad.addColorStop(0.85, '#e78b48'); // Warm orange
  skyGrad.addColorStop(1, '#f9dc5c'); // Bright sunset yellow
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, 900, 800);

  // Glowing Sun
  const sunGrad = ctx.createRadialGradient(450, 620, 10, 450, 620, 180);
  sunGrad.addColorStop(0, '#fffbf0');
  sunGrad.addColorStop(0.2, '#ffe373');
  sunGrad.addColorStop(0.6, 'rgba(247, 142, 60, 0.4)');
  sunGrad.addColorStop(1, 'rgba(235, 77, 75, 0)');
  ctx.fillStyle = sunGrad;
  ctx.beginPath();
  ctx.arc(450, 620, 180, 0, Math.PI * 2);
  ctx.fill();

  // Ocean Water
  const waterGrad = ctx.createLinearGradient(0, 750, 0, 1200);
  waterGrad.addColorStop(0, '#31233e');
  waterGrad.addColorStop(0.2, '#c86f3a');
  waterGrad.addColorStop(0.5, '#1e3046');
  waterGrad.addColorStop(1, '#0e1724');
  ctx.fillStyle = waterGrad;
  ctx.fillRect(0, 750, 900, 450);

  // Water reflections
  ctx.fillStyle = 'rgba(255, 230, 140, 0.25)';
  for (let y = 760; y < 1180; y += 8) {
    const w = 40 + (y - 750) * 0.9;
    const x = 450 - w / 2 + (Math.sin(y * 0.1) * 20);
    ctx.fillRect(x, y, w, 3);
  }

  // Palm tree silhouette
  ctx.fillStyle = '#0a0910';
  ctx.beginPath();
  ctx.moveTo(180, 1200);
  ctx.bezierCurveTo(190, 850, 130, 550, 240, 320);
  ctx.bezierCurveTo(246, 320, 204, 550, 210, 1200);
  ctx.fill();

  // Palm Fronds
  const drawFrond = (ox: number, oy: number, cpx: number, cpy: number, ex: number, ey: number) => {
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.quadraticCurveTo(cpx, cpy, ex, ey);
    ctx.strokeStyle = '#0a0910';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Leaf needles
    for (let t = 0.2; t <= 1; t += 0.08) {
      const px = (1 - t) * (1 - t) * ox + 2 * (1 - t) * t * cpx + t * t * ex;
      const py = (1 - t) * (1 - t) * oy + 2 * (1 - t) * t * cpy + t * t * ey;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + (Math.sin(t * 8) * 35), py + 30);
      ctx.lineWidth = 4;
      ctx.stroke();
    }
  };

  drawFrond(240, 320, 380, 260, 460, 340);
  drawFrond(240, 320, 120, 220, 40, 280);
  drawFrond(240, 320, 320, 180, 390, 220);
  drawFrond(240, 320, 180, 190, 140, 210);
  drawFrond(240, 320, 260, 140, 270, 180);
  drawFrond(240, 320, 90, 340, 30, 440);

  return canvas.toDataURL('image/jpeg', 0.92);
}

function createCityNeonSample(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 900;
  canvas.height = 1200;
  const ctx = canvas.getContext('2d')!;

  // Dark Night Sky
  const skyGrad = ctx.createLinearGradient(0, 0, 0, 1200);
  skyGrad.addColorStop(0, '#06060c');
  skyGrad.addColorStop(0.6, '#110d21');
  skyGrad.addColorStop(1, '#080811');
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, 900, 1200);

  // Distant Buildings
  const drawBuilding = (x: number, y: number, w: number, h: number, col: string) => {
    ctx.fillStyle = col;
    ctx.fillRect(x, y, w, h);

    // Glowing windows
    for (let r = y + 20; r < y + h - 20; r += 28) {
      for (let c = x + 15; c < x + w - 15; c += 22) {
        if ((r * 13 + c * 7) % 3 !== 0) {
          ctx.fillStyle = (r + c) % 5 === 0 ? '#ffea79' : '#88d8f7';
          ctx.fillRect(c, r, 12, 16);
        }
      }
    }
  };

  drawBuilding(50, 350, 220, 650, '#12121f');
  drawBuilding(300, 200, 280, 800, '#181729');
  drawBuilding(610, 310, 240, 690, '#131122');

  // Wet reflective street
  const streetGrad = ctx.createLinearGradient(0, 900, 0, 1200);
  streetGrad.addColorStop(0, '#1a1827');
  streetGrad.addColorStop(1, '#0c0a13');
  ctx.fillStyle = streetGrad;
  ctx.fillRect(0, 900, 900, 300);

  // Neon Sign
  ctx.font = '900 64px sans-serif';
  ctx.fillStyle = '#ff0077';
  ctx.shadowColor = '#ff0077';
  ctx.shadowBlur = 35;
  ctx.fillText('RETRO WAVE', 220, 320);

  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 30;
  ctx.fillText('HOTEL 1984', 270, 410);

  // Neon Reflections on wet asphalt
  ctx.fillStyle = 'rgba(255, 0, 119, 0.25)';
  ctx.fillRect(200, 940, 500, 80);
  ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
  ctx.fillRect(260, 1030, 380, 70);

  return canvas.toDataURL('image/jpeg', 0.92);
}

function createVintageCarSample(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 1200;
  canvas.height = 900;
  const ctx = canvas.getContext('2d')!;

  // Retro Desert Highway background
  const sky = ctx.createLinearGradient(0, 0, 0, 500);
  sky.addColorStop(0, '#3a7bd5');
  sky.addColorStop(0.6, '#93d5ed');
  sky.addColorStop(1, '#fbd786');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, 1200, 500);

  // Red Rock Mountains
  ctx.fillStyle = '#9e4732';
  ctx.beginPath();
  ctx.moveTo(0, 500);
  ctx.lineTo(200, 380);
  ctx.lineTo(420, 440);
  ctx.lineTo(680, 340);
  ctx.lineTo(950, 430);
  ctx.lineTo(1200, 360);
  ctx.lineTo(1200, 500);
  ctx.fill();

  // Desert Ground & Asphalt
  ctx.fillStyle = '#c79d67';
  ctx.fillRect(0, 500, 1200, 400);

  ctx.fillStyle = '#3a3835';
  ctx.beginPath();
  ctx.moveTo(400, 500);
  ctx.lineTo(800, 500);
  ctx.lineTo(1200, 900);
  ctx.lineTo(0, 900);
  ctx.fill();

  // Road Dashes
  ctx.fillStyle = '#ffd54f';
  for (let i = 0; i < 6; i++) {
    const y = 520 + i * 65;
    const h = 35 + i * 8;
    const w = 14 + i * 4;
    ctx.fillRect(600 - w / 2, y, w, h);
  }

  // Classic Turquoise 60s Muscle Car Silhouette / Shape
  ctx.fillStyle = '#20b2aa';
  ctx.beginPath();
  ctx.ellipse(600, 680, 320, 100, 0, 0, Math.PI * 2);
  ctx.fill();

  // Cabin
  ctx.fillStyle = '#178a84';
  ctx.beginPath();
  ctx.ellipse(600, 620, 190, 70, 0, Math.PI, Math.PI * 2);
  ctx.fill();

  // Chrome bumpers & lights
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(320, 670, 30, 20);
  ctx.fillRect(850, 670, 30, 20);

  // Wheels
  ctx.fillStyle = '#1a1a1a';
  ctx.beginPath();
  ctx.arc(430, 740, 55, 0, Math.PI * 2);
  ctx.arc(770, 740, 55, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#e0e0e0';
  ctx.beginPath();
  ctx.arc(430, 740, 28, 0, Math.PI * 2);
  ctx.arc(770, 740, 28, 0, Math.PI * 2);
  ctx.fill();

  return canvas.toDataURL('image/jpeg', 0.92);
}

function createPortraitSample(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 900;
  canvas.height = 1200;
  const ctx = canvas.getContext('2d')!;

  // Warm outdoor bokeh background
  const bg = ctx.createRadialGradient(450, 450, 50, 450, 600, 650);
  bg.addColorStop(0, '#f9ca24');
  bg.addColorStop(0.4, '#eb4d4b');
  bg.addColorStop(0.8, '#6ab04c');
  bg.addColorStop(1, '#130f40');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 900, 1200);

  // Bokeh Orbs
  const drawBokeh = (x: number, y: number, r: number, col: string) => {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  };

  drawBokeh(150, 200, 90, 'rgba(255, 234, 167, 0.35)');
  drawBokeh(750, 300, 110, 'rgba(255, 118, 117, 0.3)');
  drawBokeh(300, 150, 60, 'rgba(129, 236, 236, 0.25)');
  drawBokeh(800, 850, 130, 'rgba(253, 203, 110, 0.35)');
  drawBokeh(100, 900, 100, 'rgba(225, 112, 85, 0.3)');

  // Stylized Silhouette Portrait
  ctx.fillStyle = '#2d3436';

  // Shoulders
  ctx.beginPath();
  ctx.ellipse(450, 980, 280, 220, 0, 0, Math.PI * 2);
  ctx.fill();

  // Neck
  ctx.fillRect(400, 650, 100, 150);

  // Head
  ctx.beginPath();
  ctx.ellipse(450, 520, 160, 200, 0, 0, Math.PI * 2);
  ctx.fill();

  // Retro Sunglasses
  ctx.fillStyle = '#0984e3';
  ctx.beginPath();
  ctx.roundRect(330, 490, 105, 65, 16);
  ctx.roundRect(465, 490, 105, 65, 16);
  ctx.fill();

  // Sunglasses Bridge
  ctx.fillStyle = '#dfe6e9';
  ctx.fillRect(435, 510, 30, 8);

  // Lens Highlights
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.beginPath();
  ctx.moveTo(350, 500);
  ctx.lineTo(400, 500);
  ctx.lineTo(370, 545);
  ctx.lineTo(340, 545);
  ctx.fill();

  return canvas.toDataURL('image/jpeg', 0.92);
}
