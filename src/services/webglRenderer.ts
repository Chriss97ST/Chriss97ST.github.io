import type { RetroSettings } from '../types/retro';

const VERTEX_SHADER_SOURCE = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`;

const FRAGMENT_SHADER_SOURCE = `
precision mediump float;

varying vec2 v_texCoord;
uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform float u_time;

// Color & Tone
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_warmth;
uniform float u_tint;
uniform float u_fade;
uniform float u_sepia;
uniform float u_monochrome;

// Effects
uniform float u_filmGrain;
uniform float u_grainSize;
uniform float u_vignette;
uniform float u_vignetteRoundness;
uniform float u_seed;
uniform float u_dustScratches;
uniform float u_lightLeak;
uniform float u_lightLeakHue;
uniform int u_lightLeakPosition; // 0..4
uniform float u_chromaticAberration;
uniform float u_bloomGlow;
uniform float u_scanlines;
uniform float u_pixelation;
uniform float u_dither;
uniform float u_lensBlur;

// Helper: Pseudo-random hash
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Helper: 2D Noise
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Helper: Convert HSL to RGB
vec3 hsl2rgb(vec3 c) {
  vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

void main() {
  vec2 uv = v_texCoord;

  // 1. Pixelation (early digicam / flip phone)
  if (u_pixelation > 0.01) {
    float pixels = mix(u_resolution.x, 90.0, u_pixelation);
    float dx = 1.0 / pixels;
    float dy = (u_resolution.y / u_resolution.x) * dx;
    uv = vec2(dx * floor(uv.x / dx), dy * floor(uv.y / dy));
  }

  // 2. Chromatic Aberration (RGB split away from center)
  vec2 center = vec2(0.5, 0.5);
  vec2 distVec = uv - center;
  float distFromCenter = length(distVec);
  
  float caOffset = u_chromaticAberration * 0.02 * (distFromCenter * 1.5 + 0.2);
  vec2 rUv = uv - distVec * caOffset;
  vec2 bUv = uv + distVec * caOffset;

  // 3. Lens Blur / Tilt-shift sampling
  vec4 color;
  if (u_lensBlur > 0.05) {
    float blurAmount = smoothstep(0.25, 0.75, distFromCenter) * u_lensBlur * 0.012;
    vec4 sum = vec4(0.0);
    sum += texture2D(u_image, vec2(rUv.x, uv.y)) * 0.2;
    sum += texture2D(u_image, vec2(uv.x + blurAmount, uv.y)) * 0.15;
    sum += texture2D(u_image, vec2(uv.x - blurAmount, uv.y)) * 0.15;
    sum += texture2D(u_image, vec2(uv.x, uv.y + blurAmount)) * 0.15;
    sum += texture2D(u_image, vec2(uv.x, uv.y - blurAmount)) * 0.15;
    sum += texture2D(u_image, vec2(bUv.x, uv.y)) * 0.2;
    color = sum;
  } else {
    float r = texture2D(u_image, rUv).r;
    float g = texture2D(u_image, uv).g;
    float b = texture2D(u_image, bUv).b;
    float a = texture2D(u_image, uv).a;
    color = vec4(r, g, b, a);
  }

  // 4. Bloom / Glow Simulation
  if (u_bloomGlow > 0.05) {
    float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    if (luminance > 0.55) {
      float glowBoost = (luminance - 0.55) * u_bloomGlow * 0.8;
      color.rgb += vec3(glowBoost);
    }
  }

  // 5. Brightness & Contrast
  color.rgb += vec3(u_brightness);
  color.rgb = (color.rgb - 0.5) * u_contrast + 0.5;

  // 6. Saturation
  float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  color.rgb = mix(vec3(gray), color.rgb, u_saturation);

  // 7. Warmth & Tint Color Temperature
  // Warmth adds Red, slightly reduces Blue
  color.r += u_warmth * 0.15;
  color.g += u_warmth * 0.05;
  color.b -= u_warmth * 0.15;

  // Tint adds Magenta (Red+Blue) or Green
  color.r += u_tint * 0.08;
  color.g -= u_tint * 0.12;
  color.b += u_tint * 0.08;

  // 8. Faded film matte curve (lifting black point)
  if (u_fade > 0.01) {
    color.rgb = color.rgb * (1.0 - u_fade * 0.35) + vec3(u_fade * 0.2);
  }

  // 9. Sepia tone
  if (u_sepia > 0.01) {
    vec3 sepiaColor;
    sepiaColor.r = dot(color.rgb, vec3(0.393, 0.769, 0.189));
    sepiaColor.g = dot(color.rgb, vec3(0.349, 0.686, 0.168));
    sepiaColor.b = dot(color.rgb, vec3(0.272, 0.534, 0.131));
    color.rgb = mix(color.rgb, sepiaColor, u_sepia);
  }

  // 10. Monochrome (B&W)
  if (u_monochrome > 0.01) {
    float mono = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    color.rgb = mix(color.rgb, vec3(mono), u_monochrome);
  }

  // 11. Dynamic Procedural Light Leak
  if (u_lightLeak > 0.01) {
    vec2 leakOrigin = vec2(1.0, 0.0);
    if (u_lightLeakPosition == 1) leakOrigin = vec2(0.0, 0.0);
    else if (u_lightLeakPosition == 2) leakOrigin = vec2(1.0, 1.0);
    else if (u_lightLeakPosition == 3) leakOrigin = vec2(0.0, 1.0);
    else if (u_lightLeakPosition == 4) leakOrigin = vec2(0.5, 0.0);

    // Seed jitter to make every seed leak unique in shape & angle
    float seedAngle = sin(u_seed * 0.13) * 0.3;
    vec2 rotatedUv = uv - leakOrigin;
    float leakDist = length(rotatedUv + vec2(seedAngle, -seedAngle));
    
    // Multi-radial flare
    float leakIntensity = smoothstep(1.3, 0.0, leakDist);
    float beam = pow(leakIntensity, 1.8) * u_lightLeak;
    
    // Color from uniform hue
    vec3 leakColor = hsl2rgb(vec3(u_lightLeakHue / 360.0, 0.85, 0.6));
    color.rgb += leakColor * beam * 0.9;
  }

  // 12. Dynamic Dust & Scratches
  if (u_dustScratches > 0.01) {
    vec2 seedUv = uv * u_resolution * 0.002 + vec2(u_seed * 17.13, u_seed * 31.77);
    
    // Vertical hairline scratches
    float scratchGrid = fract(uv.x * 25.0 + u_seed * 5.7);
    float scratchNoise = hash(vec2(floor(uv.x * 25.0 + u_seed * 5.7), u_seed));
    if (scratchNoise > (1.0 - u_dustScratches * 0.15)) {
      float scratchLine = smoothstep(0.04, 0.0, abs(scratchGrid - 0.5));
      color.rgb += vec3(scratchLine * 0.35);
    }

    // Dust particles
    float dustHash = hash(floor(seedUv * 80.0));
    if (dustHash > (1.0 - u_dustScratches * 0.008)) {
      float speck = hash(floor(seedUv * 120.0));
      if (speck > 0.5) {
        color.rgb -= vec3(0.25); // Dark speck
      } else {
        color.rgb += vec3(0.35); // White speck
      }
    }
  }

  // 13. Dynamic Film Grain (Analog noise)
  if (u_filmGrain > 0.01) {
    vec2 grainUv = uv * (u_resolution / max(u_grainSize, 0.5));
    float grain = hash(grainUv + vec2(u_seed * 19.3, fract(u_time * 0.05)));
    float grainWeight = (grain - 0.5) * u_filmGrain * 0.35;
    color.rgb += vec3(grainWeight);
  }

  // 14. Scanlines (CRT / VHS effect)
  if (u_scanlines > 0.01) {
    float scanline = sin(uv.y * u_resolution.y * 1.2) * 0.5 + 0.5;
    color.rgb *= mix(1.0, 0.7 + 0.3 * scanline, u_scanlines);
  }

  // 15. Vignette (Darkened edges)
  if (u_vignette > 0.01) {
    vec2 vigUv = (uv - vec2(0.5)) * vec2(1.0, mix(1.0, u_resolution.y / u_resolution.x, u_vignetteRoundness));
    float vigDist = length(vigUv);
    float vigFactor = smoothstep(0.75, 0.25, vigDist);
    color.rgb *= mix(1.0, vigFactor, u_vignette);
  }

  // 16. Retro Dithering / Color Depth Reduction (Bayer style)
  if (u_dither > 0.01) {
    vec2 ditherCoord = mod(floor(uv * u_resolution), 2.0);
    float ditherVal = (ditherCoord.x + 2.0 * ditherCoord.y) / 4.0 - 0.375;
    color.rgb += ditherVal * (u_dither * 0.15);
    // Quantize steps
    float steps = mix(256.0, 8.0, u_dither);
    color.rgb = floor(color.rgb * steps + 0.5) / steps;
  }

  // Clamp output colors
  gl_FragColor = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
}
`;

export class WebGLRetroRenderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private texture: WebGLTexture | null = null;
  private posBuffer: WebGLBuffer | null = null;
  private texBuffer: WebGLBuffer | null = null;
  private uniformLocs: Record<string, WebGLUniformLocation | null> = {};

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.initGL();
  }

  private initGL(): boolean {
    try {
      this.gl = this.canvas.getContext('webgl', {
        preserveDrawingBuffer: true,
        alpha: false,
        antialias: false,
      });

      if (!this.gl) {
        console.error('WebGL not supported');
        return false;
      }

      const gl = this.gl;
      const vertShader = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
      const fragShader = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

      if (!vertShader || !fragShader) return false;

      const program = gl.createProgram();
      if (!program) return false;

      gl.attachShader(program, vertShader);
      gl.attachShader(program, fragShader);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return false;
      }

      this.program = program;
      gl.useProgram(program);

      // Setup Geometry: Fullscreen quad
      const positions = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
      this.posBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

      const aPos = gl.getAttribLocation(program, 'a_position');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      // Texture coords (inverted Y for standard web image orientation)
      const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
      this.texBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.texBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

      const aTex = gl.getAttribLocation(program, 'a_texCoord');
      gl.enableVertexAttribArray(aTex);
      gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 0);

      // Cache uniform locations
      const uniformNames = [
        'u_image',
        'u_resolution',
        'u_time',
        'u_brightness',
        'u_contrast',
        'u_saturation',
        'u_warmth',
        'u_tint',
        'u_fade',
        'u_sepia',
        'u_monochrome',
        'u_filmGrain',
        'u_grainSize',
        'u_vignette',
        'u_vignetteRoundness',
        'u_seed',
        'u_dustScratches',
        'u_lightLeak',
        'u_lightLeakHue',
        'u_lightLeakPosition',
        'u_chromaticAberration',
        'u_bloomGlow',
        'u_scanlines',
        'u_pixelation',
        'u_dither',
        'u_lensBlur',
      ];

      for (const name of uniformNames) {
        this.uniformLocs[name] = gl.getUniformLocation(program, name);
      }

      // Create Texture
      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      return true;
    } catch (err) {
      console.error('WebGL init failed:', err);
      return false;
    }
  }

  private compileShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;
    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  render(
    source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageBitmap,
    settings: RetroSettings,
    time: number = 0
  ): void {
    const gl = this.gl;
    if (!gl || !this.program) return;

    // Check canvas dimensions
    const width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
    const height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;

    if (width === 0 || height === 0) return;

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      gl.viewport(0, 0, width, height);
    }

    gl.useProgram(this.program);

    // Upload texture
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    // Set uniforms
    gl.uniform1i(this.uniformLocs.u_image, 0);
    gl.uniform2f(this.uniformLocs.u_resolution, width, height);
    gl.uniform1f(this.uniformLocs.u_time, time);

    gl.uniform1f(this.uniformLocs.u_brightness, settings.brightness);
    gl.uniform1f(this.uniformLocs.u_contrast, settings.contrast);
    gl.uniform1f(this.uniformLocs.u_saturation, settings.saturation);
    gl.uniform1f(this.uniformLocs.u_warmth, settings.warmth);
    gl.uniform1f(this.uniformLocs.u_tint, settings.tint);
    gl.uniform1f(this.uniformLocs.u_fade, settings.fade);
    gl.uniform1f(this.uniformLocs.u_sepia, settings.sepia);
    gl.uniform1f(this.uniformLocs.u_monochrome, settings.monochrome);

    gl.uniform1f(this.uniformLocs.u_filmGrain, settings.filmGrain);
    gl.uniform1f(this.uniformLocs.u_grainSize, settings.grainSize);
    gl.uniform1f(this.uniformLocs.u_vignette, settings.vignette);
    gl.uniform1f(this.uniformLocs.u_vignetteRoundness, settings.vignetteRoundness);
    gl.uniform1f(this.uniformLocs.u_seed, settings.seed);
    gl.uniform1f(this.uniformLocs.u_dustScratches, settings.dustScratches);
    gl.uniform1f(this.uniformLocs.u_lightLeak, settings.lightLeak);
    gl.uniform1f(this.uniformLocs.u_lightLeakHue, settings.lightLeakHue);
    gl.uniform1i(this.uniformLocs.u_lightLeakPosition, settings.lightLeakPosition);
    gl.uniform1f(this.uniformLocs.u_chromaticAberration, settings.chromaticAberration);
    gl.uniform1f(this.uniformLocs.u_bloomGlow, settings.bloomGlow);
    gl.uniform1f(this.uniformLocs.u_scanlines, settings.scanlines);
    gl.uniform1f(this.uniformLocs.u_pixelation, settings.pixelation);
    gl.uniform1f(this.uniformLocs.u_dither, settings.dither);
    gl.uniform1f(this.uniformLocs.u_lensBlur, settings.lensBlur);

    // Draw full quad
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy(): void {
    if (!this.gl) return;
    if (this.texture) this.gl.deleteTexture(this.texture);
    if (this.posBuffer) this.gl.deleteBuffer(this.posBuffer);
    if (this.texBuffer) this.gl.deleteBuffer(this.texBuffer);
    if (this.program) this.gl.deleteProgram(this.program);
    this.gl = null;
  }
}
