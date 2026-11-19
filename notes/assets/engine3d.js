// Shared pure-Canvas 3D perspective engine for all learn-ai 3D visualizers.
// Classic script (non-module) so HTML files open cleanly offline via file:// without CORS errors.
(function (global) {
  const colorCache = new Map();

  function getThemeColors() {
    const s = getComputedStyle(document.documentElement);
    const get = (k, fb) => s.getPropertyValue(k).trim() || fb;
    return {
      bg: get("--content", "#1a1f2e"),
      fg: get("--foreground", "#e8edf7"),
      muted: get("--muted-foreground", "#94a3b8"),
      border: get("--border", "#2a3245"),
      primary: get("--primary", "#4f8cff"),
      success: get("--success", "#22c55e"),
      error: get("--error", "#ef4444"),
      warning: get("--warning", "#f59e0b"),
      card: get("--card", "#151924"),
    };
  }

  function parseRgb(cssColor) {
    if (colorCache.has(cssColor)) return colorCache.get(cssColor);
    const tmp = document.createElement("canvas").getContext("2d");
    tmp.fillStyle = cssColor;
    tmp.fillRect(0, 0, 1, 1);
    const d = tmp.getImageData(0, 0, 1, 1).data;
    const res = [d[0], d[1], d[2]];
    colorCache.set(cssColor, res);
    return res;
  }

  function clearColorCache() {
    colorCache.clear();
  }

  if (global.matchMedia) {
    const mq = global.matchMedia("(prefers-color-scheme: light)");
    if (mq.addEventListener) mq.addEventListener("change", clearColorCache);
  }
  global.addEventListener("themechange", clearColorCache);

  function shade(rgb, factor, alpha = 1) {
    const r = Math.min(255, Math.max(0, Math.round(rgb[0] * factor)));
    const g = Math.min(255, Math.max(0, Math.round(rgb[1] * factor)));
    const b = Math.min(255, Math.max(0, Math.round(rgb[2] * factor)));
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function lerpRgb(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function createCamera(canvas, opts) {
    const cam = {
      yaw: opts.yaw ?? -0.5,
      pitch: opts.pitch ?? 0.5,
      dist: opts.dist ?? 24.0,
      targetYaw: opts.yaw ?? -0.5,
      targetPitch: opts.pitch ?? 0.5,
      targetDist: opts.dist ?? 24.0,
      tx: opts.tx ?? 0.0,
      ty: opts.ty ?? 0.6,
      tz: opts.tz ?? 0.0,
      minPitch: opts.minPitch ?? 0.14,
      maxPitch: opts.maxPitch ?? 1.42,
      minDist: opts.minDist ?? 14,
      maxDist: opts.maxDist ?? 38,
      autoOrbit: false,
      step() {
        if (this.autoOrbit) this.targetYaw += 0.005;
        this.yaw += (this.targetYaw - this.yaw) * 0.12;
        this.pitch += (this.targetPitch - this.pitch) * 0.12;
        this.dist += (this.targetDist - this.dist) * 0.12;
      },
      setPreset(yaw, pitch, dist) {
        this.autoOrbit = false;
        this.targetYaw = yaw;
        this.targetPitch = pitch;
        this.targetDist = dist;
      },
    };

    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener("pointerdown", e => {
      dragging = true;
      cam.autoOrbit = false;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (!dragging) return;
      cam.targetYaw -= (e.clientX - lastX) * 0.008;
      cam.targetPitch = Math.max(
        cam.minPitch,
        Math.min(cam.maxPitch, cam.targetPitch + (e.clientY - lastY) * 0.007)
      );
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", () => {
      dragging = false;
    });
    canvas.addEventListener(
      "wheel",
      e => {
        e.preventDefault();
        cam.targetDist = Math.max(
          cam.minDist,
          Math.min(cam.maxDist, cam.targetDist + e.deltaY * 0.015)
        );
      },
      { passive: false }
    );

    return cam;
  }

  function beginFrame(canvas, ctx, theme) {
    const dpr = global.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      const tw = Math.round(rect.width * dpr);
      const th = Math.round(rect.height * dpr);
      if (canvas.width !== tw || canvas.height !== th) {
        canvas.width = tw;
        canvas.height = th;
      }
    }
    const W = rect.width;
    const H = rect.height;
    ctx.save();
    ctx.scale(dpr, dpr);

    const bgGrad = ctx.createRadialGradient(
      W * 0.5,
      H * 0.42,
      40,
      W * 0.5,
      H * 0.5,
      Math.max(W, H) * 0.72
    );
    bgGrad.addColorStop(0, theme.card);
    bgGrad.addColorStop(1, theme.bg);
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);
    return { W, H };
  }

  function makeProjector(W, H, cam, focalFactor = 1.18, centerY = 0.53) {
    const cy = Math.cos(cam.yaw);
    const sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch);
    const sp = Math.sin(cam.pitch);
    const focal = Math.min(W, H) * focalFactor;
    const eye = {
      x: cam.tx + cam.dist * cp * sy,
      y: cam.ty + cam.dist * sp,
      z: cam.tz + cam.dist * cp * cy,
    };

    function project(wx, wy, wz) {
      const dx = wx - cam.tx;
      const dy = wy - cam.ty;
      const dz = wz - cam.tz;
      const x1 = dx * cy - dz * sy;
      const z1 = dx * sy + dz * cy;
      const y2 = dy * cp - z1 * sp;
      const z2 = dy * sp + z1 * cp;
      const depth = cam.dist - z2;
      if (depth < 0.5) return null;
      const sc = focal / depth;
      return { x: W * 0.5 + x1 * sc, y: H * centerY - y2 * sc, depth, sc };
    }

    return { project, eye };
  }

  function pushBox(dl, ctx, proj, eye, x0, y0, z0, x1, y1, z1, rgb, alpha = 1, strokeCss = null) {
    const cs = [
      proj(x0, y0, z0), proj(x1, y0, z0), proj(x1, y0, z1), proj(x0, y0, z1),
      proj(x0, y1, z0), proj(x1, y1, z0), proj(x1, y1, z1), proj(x0, y1, z1),
    ];
    if (cs.some(c => !c)) return;
    const cp = proj((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
    if (!cp) return;
    const faces = [];
    if (eye.y > y1) faces.push({ i: [4, 5, 6, 7], l: 1.12 });
    if (eye.y < y0) faces.push({ i: [0, 3, 2, 1], l: 0.55 });
    if (eye.x > x1) faces.push({ i: [1, 2, 6, 5], l: 0.88 });
    if (eye.x < x0) faces.push({ i: [0, 4, 7, 3], l: 0.72 });
    if (eye.z > z1) faces.push({ i: [3, 7, 6, 2], l: 0.96 });
    if (eye.z < z0) faces.push({ i: [0, 1, 5, 4], l: 0.66 });
    dl.push({
      depth: cp.depth,
      draw: () => {
        for (const f of faces) {
          ctx.beginPath();
          ctx.moveTo(cs[f.i[0]].x, cs[f.i[0]].y);
          ctx.lineTo(cs[f.i[1]].x, cs[f.i[1]].y);
          ctx.lineTo(cs[f.i[2]].x, cs[f.i[2]].y);
          ctx.lineTo(cs[f.i[3]].x, cs[f.i[3]].y);
          ctx.closePath();
          ctx.fillStyle = shade(rgb, f.l, alpha);
          ctx.fill();
          if (strokeCss) {
            ctx.strokeStyle = strokeCss;
            ctx.lineWidth = 0.85;
            ctx.stroke();
          }
        }
      },
    });
  }

  function pushOrb(dl, ctx, proj, wx, wy, wz, radiusWorld, colorCss, glowCss = null, label = null, fgCss = "#fff") {
    const p = proj(wx, wy, wz);
    if (!p) return;
    const r = Math.max(3, radiusWorld * p.sc);
    dl.push({
      depth: p.depth - 0.18,
      draw: () => {
        ctx.save();
        if (glowCss) {
          ctx.shadowColor = glowCss;
          ctx.shadowBlur = 15;
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = colorCss;
        ctx.fill();
        ctx.restore();
        if (label) {
          ctx.fillStyle = fgCss;
          ctx.font = `bold ${Math.max(10, Math.round(r * 0.92))}px monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, p.x, p.y + 0.5);
        }
      },
    });
  }

  function pushArrow3D(dl, ctx, proj, wx, y0, y1, wz, colorCss, lineWidth = 3) {
    const p0 = proj(wx, y0, wz);
    const p1 = proj(wx, y1, wz);
    if (!p0 || !p1) return;
    dl.push({
      depth: (p0.depth + p1.depth) * 0.5 - 0.28,
      draw: () => {
        ctx.save();
        ctx.strokeStyle = colorCss;
        ctx.fillStyle = colorCss;
        ctx.lineWidth = lineWidth;
        ctx.shadowColor = colorCss;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
        const a = Math.atan2(p1.y - p0.y, p1.x - p0.x);
        const head = 7 + lineWidth * 0.55;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p1.x - head * Math.cos(a - 0.5), p1.y - head * Math.sin(a - 0.5));
        ctx.lineTo(p1.x - head * Math.cos(a + 0.5), p1.y - head * Math.sin(a + 0.5));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      },
    });
  }

  function pushLabel3D(dl, ctx, proj, wx, wy, wz, text, colorCss, font = "11px monospace", bias = -0.15) {
    const p = proj(wx, wy, wz);
    if (!p) return;
    dl.push({
      depth: p.depth + bias,
      draw: () => {
        ctx.fillStyle = colorCss;
        ctx.font = font;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, p.x, p.y);
      },
    });
  }

  function drawPipe3D(ctx, proj, a, b, colorCss, width = 2) {
    const p0 = proj(a.x, a.y, a.z);
    const p1 = proj(b.x, b.y, b.z);
    if (!p0 || !p1) return;
    ctx.save();
    ctx.strokeStyle = colorCss;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
    ctx.restore();
  }

  function drawBezierArc3D(ctx, proj, a, b, arcH, tProg, colorCss, width = 2, dashed = false) {
    ctx.save();
    ctx.strokeStyle = colorCss;
    ctx.lineWidth = width;
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= 20; i++) {
      const u = (i / 20) * tProg;
      const p = proj(
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u + Math.sin(u * Math.PI) * arcH,
        a.z + (b.z - a.z) * u
      );
      if (!p) continue;
      if (!started) {
        ctx.moveTo(p.x, p.y);
        started = true;
      } else {
        ctx.lineTo(p.x, p.y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function evalArc(a, b, arcH, u) {
    return {
      x: a.x + (b.x - a.x) * u,
      y: a.y + (b.y - a.y) * u + Math.sin(u * Math.PI) * arcH,
      z: a.z + (b.z - a.z) * u,
    };
  }

  function flushDrawList(dl) {
    dl.sort((a, b) => b.depth - a.depth);
    for (const item of dl) item.draw();
  }

  global.Mini3D = {
    getThemeColors,
    parseRgb,
    shade,
    lerpRgb,
    createCamera,
    beginFrame,
    makeProjector,
    pushBox,
    pushOrb,
    pushArrow3D,
    pushLabel3D,
    drawPipe3D,
    drawBezierArc3D,
    evalArc,
    flushDrawList,
  };
})(window);
