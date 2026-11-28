// Glyd's chat diagrams: the Mermaid flowcharts models write, drawn as SVG with our own layered layout.
// Reads flowchart/graph TD|TB|BT|LR|RL; shapes [] () ([]) (()) {} {{}} [()] [[]] [/ /] >]; edges --> --- -.-> ==> --o --x ~~~ with
// |labels| or "-- label -->"; chains (A --> B --> C) and "A & B --> C". Styles, classes and subgraph boxes are skipped; any other kind of
// diagram returns null and the page shows its source. Every piece of text is escaped.
(function () {
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const SHAPES = [["([", "])", "stadium"], ["[[", "]]", "sub"], ["[(", ")]", "db"], ["((", "))", "circle"], ["{{", "}}", "hex"],
                  ["[/", "/]", "para"], ["[\\", "\\]", "para"], ["[", "]", "rect"], ["(", ")", "round"], ["{", "}", "diamond"], [">", "]", "flag"]];
  const ARROW = "-{2,}>|-{3,}|={2,}>|={3,}|-\\.+->|-\\.+-|--o|--x|~~~";
  const EDGE = new RegExp(`^\\s*(<)?(${ARROW})\\s*(?:\\|([^|]*)\\|)?`);
  const EDGE_TEXT = /^\s*(--|==|-\.)\s+(.+?)\s+(-{2,}>|-{3,}|={2,}>|={3,}|\.+->|\.+-)/;
  const SKIP = /^(classDef|class|style|linkStyle|click|subgraph|end|direction|%%)\b/;
  const kindOf = t => ({ dot: t.includes("."), thick: t.includes("="), arrow: />$|o$|x$/.test(t), hidden: t === "~~~" });

  let ctx = null;
  function measure(t, font) {
    if (typeof document === "undefined") return t.length * 7.2;
    ctx = ctx || document.createElement("canvas").getContext("2d");
    ctx.font = font; return ctx.measureText(t).width;
  }
  const FONT = '13.5px "Instrument Sans", system-ui, sans-serif', LH = 18;
  function lines(text) {
    const out = [];
    for (const part of text.replace(/#quot;/g, '"').replace(/[`*]/g, "").split(/<br\s*\/?>|\\n/i)) {
      let cur = "";
      for (const w of part.trim().split(/\s+/)) {
        if (cur && (cur + " " + w).length > 26) { out.push(cur); cur = w; } else cur = cur ? cur + " " + w : w;
      }
      out.push(cur);
    }
    return out;
  }

  function node(s, g) {
    const m = s.match(/^\s*([\w$À-￿]+)/);
    if (!m) return null;
    let rest = s.slice(m[0].length), text = null, shape = null;
    for (const [o, c, sh] of SHAPES) {
      if (!rest.startsWith(o)) continue;
      const body = rest.slice(o.length);
      let end;
      if (body.startsWith('"')) { const q = body.indexOf('"', 1); if (q < 0) return null; text = body.slice(1, q); end = body.indexOf(c, q); }
      else { end = body.indexOf(c); text = body.slice(0, end); }
      if (end < 0) return null;
      rest = body.slice(end + c.length); shape = sh; break;
    }
    rest = rest.replace(/^:::[\w-]+/, "");
    const id = m[1];
    if (!g.nodes[id]) { g.nodes[id] = { id, text: id, shape: "rect" }; g.order.push(id); }
    if (text !== null) { g.nodes[id].text = text.trim() || " "; g.nodes[id].shape = shape; }
    return { id, rest };
  }

  function statement(line, g) {
    let s = line, prev = null;
    for (;;) {
      const ids = [];
      for (;;) {
        const r = node(s, g);
        if (!r) return;
        ids.push(r.id); s = r.rest;
        const amp = s.match(/^\s*&/);
        if (!amp) break;
        s = s.slice(amp[0].length);
      }
      if (prev) for (const a of prev.ids) for (const b of ids) g.edges.push({ a, b, ...prev.edge });
      if (!s.trim()) return;
      let m = s.match(EDGE), e;
      if (m) e = { label: (m[3] || "").trim(), ...kindOf(m[2]) };
      else if ((m = s.match(EDGE_TEXT))) e = { label: m[2].trim(), ...kindOf(m[1] + m[3]) };
      else return;
      s = s.slice(m[0].length); prev = { ids, edge: e };
    }
  }

  function parse(src) {
    const rows = src.replace(/\r/g, "").split("\n").map(l => l.replace(/%%.*$/, "").trim().replace(/;$/, "")).filter(Boolean);
    const head = rows.length && rows[0].match(/^(flowchart|graph)(?:\s+(TD|TB|BT|LR|RL))?\s*$/i);
    if (!head) return null;
    const g = { dir: (head[2] || "TD").toUpperCase(), nodes: {}, order: [], edges: [] };
    for (const l of rows.slice(1)) if (!SKIP.test(l)) statement(l, g);
    return g.order.length ? g : null;
  }

  function layout(g) {
    const N = g.order.map(id => g.nodes[id]), lr = g.dir === "LR" || g.dir === "RL";
    for (const n of N) {
      n.lines = lines(n.text);
      const tw = Math.max(...n.lines.map(t => measure(t, FONT))), th = n.lines.length * LH;
      [n.w, n.h] = n.shape === "diamond" ? [tw * 1.35 + 40, th * 1.5 + 30] : n.shape === "circle" ? Array(2).fill(Math.max(tw, th) + 30)
        : [tw + (n.shape === "hex" || n.shape === "para" || n.shape === "flag" ? 50 : 32), th + (n.shape === "db" ? 34 : 20)];
      n.su = lr ? n.h : n.w; n.sv = lr ? n.w : n.h;   // its size across the layer (u) and along the ranks (v)
    }
    // ranks: the longest path, with the edges that close a cycle (found by a depth-first walk from the sources) left out
    const out = {}, inn = {};
    N.forEach(n => { out[n.id] = []; inn[n.id] = 0; });
    const E = g.edges.filter(e => e.a !== e.b);
    E.forEach(e => { out[e.a].push(e); inn[e.b]++; });
    const seen = {};
    const dfs = id => { seen[id] = 1; for (const e of out[id]) { if (seen[e.b] === 1) e.back = true; else if (!seen[e.b]) dfs(e.b); } seen[id] = 2; };
    N.filter(n => !inn[n.id]).concat(N).forEach(n => { if (!seen[n.id]) dfs(n.id); });
    const rank = {}, nbCount = id => E.some(e => e.a === id || e.b === id);
    N.forEach(n => rank[n.id] = 0);
    for (let i = 0; i < N.length; i++) for (const e of E) if (!e.back && rank[e.b] < rank[e.a] + 1) rank[e.b] = rank[e.a] + 1;
    const last = Math.max(0, ...N.map(n => rank[n.id]));
    N.forEach(n => { if (!nbCount(n.id)) rank[n.id] = last; });   // a node with no edges goes at the end, not beside the start
    const layers = [];
    N.forEach(n => (layers[rank[n.id]] = layers[rank[n.id]] || []).push(n));
    // order within a layer: a few sweeps toward the mean place of each node's neighbors (fewer crossings)
    const nb = {};
    N.forEach(n => nb[n.id] = []);
    E.forEach(e => { nb[e.a].push(e.b); nb[e.b].push(e.a); });
    const pos = {};
    const index = () => layers.forEach(L => L.forEach((n, i) => pos[n.id] = i - (L.length - 1) / 2));
    index();
    for (let it = 0; it < 6; it++) {
      const down = it % 2 === 0;
      for (let k = 0; k < layers.length; k++) {
        const r = down ? k : layers.length - 1 - k, L = layers[r];
        const key = n => { const o = nb[n.id].filter(id => down ? rank[id] < r : rank[id] > r); return o.length ? o.reduce((s, id) => s + pos[id], 0) / o.length : pos[n.id]; };
        const keys = new Map(L.map(n => [n, key(n)]));
        L.sort((a, b) => keys.get(a) - keys.get(b));
        L.forEach((n, i) => pos[n.id] = i - (L.length - 1) / 2);
      }
    }
    // places across the layer: packed and centered, then pulled toward the neighbors' mean (pushed apart left-to-right and
    // right-to-left, the two averaged) so chains run straight
    const GAP = 34, u = {};
    layers.forEach(L => { let x = -L.reduce((s, n) => s + n.su + GAP, -GAP) / 2; L.forEach(n => { u[n.id] = x + n.su / 2; x += n.su + GAP; }); });
    for (let it = 0; it < 8; it++) {
      const down = it % 2 === 0;
      for (let k = 1; k < layers.length; k++) {
        const r = down ? k : layers.length - 1 - k, L = layers[r];
        const want = L.map(n => { const o = nb[n.id].filter(id => down ? rank[id] < r : rank[id] > r); return o.length ? o.reduce((s, id) => s + u[id], 0) / o.length : u[n.id]; });
        const a = want.slice(), b = want.slice();
        for (let i = 1; i < L.length; i++) a[i] = Math.max(a[i], a[i - 1] + (L[i - 1].su + L[i].su) / 2 + GAP);
        for (let i = L.length - 2; i >= 0; i--) b[i] = Math.min(b[i], b[i + 1] - (L[i + 1].su + L[i].su) / 2 - GAP);
        L.forEach((n, i) => u[n.id] = (a[i] + b[i]) / 2);
        for (let i = 1; i < L.length; i++) u[L[i].id] = Math.max(u[L[i].id], u[L[i - 1].id] + (L[i - 1].su + L[i].su) / 2 + GAP);
      }
    }
    const v = {}, RGAP = E.some(e => e.label) ? 70 : 54;
    let y = 0;
    layers.forEach(L => { const h = Math.max(...L.map(n => n.sv)); L.forEach(n => v[n.id] = y + h / 2); y += h + RGAP; });
    const minU = Math.min(...N.map(n => u[n.id] - n.su / 2)), maxU = Math.max(...N.map(n => u[n.id] + n.su / 2)), maxV = y - RGAP;
    const back = E.filter(e => e.back || rank[e.b] <= rank[e.a]), PAD = 12;
    const side = back.length ? 40 + 14 * back.length + Math.max(0, ...back.map(e => e.label ? Math.max(...lines(e.label).map(t => measure(t, FONT))) / 2 + 10 : 0)) : 0;
    const flip = g.dir === "BT" || g.dir === "RL";
    N.forEach(n => { n.u = u[n.id] - minU + PAD; n.v = flip ? maxV - v[n.id] + PAD : v[n.id] + PAD; });
    return { N, E, lr, flip, W: maxU - minU + 2 * PAD + side, H: maxV + 2 * PAD, rank, R: maxU - minU + PAD };
  }

  let uid = 0;
  function render(src) {
    let g;
    try { g = parse(src); } catch (e) { return null; }
    if (!g || g.order.length > 120) return null;
    const { N, E, lr, flip, W, H, rank, R } = layout(g), id = "f" + (++uid);
    const xy = (uu, vv) => lr ? [vv, uu] : [uu, vv];   // (across, along) to (x, y)
    const [SW, SH] = xy(W, H);
    const P = (uu, vv) => xy(uu, vv).map(c => c.toFixed(1)).join(" ");
    let edges = "", labels = "", k = 0;
    for (const e of E) {
      if (e.hidden) continue;
      const a = g.nodes[e.a], b = g.nodes[e.b], dir = flip ? -1 : 1;
      let d, mid;
      if (!e.back && rank[e.b] > rank[e.a]) {
        const s = [a.u, a.v + dir * a.sv / 2], t = [b.u, b.v - dir * b.sv / 2], c = (s[1] + t[1]) / 2;
        d = `M${P(...s)} C${P(s[0], c)} ${P(t[0], c)} ${P(...t)}`;
        mid = [(s[0] + t[0]) / 2, c];
      } else {   // back to an earlier layer: out to the right, up past everything, and in
        const off = 24 + 14 * (k++), s = [a.u + a.su / 2, a.v], t = [b.u + b.su / 2, b.v], x = R + off;
        d = `M${P(...s)} L${P(x - 12, s[1])} Q${P(x, s[1])} ${P(x, s[1] - 12 * dir)} L${P(x, t[1] + 12 * dir)} Q${P(x, t[1])} ${P(x - 12, t[1])} L${P(...t)}`;
        mid = [x, (s[1] + t[1]) / 2];
      }
      edges += `<path class="ed${e.dot ? " dot" : ""}${e.thick ? " thick" : ""}" d="${d}"${e.arrow ? ` marker-end="url(#${id}a)"` : ""}/>`;
      if (e.label) {
        const t = lines(e.label), w = Math.max(...t.map(s => measure(s, FONT))) + 12, h = t.length * LH + 4, [mx, my] = xy(...mid);
        labels += `<rect class="el" x="${(mx - w / 2).toFixed(1)}" y="${(my - h / 2).toFixed(1)}" width="${w.toFixed(1)}" height="${h}" rx="4"/>` +
          `<text x="${mx.toFixed(1)}" y="${(my - (t.length - 1) * LH / 2).toFixed(1)}">${t.map((s, i) => `<tspan x="${mx.toFixed(1)}" dy="${i ? LH : 0}">${esc(s)}</tspan>`).join("")}</text>`;
      }
    }
    let nodes = "";
    for (const n of N) {
      const [cx, cy] = xy(n.u, n.v), w = n.w, h = n.h, l = cx - w / 2, t = cy - h / 2, f = v => v.toFixed(1);
      const poly = pts => `<polygon class="nd" points="${pts.map(p => p.map(f).join(",")).join(" ")}"/>`;
      let sh;
      switch (n.shape) {
        case "diamond": sh = poly([[cx, t], [l + w, cy], [cx, t + h], [l, cy]]); break;
        case "hex": sh = poly([[l + 16, t], [l + w - 16, t], [l + w, cy], [l + w - 16, t + h], [l + 16, t + h], [l, cy]]); break;
        case "para": sh = poly([[l + 14, t], [l + w, t], [l + w - 14, t + h], [l, t + h]]); break;
        case "flag": sh = poly([[l, t], [l + w, t], [l + w, t + h], [l, t + h], [l + 14, cy]]); break;
        case "circle": sh = `<circle class="nd" cx="${f(cx)}" cy="${f(cy)}" r="${f(w / 2)}"/>`; break;
        case "db": sh = `<path class="nd" d="M${f(l)} ${f(t + 8)} a${f(w / 2)} 8 0 0 1 ${f(w)} 0 v${f(h - 16)} a${f(w / 2)} 8 0 0 1 ${f(-w)} 0 z m0 0 a${f(w / 2)} 8 0 0 0 ${f(w)} 0"/>`; break;
        case "sub": sh = `<rect class="nd" x="${f(l)}" y="${f(t)}" width="${f(w)}" height="${f(h)}" rx="4"/><path class="nd" d="M${f(l + 8)} ${f(t)} v${f(h)} M${f(l + w - 8)} ${f(t)} v${f(h)}"/>`; break;
        default: sh = `<rect class="nd" x="${f(l)}" y="${f(t)}" width="${f(w)}" height="${f(h)}" rx="${n.shape === "stadium" ? f(h / 2) : n.shape === "round" ? 14 : 6}"/>`;
      }
      const y0 = cy - (n.lines.length - 1) * LH / 2 + (n.shape === "db" ? 4 : 0);
      nodes += `<g class="${n.shape === "diamond" ? "q" : ""}">${sh}<text x="${f(cx)}" y="${f(y0)}">${n.lines.map((s, i) => `<tspan x="${f(cx)}" dy="${i ? LH : 0}">${esc(s)}</tspan>`).join("")}</text></g>`;
    }
    return `<svg class="flow" viewBox="0 0 ${SW.toFixed(0)} ${SH.toFixed(0)}" width="${SW.toFixed(0)}" role="img" aria-label="flowchart">` +
      `<defs><marker id="${id}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0 0 L10 5 L0 10 z"/></marker></defs>` +
      `${edges}${nodes}${labels}</svg>`;
  }
  const api = { render, parse };
  if (typeof window !== "undefined") window.HBflow = api;
  if (typeof module !== "undefined") module.exports = api;
})();
