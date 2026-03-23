import { useState, useMemo, useEffect, useRef } from "react";

/* ─── PRNG ─── */
function rng(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ─── Linear Algebra ─── */
function matMul(A, B, rA, cA, cB) {
  const C = new Array(rA * cB).fill(0);
  for (let i = 0; i < rA; i++)
    for (let j = 0; j < cB; j++)
      for (let k = 0; k < cA; k++) C[i * cB + j] += A[i * cA + k] * B[k * cB + j];
  return C;
}
function addMat(A, B) { return A.map((v, i) => v + (B[i] || 0)); }
function transpose(f, r, c) { const o = new Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) o[j * r + i] = f[i * c + j]; return o; }
function softmaxRows(f, r, c) {
  const o = [...f];
  for (let i = 0; i < r; i++) {
    let mx = -Infinity; for (let j = 0; j < c; j++) mx = Math.max(mx, o[i * c + j]);
    let s = 0; for (let j = 0; j < c; j++) { o[i * c + j] = Math.exp(o[i * c + j] - mx); s += o[i * c + j]; }
    for (let j = 0; j < c; j++) o[i * c + j] /= s;
  }
  return o;
}
function layerNorm(f, r, c) {
  const o = [...f];
  for (let i = 0; i < r; i++) {
    let m = 0; for (let j = 0; j < c; j++) m += o[i * c + j]; m /= c;
    let v = 0; for (let j = 0; j < c; j++) v += (o[i * c + j] - m) ** 2;
    v = Math.sqrt(v / c + 1e-5);
    for (let j = 0; j < c; j++) o[i * c + j] = (o[i * c + j] - m) / v;
  }
  return o;
}
function gelu(x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))); }
function matStats(f) {
  const mn = Math.min(...f), mx = Math.max(...f);
  const avg = f.reduce((a, b) => a + b, 0) / f.length;
  const std = Math.sqrt(f.reduce((a, b) => a + (b - avg) ** 2, 0) / f.length);
  return { min: mn, max: mx, mean: avg, std };
}

/* ─── Build computation graph ─── */
function buildGraph(seq, dModel, nHeads, ffMult, nLayers, seed) {
  const n = seq, dH = Math.floor(dModel / nHeads), dFF = dModel * ffMult;
  const r0 = rng(seed);
  const nodes = [];
  const add = (o) => { nodes.push({ ...o, id: nodes.length }); return nodes.length - 1; };

  const X0 = new Array(n * dModel).fill(0).map(() => (r0() - 0.5) * 2);
  let prevId = add({
    label: "Input X", desc: `Token + Positional Embeddings`,
    data: X0, rows: n, cols: dModel, phase: "input", layer: -1, head: -1, parents: [],
    dimOp: `${n}×${dModel}`,
  });

  for (let L = 0; L < nLayers; L++) {
    const inputId = prevId;
    const inputData = nodes[inputId].data;
    const headOutputIds = [];

    for (let h = 0; h < nHeads; h++) {
      const hr = rng(seed * 1000 + L * 137 + h * 31);
      const Wq = new Array(dModel * dH).fill(0).map(() => (hr() - 0.5) * (2 / Math.sqrt(dModel)));
      const Wk = new Array(dModel * dH).fill(0).map(() => (hr() - 0.5) * (2 / Math.sqrt(dModel)));
      const Wv = new Array(dModel * dH).fill(0).map(() => (hr() - 0.5) * (2 / Math.sqrt(dModel)));

      const Qd = matMul(inputData, Wq, n, dModel, dH);
      const qId = add({ label: `Q`, desc: `X·Wq projection`, data: Qd, rows: n, cols: dH,
        phase: "qkv", layer: L, head: h, parents: [inputId],
        dimOp: `[${n}×${dModel}]·[${dModel}×${dH}] = [${n}×${dH}]` });

      const Kd = matMul(inputData, Wk, n, dModel, dH);
      const kId = add({ label: `K`, desc: `X·Wk projection`, data: Kd, rows: n, cols: dH,
        phase: "qkv", layer: L, head: h, parents: [inputId],
        dimOp: `[${n}×${dModel}]·[${dModel}×${dH}] = [${n}×${dH}]` });

      const Vd = matMul(inputData, Wv, n, dModel, dH);
      const vId = add({ label: `V`, desc: `X·Wv projection`, data: Vd, rows: n, cols: dH,
        phase: "qkv", layer: L, head: h, parents: [inputId],
        dimOp: `[${n}×${dModel}]·[${dModel}×${dH}] = [${n}×${dH}]` });

      const Kt = transpose(Kd, n, dH);
      let scores = matMul(Qd, Kt, n, dH, n).map(v => v / Math.sqrt(dH));
      const scId = add({ label: `Q·Kᵀ/√d`, desc: `Scaled dot-product`, data: scores, rows: n, cols: n,
        phase: "scores", layer: L, head: h, parents: [qId, kId],
        dimOp: `[${n}×${dH}]·[${dH}×${n}] = [${n}×${n}]` });

      const attn = softmaxRows(scores, n, n);
      const attnId = add({ label: `Softmax`, desc: `Row-wise softmax → weights`, data: attn, rows: n, cols: n,
        phase: "attn", layer: L, head: h, parents: [scId],
        dimOp: `[${n}×${n}] → [${n}×${n}]` });

      const ctx = matMul(attn, Vd, n, n, dH);
      const ctxId = add({ label: `Attn·V`, desc: `Weighted sum of values`, data: ctx, rows: n, cols: dH,
        phase: "context", layer: L, head: h, parents: [attnId, vId],
        dimOp: `[${n}×${n}]·[${n}×${dH}] = [${n}×${dH}]` });

      headOutputIds.push(ctxId);
    }

    const lastCtx = nodes[headOutputIds[headOutputIds.length - 1]];
    const hr2 = rng(seed * 2000 + L * 53);
    const Wo = new Array(dH * dModel).fill(0).map(() => (hr2() - 0.5) * (2 / Math.sqrt(dH)));
    const projected = matMul(lastCtx.data, Wo, n, dH, dModel);
    const normed = layerNorm(addMat(inputData, projected), n, dModel);

    const paId = add({ label: `+Res+LN`, desc: `Output projection + Residual + LayerNorm`,
      data: normed, rows: n, cols: dModel, phase: "post-attn", layer: L, head: -1,
      parents: [...headOutputIds, inputId],
      dimOp: `[${n}×${dH}]·[${dH}×${dModel}] + Res → [${n}×${dModel}]` });

    const fr = rng(seed * 500 + L * 251);
    const W1 = new Array(dModel * dFF).fill(0).map(() => (fr() - 0.5) * (2 / Math.sqrt(dModel)));
    const B1 = new Array(dFF).fill(0).map(() => (fr() - 0.5) * 0.1);
    const W2 = new Array(dFF * dModel).fill(0).map(() => (fr() - 0.5) * (2 / Math.sqrt(dFF)));

    let hid = matMul(normed, W1, n, dModel, dFF);
    for (let i = 0; i < n; i++) for (let j = 0; j < dFF; j++) hid[i * dFF + j] += B1[j];
    const upId = add({ label: `FFN ↑`, desc: `Up-projection + bias`,
      data: hid, rows: n, cols: dFF, phase: "ffn-up", layer: L, head: -1, parents: [paId],
      dimOp: `[${n}×${dModel}]·[${dModel}×${dFF}] = [${n}×${dFF}]` });

    const act = hid.map(gelu);
    const geluId = add({ label: `GELU`, desc: `Gaussian Error Linear Unit`,
      data: act, rows: n, cols: dFF, phase: "ffn-act", layer: L, head: -1, parents: [upId],
      dimOp: `[${n}×${dFF}] → [${n}×${dFF}]` });

    const down = matMul(act, W2, n, dFF, dModel);
    const out = layerNorm(addMat(normed, down), n, dModel);
    const dnId = add({ label: `FFN ↓+LN`, desc: `Down-projection + Residual + LayerNorm`,
      data: out, rows: n, cols: dModel, phase: "ffn-down", layer: L, head: -1, parents: [geluId, paId],
      dimOp: `[${n}×${dFF}]·[${dFF}×${dModel}] + Res → [${n}×${dModel}]` });

    prevId = dnId;
  }
  return nodes;
}

/* ─── Assign positions ─── */
function layoutNodes(nodes, nHeads, nLayers) {
  const pos = [];
  const NW = 96, NH = 56;
  const headSpacing = 145;
  const layerW = 730;

  nodes.forEach((nd) => {
    let x, y;
    if (nd.phase === "input") {
      x = 30; y = 50 + nHeads * headSpacing / 2 - 28;
    } else if (nd.phase === "qkv") {
      const baseX = 170 + nd.layer * layerW;
      x = baseX;
      const qkvOff = { Q: 0, K: 1, V: 2 }[nd.label] ?? 0;
      y = 30 + nd.head * headSpacing + qkvOff * 46;
    } else if (nd.phase === "scores") {
      x = 290 + nd.layer * layerW;
      y = 30 + nd.head * headSpacing + 30;
    } else if (nd.phase === "attn") {
      x = 390 + nd.layer * layerW;
      y = 30 + nd.head * headSpacing + 30;
    } else if (nd.phase === "context") {
      x = 490 + nd.layer * layerW;
      y = 30 + nd.head * headSpacing + 30;
    } else if (nd.phase === "post-attn") {
      x = 600 + nd.layer * layerW;
      y = 50 + nHeads * headSpacing / 2 - 28;
    } else if (nd.phase === "ffn-up") {
      x = 700 + nd.layer * layerW;
      y = 50 + nHeads * headSpacing / 2 - 70;
    } else if (nd.phase === "ffn-act") {
      x = 790 + nd.layer * layerW;
      y = 50 + nHeads * headSpacing / 2 - 70;
    } else if (nd.phase === "ffn-down") {
      x = 880 + nd.layer * layerW;
      y = 50 + nHeads * headSpacing / 2 - 28;
    } else {
      x = nd.id * 110; y = 200;
    }
    pos.push({ x, y });
  });
  return pos;
}

/* ─── Colors ─── */
const PC = {
  input: "#8a8a8a", qkv: "#5a9fd0", scores: "#d0884a", attn: "#e8a030",
  context: "#50c878", "post-attn": "#7ab060", "ffn-up": "#c09830",
  "ffn-act": "#d06850", "ffn-down": "#a070b0",
};
function valColor(v, mx) {
  const t = Math.max(-1, Math.min(1, v / (mx || 1)));
  if (t >= 0) return `rgb(${Math.round(15+t*230)},${Math.round(12+t*110)},${Math.round(8+t*15)})`;
  const a = -t; return `rgb(${Math.round(8+a*15)},${Math.round(18+a*90)},${Math.round(15+a*230)})`;
}

/* ─── Full Matrix ─── */
function FullMatrix({ data, rows, cols }) {
  const mx = Math.max(0.01, ...data.map(Math.abs));
  const maxS = 22;
  const sr = rows > maxS ? Math.ceil(rows / maxS) : 1;
  const sc = cols > maxS ? Math.ceil(cols / maxS) : 1;
  const vr = Math.ceil(rows / sr), vc = Math.ceil(cols / sc);
  const cw = Math.min(Math.max(8, Math.floor(480 / Math.max(vr, vc))), 30);
  const showNum = cw >= 22;
  return (
    <div>
      <svg width={vc * cw + 2} height={vr * cw + 2} style={{ borderRadius: 4, display: "block" }}>
        {Array.from({ length: vr }).map((_, i) =>
          Array.from({ length: vc }).map((_, j) => {
            const val = data[(i * sr) * cols + j * sc] || 0;
            return (
              <g key={`${i}-${j}`}>
                <rect x={j*cw} y={i*cw} width={cw-0.5} height={cw-0.5}
                  fill={valColor(val, mx)} rx={cw > 12 ? 1 : 0} />
                {showNum && <text x={j*cw+cw/2} y={i*cw+cw/2+3} textAnchor="middle"
                  fontSize={Math.min(7, cw-4)} fill="rgba(255,255,255,0.7)" fontFamily="var(--mono)">
                  {Math.abs(val) >= 10 ? val.toFixed(0) : val.toFixed(2)}</text>}
              </g>
            );
          })
        )}
      </svg>
      {(sr > 1 || sc > 1) && <div style={{ fontSize: 8, color: "#4a4535", marginTop: 2 }}>Sampled {vr}×{vc} of {rows}×{cols}</div>}
    </div>
  );
}

/* ─── Mini Matrix in parent panel ─── */
function MiniMat({ data, rows, cols, size = 64 }) {
  const mx = Math.max(0.01, ...data.map(Math.abs));
  const sr = rows > 14 ? Math.ceil(rows / 14) : 1;
  const sc = cols > 14 ? Math.ceil(cols / 14) : 1;
  const vr = Math.ceil(rows / sr), vc = Math.ceil(cols / sc);
  const cw = Math.max(2, Math.floor(size / Math.max(vr, vc)));
  return (
    <svg width={vc * cw} height={vr * cw} style={{ borderRadius: 2, display: "block" }}>
      {Array.from({ length: vr }).map((_, i) =>
        Array.from({ length: vc }).map((_, j) => {
          const val = data[(i * sr) * cols + j * sc] || 0;
          return <rect key={`${i}-${j}`} x={j*cw} y={i*cw} width={cw-(cw>3?.4:0)} height={cw-(cw>3?.4:0)}
            fill={valColor(val, mx)} />;
        })
      )}
    </svg>
  );
}

/* ═══ MAIN ═══ */
export default function MatrixDAG() {
  const [seq, setSeq] = useState(6);
  const [dModel, setDModel] = useState(8);
  const [nHeads, setNHeads] = useState(2);
  const [ffMult, setFFMult] = useState(4);
  const [nLayers, setNLayers] = useState(2);
  const [seed, setSeed] = useState(42);
  const [sel, setSel] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(700);
  const playRef = useRef();
  const gRef = useRef();

  const nodes = useMemo(() => buildGraph(seq, dModel, nHeads, ffMult, nLayers, seed),
    [seq, dModel, nHeads, ffMult, nLayers, seed]);
  const layout = useMemo(() => layoutNodes(nodes, nHeads, nLayers), [nodes, nHeads, nLayers]);

  useEffect(() => { setSel(0); }, [seq, dModel, nHeads, ffMult, nLayers, seed]);

  useEffect(() => {
    if (!playing) { clearInterval(playRef.current); return; }
    let i = sel;
    playRef.current = setInterval(() => {
      i++;
      if (i >= nodes.length) { setPlaying(false); return; }
      setSel(i);
    }, speed);
    return () => clearInterval(playRef.current);
  }, [playing, speed, nodes.length]);

  useEffect(() => {
    if (gRef.current && layout[sel]) {
      gRef.current.scrollLeft = Math.max(0, layout[sel].x - 280);
    }
  }, [sel, layout]);

  const nd = nodes[sel];
  const stats = useMemo(() => nd ? matStats(nd.data) : null, [nd]);

  const NW = 96, NH = 56;
  const totalW = Math.max(900, ...layout.map(p => p.x)) + 140;
  const totalH = Math.max(350, ...layout.map(p => p.y)) + 100;

  return (
    <div style={{
      "--mono": "'IBM Plex Mono', monospace", "--disp": "'Playfair Display', Georgia, serif",
      minHeight: "100vh", background: "#0c0a07", color: "#c8b890", fontFamily: "var(--mono)",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Playfair+Display:wght@700;900&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:6px;height:6px}
        ::-webkit-scrollbar-thumb{background:rgba(180,150,80,0.25);border-radius:3px}
        @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:0.6}50%{opacity:1}}
        input[type=range]{accent-color:#d4a840}
      `}</style>

      {/* HEADER */}
      <header style={{ padding: "16px 24px 10px", borderBottom: "1px solid rgba(180,150,80,0.1)",
        background: "linear-gradient(180deg, rgba(20,16,8,1), #0c0a07)" }}>
        <h1 style={{ fontFamily: "var(--disp)", fontSize: 26, fontWeight: 900, color: "#d4a840" }}>
          Matrix Flow Graph
        </h1>
        <p style={{ fontSize: 10, color: "#5a5040", marginTop: 3 }}>
          Q, K, V branch <strong style={{ color: "#5a9fd0" }}>in parallel</strong> from X —
          Q and K converge in the dot-product, then attention weights multiply V.
          Click any node to inspect its matrix.
        </p>
      </header>

      {/* CONTROLS */}
      <div style={{ padding: "8px 24px", borderBottom: "1px solid rgba(180,150,80,0.1)",
        display: "flex", gap: 14, flexWrap: "wrap", alignItems: "end" }}>
        {[
          { l:"Seq", v:seq, s:setSeq, mn:2, mx:12 },
          { l:"d_model", v:dModel, s:setDModel, mn:4, mx:32, st:4 },
          { l:"Heads", v:nHeads, s:setNHeads, mn:1, mx:4 },
          { l:"FF×", v:ffMult, s:setFFMult, mn:2, mx:4 },
          { l:"Layers", v:nLayers, s:setNLayers, mn:1, mx:3 },
        ].map(({l,v,s,mn,mx,st})=>(
          <div key={l} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <label style={{ fontSize: 8, color: "#4a4535" }}>{l}</label>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <input type="range" min={mn} max={mx} step={st||1} value={v}
                onChange={e=>s(parseInt(e.target.value))} style={{ width: 56 }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: "#d4a840", minWidth: 18 }}>{v}</span>
            </div>
          </div>
        ))}
        <div style={{ display: "flex", gap: 4, marginLeft: 6 }}>
          <button onClick={() => { setSel(0); setPlaying(true); }}
            style={{ padding: "4px 12px", borderRadius: 3, fontSize: 10, cursor: "pointer",
              background: playing ? "#d4a840" : "transparent", border: "1px solid #d4a840",
              color: playing ? "#0c0a07" : "#d4a840", fontFamily: "var(--mono)", fontWeight: 600 }}>
            {playing ? "⏸" : "▶"}</button>
          <button onClick={() => setSel(s => Math.max(0, s-1))}
            style={{ padding: "4px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer",
              background: "transparent", border: "1px solid rgba(180,150,80,0.2)", color: "#8a7a55", fontFamily: "var(--mono)" }}>←</button>
          <button onClick={() => setSel(s => Math.min(nodes.length-1, s+1))}
            style={{ padding: "4px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer",
              background: "transparent", border: "1px solid rgba(180,150,80,0.2)", color: "#8a7a55", fontFamily: "var(--mono)" }}>→</button>
          <span style={{ fontSize: 9, color: "#5a5040", alignSelf: "center" }}>{sel+1}/{nodes.length}</span>
        </div>
      </div>

      {/* GRAPH */}
      <div ref={gRef} style={{
        overflowX: "auto", overflowY: "auto",
        maxHeight: Math.min(totalH + 30, 480),
        borderBottom: "1px solid rgba(180,150,80,0.1)",
        background: "rgba(0,0,0,0.25)",
      }}>
        <svg width={totalW} height={totalH}>
          {/* Layer + head backgrounds */}
          {Array.from({ length: nLayers }).map((_, L) => (
            <g key={L}>
              <rect x={140 + L * 730} y={8} width={710} height={totalH - 16}
                fill="rgba(180,150,80,0.015)" stroke="rgba(180,150,80,0.05)" strokeWidth={1} rx={6} />
              <text x={148 + L * 730} y={24} fontSize={10} fill="#3a3525"
                fontFamily="var(--disp)" fontWeight={700}>Layer {L + 1}</text>
              {Array.from({ length: nHeads }).map((_, h) => (
                <g key={h}>
                  <rect x={155 + L * 730} y={22 + h * 145} width={430} height={135}
                    fill="rgba(90,159,208,0.02)" stroke="rgba(90,159,208,0.05)"
                    strokeWidth={0.5} rx={4} strokeDasharray="4 2" />
                  <text x={160 + L * 730} y={34 + h * 145} fontSize={8} fill="rgba(90,159,208,0.4)"
                    fontFamily="var(--mono)">Head {h + 1}</text>
                </g>
              ))}
            </g>
          ))}

          {/* Edges */}
          {nodes.map((n, i) => n.parents.map(pid => {
            const f = layout[pid], t = layout[i];
            const fx = f.x + NW/2, fy = f.y + NH/2, tx = t.x + NW/2, ty = t.y + NH/2;
            const active = i === sel || pid === sel;
            const isInput = i === sel;
            const mx = (fx + tx) / 2;
            return (
              <path key={`${pid}-${i}`}
                d={`M${fx},${fy} C${mx},${fy} ${mx},${ty} ${tx},${ty}`}
                fill="none"
                stroke={active ? (PC[n.phase] || "#666") : "rgba(180,150,80,0.08)"}
                strokeWidth={active ? 2.5 : 0.7}
                opacity={active ? 0.85 : 0.35}
              />
            );
          }))}

          {/* Nodes */}
          {nodes.map((n, i) => {
            const p = layout[i];
            const c = PC[n.phase] || "#666";
            const isSel = i === sel;
            const isPar = nd && nd.parents.includes(i);
            return (
              <g key={i} onClick={() => setSel(i)} style={{ cursor: "pointer" }}
                transform={`translate(${p.x},${p.y})`}>
                {isSel && <rect x={-4} y={-4} width={NW+8} height={NH+8} fill="none"
                  stroke={c} strokeWidth={2.5} rx={7} opacity={0.6}
                  style={{ animation: "pulse 1.5s ease infinite" }} />}
                <rect width={NW} height={NH} rx={4}
                  fill={isSel ? c : isPar ? `${c}22` : "rgba(12,10,7,0.92)"}
                  stroke={isSel ? c : isPar ? c : "rgba(180,150,80,0.12)"}
                  strokeWidth={isSel ? 2 : isPar ? 1.5 : 0.5} />
                <text x={NW/2} y={16} textAnchor="middle" fontSize={9} fontWeight={600}
                  fill={isSel ? "#0c0a07" : c} fontFamily="var(--mono)">{n.label}</text>
                <text x={NW/2} y={32} textAnchor="middle" fontSize={13} fontWeight={700}
                  fill={isSel ? "#0c0a07" : "#c8b890"} fontFamily="var(--mono)">
                  {n.rows}×{n.cols}</text>
                <text x={NW/2} y={46} textAnchor="middle" fontSize={7}
                  fill={isSel ? "rgba(12,10,7,0.5)" : "#4a4535"} fontFamily="var(--mono)">
                  {(n.rows*n.cols).toLocaleString()} vals</text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* DETAIL */}
      {nd && (
        <div style={{ padding: "14px 24px", animation: "fadeIn 0.2s ease",
          display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }} key={sel}>

          {/* Matrix */}
          <div style={{ flex: "1 1 460px", minWidth: 280 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: PC[nd.phase] || "#666" }} />
              <span style={{ fontSize: 16, fontFamily: "var(--disp)", fontWeight: 700, color: "#d4a840" }}>
                {nd.label}
              </span>
              {nd.layer >= 0 && <span style={{ fontSize: 10, color: "#5a5040" }}>
                Layer {nd.layer+1}{nd.head >= 0 ? ` · Head ${nd.head+1}` : ""}</span>}
            </div>
            <div style={{ fontSize: 10, color: "#6a6050", marginBottom: 4 }}>{nd.desc}</div>

            {nd.dimOp && (
              <div style={{ display: "inline-block", padding: "3px 10px", borderRadius: 3, marginBottom: 8,
                background: "rgba(180,150,80,0.06)", border: "1px solid rgba(180,150,80,0.12)",
                fontSize: 12, fontWeight: 600, color: "#d4a840" }}>
                {nd.dimOp}
              </div>
            )}

            <div style={{ display: "inline-flex", gap: 8, alignItems: "center",
              padding: "3px 10px", borderRadius: 3, marginBottom: 8, marginLeft: 6,
              background: "rgba(180,150,80,0.04)", border: "1px solid rgba(180,150,80,0.08)" }}>
              <span style={{ fontSize: 18, fontFamily: "var(--disp)", fontWeight: 700, color: "#d4a840" }}>
                {nd.rows}×{nd.cols}</span>
              <span style={{ fontSize: 9, color: "#5a5040" }}>= {(nd.rows*nd.cols).toLocaleString()} values</span>
            </div>

            <div style={{ marginBottom: 8 }}>
              <FullMatrix data={nd.data} rows={nd.rows} cols={nd.cols} />
            </div>

            {stats && (
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 10 }}>
                {[
                  { k:"min", v:stats.min, c:"#4a8ad0" }, { k:"max", v:stats.max, c:"#d0884a" },
                  { k:"mean", v:stats.mean, c:"#7ab060" }, { k:"std", v:stats.std, c:"#b07a60" },
                ].map(({k,v,c})=>(
                  <span key={k}><span style={{color:"#5a5040"}}>{k}</span>{" "}
                    <span style={{color:c}}>{v.toFixed(4)}</span></span>
                ))}
              </div>
            )}
          </div>

          {/* Parents */}
          {nd.parents.length > 0 && (
            <div style={{ flex: "0 0 220px" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "#6a6050", marginBottom: 8,
                letterSpacing: "0.08em", textTransform: "uppercase" }}>
                Inputs ({nd.parents.length})
              </div>
              {nd.parents.map(pid => {
                const p = nodes[pid];
                return (
                  <div key={pid} onClick={() => setSel(pid)} style={{
                    padding: 6, marginBottom: 6, borderRadius: 4, cursor: "pointer",
                    background: "rgba(180,150,80,0.03)", border: "1px solid rgba(180,150,80,0.08)",
                    transition: "background 0.15s",
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: PC[p.phase], marginBottom: 3 }}>
                      {p.label} <span style={{ color: "#5a5040", fontWeight: 400 }}>{p.rows}×{p.cols}</span>
                    </div>
                    <MiniMat data={p.data} rows={p.rows} cols={p.cols} />
                  </div>
                );
              })}
              <div style={{ marginTop: 10, padding: 8, borderRadius: 4,
                background: "rgba(90,159,208,0.04)", border: "1px solid rgba(90,159,208,0.1)",
                fontSize: 9, color: "#5a8aaa", lineHeight: 1.6 }}>
                <strong>Parallel flow:</strong> Q, K, V are all computed from the same input X
                independently. Q·Kᵀ takes Q and K as inputs. Attn·V takes Softmax output and V.
              </div>
            </div>
          )}
        </div>
      )}

      <footer style={{ padding: "10px 24px", borderTop: "1px solid rgba(180,150,80,0.08)",
        fontSize: 9, color: "#3a3525", textAlign: "center" }}>
        Real matrix multiplications · {nodes.length} operations · seq={seq} d_model={dModel} heads={nHeads} d_ff={dModel*ffMult} layers={nLayers}
      </footer>
    </div>
  );
}
