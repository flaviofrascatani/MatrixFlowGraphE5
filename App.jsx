import { useState, useMemo, useEffect } from "react";

/* ═══ MATH ═══ */
function rng(a){return()=>{a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296}}
function matMul(A,B,rA,cA,cB){const C=new Array(rA*cB).fill(0);for(let i=0;i<rA;i++)for(let j=0;j<cB;j++)for(let k=0;k<cA;k++)C[i*cB+j]+=A[i*cA+k]*B[k*cB+j];return C}
function addMat(A,B){return A.map((v,i)=>v+(B[i]||0))}
function softmaxRows(f,r,c){const o=[...f];for(let i=0;i<r;i++){let mx=-Infinity;for(let j=0;j<c;j++)mx=Math.max(mx,o[i*c+j]);let s=0;for(let j=0;j<c;j++){o[i*c+j]=Math.exp(o[i*c+j]-mx);s+=o[i*c+j]}for(let j=0;j<c;j++)o[i*c+j]/=s}return o}
function layerNorm(f,r,c){const o=[...f];for(let i=0;i<r;i++){let m=0;for(let j=0;j<c;j++)m+=o[i*c+j];m/=c;let v=0;for(let j=0;j<c;j++)v+=(o[i*c+j]-m)**2;v=Math.sqrt(v/c+1e-5);for(let j=0;j<c;j++)o[i*c+j]=(o[i*c+j]-m)/v}return o}
function gelu(x){return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x**3)))}
function transpose(f,r,c){const o=new Array(r*c);for(let i=0;i<r;i++)for(let j=0;j<c;j++)o[j*r+i]=f[i*c+j];return o}
function matStats(f){if(!f||!f.length)return{min:0,max:0,mean:0,std:0};const mn=Math.min(...f),mx=Math.max(...f),avg=f.reduce((a,b)=>a+b,0)/f.length;const std=Math.sqrt(f.reduce((a,b)=>a+(b-avg)**2,0)/f.length);return{min:mn,max:mx,mean:avg,std}}

/* ═══ SIMULATION ═══ */
function simulate(seq,dModel,nHeads,ffMult,nLayers,seed){
  const n=seq,dH=Math.floor(dModel/nHeads),dFF=dModel*ffMult,r0=rng(seed);
  let X=new Array(n*dModel).fill(0).map(()=>(r0()-0.5)*2);
  const input={id:"input",label:"Input X",dim:[n,dModel],data:X,formula:`Random init [${n}×${dModel}]`,color:"#777"};
  const layers=[];

  for(let L=0;L<nLayers;L++){
    const Xin=X;
    const heads=[];
    const allCtx=[];
    
    for(let h=0;h<nHeads;h++){
      const hr=rng(seed*1000+L*137+h*31);
      const Wq=new Array(dModel*dH).fill(0).map(()=>(hr()-0.5)*(2/Math.sqrt(dModel)));
      const Wk=new Array(dModel*dH).fill(0).map(()=>(hr()-0.5)*(2/Math.sqrt(dModel)));
      const Wv=new Array(dModel*dH).fill(0).map(()=>(hr()-0.5)*(2/Math.sqrt(dModel)));
      const Q=matMul(X,Wq,n,dModel,dH);
      const K=matMul(X,Wk,n,dModel,dH);
      const V=matMul(X,Wv,n,dModel,dH);
      const Kt=transpose(K,n,dH);
      let scores=matMul(Q,Kt,n,dH,n);
      const sc=Math.sqrt(dH);scores=scores.map(v=>v/sc);
      const attn=softmaxRows(scores,n,n);
      const ctx=matMul(attn,V,n,n,dH);
      allCtx.push(ctx);
      
      heads.push({
        Q:{id:`L${L}H${h}Q`,label:"Q",dim:[n,dH],data:Q,formula:`X[${n}×${dModel}] · Wq[${dModel}×${dH}] = [${n}×${dH}]`,color:"#5a9fd0"},
        K:{id:`L${L}H${h}K`,label:"K",dim:[n,dH],data:K,formula:`X[${n}×${dModel}] · Wk[${dModel}×${dH}] = [${n}×${dH}]`,color:"#50b880"},
        V:{id:`L${L}H${h}V`,label:"V",dim:[n,dH],data:V,formula:`X[${n}×${dModel}] · Wv[${dModel}×${dH}] = [${n}×${dH}]`,color:"#c08040"},
        scores:{id:`L${L}H${h}sc`,label:"Q·Kᵀ/√d",dim:[n,n],data:scores,formula:`Q[${n}×${dH}] · Kᵀ[${dH}×${n}] / √${dH} = [${n}×${n}]`,color:"#d0a040"},
        attn:{id:`L${L}H${h}attn`,label:"Softmax",dim:[n,n],data:attn,formula:`softmax([${n}×${n}]) = [${n}×${n}]`,color:"#e08830"},
        ctx:{id:`L${L}H${h}ctx`,label:"Attn·V",dim:[n,dH],data:ctx,formula:`Attn[${n}×${n}] · V[${n}×${dH}] = [${n}×${dH}]`,color:"#b060a0"},
      });
    }

    const concatDim=dH*nHeads;
    const concatData=new Array(n*concatDim).fill(0);
    for(let i=0;i<n;i++)for(let h=0;h<nHeads;h++)for(let d=0;d<dH;d++)
      concatData[i*concatDim+h*dH+d]=allCtx[h][i*dH+d];
    const hr2=rng(seed*2000+L*97);
    const Wo=new Array(concatDim*dModel).fill(0).map(()=>(hr2()-0.5)*(2/Math.sqrt(concatDim)));
    const projected=matMul(concatData,Wo,n,concatDim,dModel);
    const normed=layerNorm(addMat(Xin,projected),n,dModel);
    const fr=rng(seed*500+L*251);
    const W1=new Array(dModel*dFF).fill(0).map(()=>(fr()-0.5)*(2/Math.sqrt(dModel)));
    const B1=new Array(dFF).fill(0).map(()=>(fr()-0.5)*0.1);
    const W2=new Array(dFF*dModel).fill(0).map(()=>(fr()-0.5)*(2/Math.sqrt(dFF)));
    let hidden=matMul(normed,W1,n,dModel,dFF);
    for(let i=0;i<n;i++)for(let j=0;j<dFF;j++)hidden[i*dFF+j]+=B1[j];
    const activated=hidden.map(gelu);
    const down=matMul(activated,W2,n,dFF,dModel);
    const out=layerNorm(addMat(normed,down),n,dModel);

    layers.push({
      layer:L, heads,
      concat:{id:`L${L}concat`,label:"Concat",dim:[n,concatDim],data:concatData,formula:`${nHeads} heads × [${n}×${dH}] = [${n}×${concatDim}]`,color:"#8070c0"},
      postMHA:{id:`L${L}post`,label:"Proj+Res+LN",dim:[n,dModel],data:normed,formula:`[${n}×${concatDim}]·Wo[${concatDim}×${dModel}]+Res+LN = [${n}×${dModel}]`,color:"#60b0a0"},
      ffnUp:{id:`L${L}up`,label:"FFN Up",dim:[n,dFF],data:hidden,formula:`[${n}×${dModel}]·W1[${dModel}×${dFF}]+b = [${n}×${dFF}]`,color:"#d4a840"},
      ffnGelu:{id:`L${L}gelu`,label:"GELU",dim:[n,dFF],data:activated,formula:`GELU([${n}×${dFF}]) = [${n}×${dFF}]`,color:"#e07040"},
      ffnDown:{id:`L${L}down`,label:"FFN Down+LN",dim:[n,dModel],data:out,formula:`[${n}×${dFF}]·W2[${dFF}×${dModel}]+Res+LN = [${n}×${dModel}]`,color:"#80b060"},
    });
    X=out;
  }
  return{input,layers,p:{n,dModel,dH,dFF,nHeads}};
}

/* ═══ MINI MATRIX ═══ */
function MiniMat({data,rows,cols,w=120,h=80,isAttn}){
  const mx=Math.max(0.01,...data.map(Math.abs));
  const cw=Math.max(2,Math.floor(w/cols));
  const ch=Math.max(2,Math.floor(h/rows));
  return(
    <svg width={Math.min(w,cols*cw)} height={Math.min(h,rows*ch)} style={{display:"block",borderRadius:3}}>
      {Array.from({length:Math.min(rows,Math.floor(h/ch))}).map((_,i)=>
        Array.from({length:Math.min(cols,Math.floor(w/cw))}).map((_,j)=>{
          const v=data[i*cols+j]||0;
          let fill;
          if(isAttn){const t=Math.min(v*1.3,1);fill=`rgb(${12+t*243|0},${10+t*190|0},${5+t*35|0})`;}
          else{const t=Math.max(-1,Math.min(1,v/mx));fill=t>=0?`rgb(${15+t*230|0},${12+t*110|0},${8+t*15|0})`:`rgb(${8-t*15|0},${15-t*80|0},${15-t*230|0})`;}
          return <rect key={`${i}-${j}`} x={j*cw} y={i*ch} width={cw-0.5} height={ch-0.5} fill={fill} rx={cw>4?1:0}/>;
        })
      )}
    </svg>
  );
}

/* ═══ BIG MATRIX ═══ */
function BigMat({node}){
  if(!node)return null;
  const{data,dim,label,formula,color}=node;
  const[rows,cols]=dim;
  const stats=matStats(data);
  const mx=Math.max(0.01,...data.map(Math.abs));
  const isAttn=label==="Softmax";
  const maxVis=1200;
  const skip=rows*cols>maxVis?Math.ceil(Math.sqrt(rows*cols/maxVis)):1;
  const vr=Math.ceil(rows/skip),vc=Math.ceil(cols/skip);
  const cw=Math.min(Math.max(4,Math.floor(600/vc)),28);
  const ch=Math.min(Math.max(4,Math.floor(400/vr)),28);
  const showNum=cw>=20&&ch>=16;

  return(
    <div style={{background:"rgba(0,0,0,0.3)",borderRadius:10,padding:20,border:`1px solid ${color}44`,
      boxShadow:`0 0 40px ${color}11`}}>
      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:6}}>
        <div style={{width:14,height:14,borderRadius:4,background:color}}/>
        <span style={{fontSize:22,fontWeight:900,fontFamily:"var(--disp)",color}}>{label}</span>
      </div>
      <div style={{fontSize:12,color:"#8a7a55",marginBottom:4,fontFamily:"var(--mono)"}}>{formula}</div>
      <div style={{display:"flex",alignItems:"baseline",gap:12,marginBottom:12}}>
        <span style={{fontSize:32,fontWeight:900,fontFamily:"var(--disp)",color:"#d4a840"}}>{rows} × {cols}</span>
        <span style={{fontSize:11,color:"#6a6050"}}>= {(rows*cols).toLocaleString()} values</span>
      </div>

      <svg width={vc*cw} height={vr*ch} style={{display:"block",borderRadius:4,marginBottom:12}}>
        {Array.from({length:vr}).map((_,ri)=>
          Array.from({length:vc}).map((_,ci)=>{
            const v=data[(ri*skip)*cols+(ci*skip)]||0;
            let fill;
            if(isAttn){const t=Math.min(v*1.3,1);fill=`rgb(${12+t*243|0},${10+t*190|0},${5+t*35|0})`;}
            else{const t=Math.max(-1,Math.min(1,v/mx));fill=t>=0?`rgb(${15+t*230|0},${12+t*110|0},${8+t*15|0})`:`rgb(${8-t*15|0},${15-t*80|0},${15-t*230|0})`;}
            return(
              <g key={`${ri}-${ci}`}>
                <rect x={ci*cw} y={ri*ch} width={cw-0.6} height={ch-0.6} fill={fill} rx={1}/>
                {showNum&&<text x={ci*cw+cw/2} y={ri*ch+ch/2+3} textAnchor="middle"
                  fontSize={Math.min(7,cw-3)} fill="rgba(255,255,255,0.7)" fontFamily="var(--mono)">
                  {v.toFixed(2)}</text>}
              </g>
            );
          })
        )}
      </svg>
      {skip>1&&<div style={{fontSize:9,color:"#4a4535",marginBottom:6}}>Sampled {vr}×{vc} of {rows}×{cols}</div>}

      <div style={{display:"flex",gap:18}}>
        {[["min",stats.min,"#5a9fd0"],["max",stats.max,"#d0884a"],["μ",stats.mean,"#7ab060"],["σ",stats.std,"#b07a60"]].map(([k,v,c])=>(
          <span key={k} style={{fontSize:11,fontFamily:"var(--mono)"}}>
            <span style={{color:"#5a5040"}}>{k}</span> <span style={{color:c}}>{v.toFixed(4)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ═══ NODE PILL ═══ */
function Pill({node,selected,onClick,showMatrix}){
  const{label,dim,color}=node;
  const isAttn=label==="Softmax";
  return(
    <div onClick={onClick} style={{
      display:"flex",flexDirection:"column",alignItems:"center",gap:4,
      padding:"8px 10px",borderRadius:8,cursor:"pointer",
      background:selected?`${color}18`:"rgba(255,255,255,0.02)",
      border:selected?`2px solid ${color}`:"1px solid rgba(255,255,255,0.06)",
      boxShadow:selected?`0 0 16px ${color}22`:"none",
      transition:"all 0.15s",minWidth:90,
    }}>
      <div style={{display:"flex",alignItems:"center",gap:5}}>
        <div style={{width:8,height:8,borderRadius:3,background:color,flexShrink:0}}/>
        <span style={{fontSize:11,fontWeight:700,color,fontFamily:"var(--mono)",whiteSpace:"nowrap"}}>{label}</span>
      </div>
      <span style={{fontSize:16,fontWeight:800,color:"#d4a840",fontFamily:"var(--disp)"}}>{dim[0]}×{dim[1]}</span>
      {showMatrix&&<MiniMat data={node.data} rows={dim[0]} cols={dim[1]} w={80} h={50} isAttn={isAttn}/>}
    </div>
  );
}

/* ═══ ARROW ═══ */
function Arrow({text,down,color="#3a3525"}){
  if(down) return(
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:0,padding:"2px 0"}}>
      <div style={{width:1,height:12,background:color}}/>
      {text&&<span style={{fontSize:8,color:"#5a5040",fontFamily:"var(--mono)"}}>{text}</span>}
      <div style={{width:0,height:0,borderLeft:"4px solid transparent",borderRight:"4px solid transparent",borderTop:`6px solid ${color}`}}/>
    </div>
  );
  return(
    <div style={{display:"flex",alignItems:"center",gap:2,padding:"0 3px"}}>
      <div style={{width:14,height:1,background:color}}/>
      {text&&<span style={{fontSize:8,color:"#5a5040",fontFamily:"var(--mono)",whiteSpace:"nowrap"}}>{text}</span>}
      <div style={{width:0,height:0,borderTop:"4px solid transparent",borderBottom:"4px solid transparent",borderLeft:`6px solid ${color}`}}/>
    </div>
  );
}

/* ═══ MAIN ═══ */
export default function App(){
  const [seq,setSeq]=useState(6);
  const [dModel,setDModel]=useState(8);
  const [nHeads,setNHeads]=useState(2);
  const [ffMult,setFFMult]=useState(4);
  const [nLayers,setNLayers]=useState(2);
  const [seed,setSeed]=useState(42);
  const [selLayer,setSelLayer]=useState(0);
  const [selId,setSelId]=useState(null);
  const [showMini,setShowMini]=useState(true);

  const data=useMemo(()=>simulate(seq,dModel,nHeads,ffMult,nLayers,seed),[seq,dModel,nHeads,ffMult,nLayers,seed]);
  useEffect(()=>{setSelLayer(0);setSelId(null)},[seq,dModel,nHeads,ffMult,nLayers,seed]);

  const ly=data.layers[selLayer];
  // Collect all nodes for lookup
  const allNodes=[];
  if(ly){
    ly.heads.forEach(h=>{allNodes.push(h.Q,h.K,h.V,h.scores,h.attn,h.ctx)});
    allNodes.push(ly.concat,ly.postMHA,ly.ffnUp,ly.ffnGelu,ly.ffnDown);
  }
  allNodes.push(data.input);
  const selNode=selId?allNodes.find(n=>n.id===selId):null;

  const p=data.p;

  return(
    <div style={{minHeight:"100vh",background:"#0b0907",color:"#c8b890",fontFamily:"'IBM Plex Mono',monospace"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Playfair+Display:wght@700;900&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-thumb{background:rgba(180,150,80,0.2);border-radius:3px}
        input[type=range]{accent-color:#d4a840}
        @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
      `}</style>

      {/* ═══ HEADER ═══ */}
      <header style={{padding:"20px 24px 14px",borderBottom:"1px solid rgba(180,150,80,0.1)",
        background:"linear-gradient(180deg,#13100a,#0b0907)"}}>
        <h1 style={{fontFamily:"'Playfair Display',serif",fontSize:28,fontWeight:900,color:"#d4a840"}}>
          Matrix Multiplication Graph
        </h1>
        <p style={{fontSize:11,color:"#6a6050",marginTop:4,maxWidth:650,lineHeight:1.6}}>
          X gets projected <em>in parallel</em> into Q, K, V (three branches) — Q and K merge into scores — softmax — multiplied by V — concat heads — FFN up/down. Click any node to inspect its full matrix.
        </p>
      </header>

      {/* ═══ CONFIG ═══ */}
      <div style={{padding:"10px 24px",borderBottom:"1px solid rgba(180,150,80,0.1)",
        display:"flex",gap:18,flexWrap:"wrap",alignItems:"end"}}>
        {[
          {l:"Seq",v:seq,s:setSeq,mn:2,mx:12},
          {l:"d_model",v:dModel,s:setDModel,mn:4,mx:32,st:4},
          {l:"Heads",v:nHeads,s:setNHeads,mn:1,mx:4},
          {l:"FF mult",v:ffMult,s:setFFMult,mn:2,mx:4},
          {l:"Layers",v:nLayers,s:setNLayers,mn:1,mx:4},
        ].map(({l,v,s,mn,mx,st})=>(
          <div key={l}>
            <div style={{fontSize:8,color:"#5a5040",letterSpacing:"0.08em",marginBottom:2}}>{l}</div>
            <div style={{display:"flex",alignItems:"center",gap:5}}>
              <input type="range" min={mn} max={mx} step={st||1} value={v} onChange={e=>s(+e.target.value)} style={{width:72}}/>
              <span style={{fontSize:13,fontWeight:700,color:"#d4a840",minWidth:22}}>{v}</span>
            </div>
          </div>
        ))}
        <div>
          <div style={{fontSize:8,color:"#5a5040",marginBottom:2}}>Seed</div>
          <div style={{display:"flex",gap:4}}>
            <input type="range" min={1} max={999} value={seed} onChange={e=>setSeed(+e.target.value)} style={{width:60}}/>
            <span style={{fontSize:13,fontWeight:700,color:"#d4a840"}}>{seed}</span>
            <button onClick={()=>setSeed(1+Math.random()*998|0)} style={{
              fontSize:9,padding:"2px 8px",borderRadius:3,cursor:"pointer",
              background:"transparent",border:"1px solid rgba(180,150,80,0.2)",color:"#8a7a55",fontFamily:"inherit",
            }}>🎲</button>
          </div>
        </div>
        <label style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:"#6a6050",cursor:"pointer"}}>
          <input type="checkbox" checked={showMini} onChange={e=>setShowMini(e.target.checked)}/>
          Preview matrices
        </label>
      </div>

      {/* ═══ LAYER TABS ═══ */}
      <div style={{padding:"10px 24px",borderBottom:"1px solid rgba(180,150,80,0.1)",
        display:"flex",gap:6,alignItems:"center"}}>
        <span style={{fontSize:9,color:"#5a5040",letterSpacing:"0.1em",marginRight:4}}>LAYER</span>
        {data.layers.map((_,i)=>(
          <button key={i} onClick={()=>{setSelLayer(i);setSelId(null)}} style={{
            width:36,height:30,borderRadius:4,fontSize:12,cursor:"pointer",fontWeight:700,
            fontFamily:"inherit",display:"flex",alignItems:"center",justifyContent:"center",
            background:selLayer===i?"#d4a840":"rgba(180,150,80,0.05)",
            color:selLayer===i?"#0b0907":"#5a5040",
            border:selLayer===i?"none":"1px solid rgba(180,150,80,0.12)",
          }}>{i+1}</button>
        ))}
      </div>

      {/* ═══ FLOW GRAPH ═══ */}
      {ly&&(
        <div style={{padding:"20px 24px",animation:"fadeIn 0.3s ease"}} key={selLayer}>

          {/* ── INPUT ── */}
          <div style={{display:"flex",flexDirection:"column",alignItems:"center",marginBottom:4}}>
            <Pill node={selLayer===0?data.input:data.layers[selLayer-1].ffnDown}
              selected={false} onClick={()=>{}} showMatrix={showMini}/>
            <Arrow down text="input to layer" color="#555"/>
          </div>

          {/* ── PARALLEL Q / K / V PER HEAD ── */}
          <div style={{
            background:"rgba(90,160,210,0.03)",border:"1px solid rgba(90,160,210,0.1)",
            borderRadius:12,padding:"16px 12px",marginBottom:4,
          }}>
            <div style={{fontSize:11,fontWeight:700,color:"#8a7a55",marginBottom:12,textAlign:"center",
              fontFamily:"'Playfair Display',serif",letterSpacing:"0.05em"}}>
              Multi-Head Attention — Layer {selLayer+1} — {p.nHeads} heads × d_head={p.dH}
            </div>

            {ly.heads.map((head,h)=>(
              <div key={h} style={{
                marginBottom:h<ly.heads.length-1?16:0,
                background:"rgba(255,255,255,0.01)",borderRadius:8,padding:12,
                border:"1px solid rgba(255,255,255,0.04)",
              }}>
                <div style={{fontSize:10,fontWeight:600,color:"#7a6a50",marginBottom:10}}>Head {h+1}</div>

                {/* Row 1: Q, K, V in parallel from X */}
                <div style={{display:"flex",justifyContent:"center",gap:16,flexWrap:"wrap",marginBottom:4}}>
                  <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
                    <Pill node={head.Q} selected={selId===head.Q.id} onClick={()=>setSelId(head.Q.id)} showMatrix={showMini}/>
                    <div style={{fontSize:7,color:"#4a4535",marginTop:2}}>X · Wq</div>
                  </div>
                  <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
                    <Pill node={head.K} selected={selId===head.K.id} onClick={()=>setSelId(head.K.id)} showMatrix={showMini}/>
                    <div style={{fontSize:7,color:"#4a4535",marginTop:2}}>X · Wk</div>
                  </div>
                  <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
                    <Pill node={head.V} selected={selId===head.V.id} onClick={()=>setSelId(head.V.id)} showMatrix={showMini}/>
                    <div style={{fontSize:7,color:"#4a4535",marginTop:2}}>X · Wv</div>
                  </div>
                </div>

                {/* Merge arrows */}
                <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:6,margin:"6px 0",position:"relative"}}>
                  {/* Q ↘ and K ↗ converge to scores */}
                  <svg width={260} height={30} style={{display:"block"}}>
                    {/* Q arrow */}
                    <line x1={55} y1={2} x2={110} y2={26} stroke="#5a9fd0" strokeWidth={1.2} opacity={0.5}/>
                    <text x={70} y={12} fontSize={7} fill="#5a9fd0" fontFamily="inherit">Q</text>
                    {/* K arrow */}
                    <line x1={130} y1={2} x2={130} y2={26} stroke="#50b880" strokeWidth={1.2} opacity={0.5}/>
                    <text x={136} y={12} fontSize={7} fill="#50b880" fontFamily="inherit">Kᵀ</text>
                    {/* V bypass arrow */}
                    <line x1={205} y1={2} x2={205} y2={26} stroke="#c08040" strokeWidth={1} opacity={0.3} strokeDasharray="3 2"/>
                    <text x={212} y={16} fontSize={7} fill="#c08040" fontFamily="inherit">V bypasses →</text>
                  </svg>
                </div>

                {/* Row 2: Scores → Softmax → Attn·V */}
                <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:6,flexWrap:"wrap"}}>
                  <Pill node={head.scores} selected={selId===head.scores.id} onClick={()=>setSelId(head.scores.id)} showMatrix={showMini}/>
                  <Arrow text="softmax"/>
                  <Pill node={head.attn} selected={selId===head.attn.id} onClick={()=>setSelId(head.attn.id)} showMatrix={showMini}/>
                  <Arrow text="× V"/>
                  <Pill node={head.ctx} selected={selId===head.ctx.id} onClick={()=>setSelId(head.ctx.id)} showMatrix={showMini}/>
                </div>
              </div>
            ))}
          </div>

          {/* ── CONCAT + PROJECT + FFN ── */}
          <Arrow down text="concat all heads" color="#8070c0"/>

          <div style={{
            background:"rgba(200,160,60,0.03)",border:"1px solid rgba(200,160,60,0.1)",
            borderRadius:12,padding:"16px 12px",
          }}>
            <div style={{fontSize:11,fontWeight:700,color:"#8a7a55",marginBottom:12,textAlign:"center",
              fontFamily:"'Playfair Display',serif",letterSpacing:"0.05em"}}>
              Projection + Feed-Forward Network
            </div>

            <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:8,flexWrap:"wrap"}}>
              <Pill node={ly.concat} selected={selId===ly.concat.id} onClick={()=>setSelId(ly.concat.id)} showMatrix={showMini}/>
              <Arrow text="× Wo + Res + LN"/>
              <Pill node={ly.postMHA} selected={selId===ly.postMHA.id} onClick={()=>setSelId(ly.postMHA.id)} showMatrix={showMini}/>
              <Arrow text="× W1 + b"/>
              <Pill node={ly.ffnUp} selected={selId===ly.ffnUp.id} onClick={()=>setSelId(ly.ffnUp.id)} showMatrix={showMini}/>
              <Arrow text="GELU"/>
              <Pill node={ly.ffnGelu} selected={selId===ly.ffnGelu.id} onClick={()=>setSelId(ly.ffnGelu.id)} showMatrix={showMini}/>
              <Arrow text="× W2 + Res + LN"/>
              <Pill node={ly.ffnDown} selected={selId===ly.ffnDown.id} onClick={()=>setSelId(ly.ffnDown.id)} showMatrix={showMini}/>
            </div>
          </div>

          {/* ── DIMENSION JOURNEY ── */}
          <div style={{marginTop:16,padding:"10px 14px",background:"rgba(212,168,64,0.04)",
            borderRadius:6,border:"1px solid rgba(180,150,80,0.08)"}}>
            <div style={{fontSize:9,color:"#5a5040",letterSpacing:"0.05em",marginBottom:6}}>
              DIMENSION JOURNEY — Layer {selLayer+1}
            </div>
            <div style={{display:"flex",gap:3,alignItems:"center",flexWrap:"wrap"}}>
              {[
                {l:"X",d:`${p.n}×${p.dModel}`,c:"#777"},
                {l:"Q,K,V",d:`${p.n}×${p.dH}`,c:"#5a9fd0"},
                {l:"Q·Kᵀ",d:`${p.n}×${p.n}`,c:"#d0a040"},
                {l:"Attn·V",d:`${p.n}×${p.dH}`,c:"#b060a0"},
                {l:"Concat",d:`${p.n}×${p.dH*p.nHeads}`,c:"#8070c0"},
                {l:"Proj",d:`${p.n}×${p.dModel}`,c:"#60b0a0"},
                {l:"FFN↑",d:`${p.n}×${p.dFF}`,c:"#d4a840"},
                {l:"GELU",d:`${p.n}×${p.dFF}`,c:"#e07040"},
                {l:"FFN↓",d:`${p.n}×${p.dModel}`,c:"#80b060"},
              ].map(({l,d,c},i,arr)=>(
                <div key={i} style={{display:"flex",alignItems:"center",gap:3}}>
                  <span style={{padding:"3px 7px",borderRadius:3,fontSize:10,fontWeight:600,
                    background:`${c}12`,border:`1px solid ${c}30`,color:c}}>
                    {l} <span style={{color:"#d4a840",fontWeight:800}}>{d}</span>
                  </span>
                  {i<arr.length-1&&<span style={{color:"#3a3020",fontSize:12}}>→</span>}
                </div>
              ))}
            </div>
          </div>

          {/* ── DETAIL PANEL ── */}
          {selNode&&(
            <div style={{marginTop:20,animation:"fadeIn 0.25s ease"}}>
              <BigMat node={selNode}/>
            </div>
          )}
        </div>
      )}

      <footer style={{padding:"12px 24px",borderTop:"1px solid rgba(180,150,80,0.08)",
        fontSize:9,color:"#3a3525",textAlign:"center"}}>
        Real matrix multiplications · seq={seq} d_model={dModel} d_head={p.dH} d_ff={p.dFF} · {nHeads} heads · {nLayers} layers · seed={seed}
      </footer>
    </div>
  );
}
