import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const $=s=>document.querySelector(s),R=['beranda','penerjemah','hasil','panduan','tentang','pengaturan','latih'];
let tm,snd=true,camOn=true,cur='beranda',stream=null,facing='user';
/* ---- MediaPipe (kode dari index-3.html) ---- */
const BASE=import.meta.env.BASE_URL;
const MODEL_URL=BASE+'models/hand_landmarker.task';
let MOD=null,FS=null,HL=null,HLimg=null,hlReady=false,hlErr=null,lastTs=0,lastVT=-1,lastTick=0;
/* ---- pengenalan urutan (kode dari index-3.html) ---- */
const W=10,SPAN=1200,MAXS=30,STRICT=.45,COOL=700,MFPS=15,LOST=600;
let buf=[],lastSeen=0,streak=0,cand=null,lastWord=null,lastT=0,words=[],idleT=0,lowUntil=0,released=true;
let SD={},RES={},rec=null,T=null,mjob=false,MF=[];
let SV=0,FC={v:-1,words:[],flats:[],n:[]},HLD='?';
let recBusy=false;
let DETMS=0,DFPS=0,dfc=0,dfT=0;
const SKEY='bisindo-seq-v1',UKEY='bisindo-uji-v1';
try{SD=JSON.parse(localStorage.getItem(SKEY)||'{}')||{}}catch(e){SD={}}
try{RES=JSON.parse(localStorage.getItem(UKEY)||'{}')||{}}catch(e){RES={}}
function saveSD(){SV++;try{localStorage.setItem(SKEY,JSON.stringify(SD))}catch(e){setLst('Penyimpanan penuh. Unduh data lalu hapus sebagian.')}}
function saveRES(){try{localStorage.setItem(UKEY,JSON.stringify(RES))}catch(e){}}
const nSamples=()=>Object.values(SD).reduce((a,x)=>a+x.length,0);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const setDt=m=>{$('#dt').textContent=m};
const setLst=m=>{$('#lst').textContent=m;const p=$('#pst');if(p)p.textContent=m};
const setMp=m=>{$('#mprog').textContent=m};
const tgtSel=()=>document.querySelector('.o2 select');
const tlang=()=>tgtSel().value==='English'?'en-US':'id-ID';
const EN={'tolong':'Help','terima kasih':'Thank you','maaf':'Sorry','selamat':'Congratulations','ya':'Yes','tidak':'No','toilet':'Toilet','makan':'Eat','halo':'Hello','minum':'Drink','air':'Water','sakit':'Sick','rumah':'House','sekolah':'School','nama':'Name','saya':'I'};
const shownWord=w=>tgtSel().value==='English'?(EN[w.toLowerCase()]||w):w;
function talk(text,lang){
  if(!window.speechSynthesis||!text)return;
  const u=new SpeechSynthesisUtterance(text);u.lang=lang;
  u.rate=document.querySelector('.o1 select').value==='Standar'?0.85:0.95;u.pitch=1;
  speechSynthesis.cancel();speechSynthesis.speak(u);
}
const speak=(t,l)=>{if(snd)talk(t,l)};

/* ===== fitur tangan & pengenalan (index-3.html) ===== */
function shape(l){
  const w=l[0],s=Math.hypot(l[9].x-w.x,l[9].y-w.y,l[9].z-w.z)||1,o=[];
  for(const p of l)o.push(+((p.x-w.x)/s).toFixed(2),+((p.y-w.y)/s).toFixed(2),+((p.z-w.z)/s).toFixed(2));
  o.push(+(w.x*2).toFixed(2),+(w.y*2).toFixed(2));return o;
}
function feat(hs){
  let s=[null,null];
  if(hs.length===1)s[0]=hs[0];else if(hs.length>1){const o=[...hs].sort((a,b)=>b[0].x-a[0].x);s=[o[0],o[1]]}
  return s.flatMap(h=>h?shape(h):new Array(65).fill(0));
}
function resample(b){const o=[],n=b.length;for(let i=0;i<W;i++)o.push(...b[Math.round(i*(n-1)/(W-1))].f);return o}
function dist(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}return Math.sqrt(s/a.length)}
const D=W*130;
function buildCache(){
  FC.words=[];FC.flats=[];FC.n=[];FC.thr=[];
  for(const w in SD){
    const a=(SD[w]||[]).filter(x=>x&&x.length===D);
    if(!a.length)continue;
    const f=new Float64Array(a.length*D);
    a.forEach((x,k)=>f.set(x,k*D));
    // ambang per kata: seberapa rapat contoh-contohnya sendiri (dari sebagian contoh agar ringan)
    const m=Math.min(a.length,8);let rad=0.2;
    if(m>1){
      let tot=0;
      for(let i=0;i<m;i++){
        let best=Infinity;
        for(let k=0;k<m;k++){if(k===i)continue;
          let s=0;const oi=i*D,ok=k*D;
          for(let j=0;j<D;j++){const d=f[oi+j]-f[ok+j];s+=d*d}
          const dd=Math.sqrt(s/D);if(dd<best)best=dd}
        tot+=best;
      }
      rad=tot/m;
    }
    FC.words.push(w);FC.flats.push(f);FC.n.push(a.length);
    FC.thr.push(Math.min(0.7,Math.max(0.3,1.6*rad)));
  }
  FC.v=SV;
}
function predict(x){
  if(FC.v!==SV)buildCache();
  if(!FC.words.length)return null;
  const X=x;
  const sc=[];
  for(let k=0;k<FC.words.length;k++){
    const f=FC.flats[k],n=FC.n[k];
    let b1=Infinity,b2=Infinity;
    for(let i=0;i<n;i++){
      const o=i*D;let s=0;
      const lim=b2===Infinity?Infinity:b2*b2*D;
      for(let j=0;j<D;j++){const d=X[j]-f[o+j];s+=d*d;if(s>lim)break}
      if(s>lim)continue;
      const dd=Math.sqrt(s/D);
      if(dd<b1){b2=b1;b1=dd}else if(dd<b2)b2=dd;
    }
    sc.push({word:FC.words[k],d:n>1?(b1+b2)/2:b1,thr:FC.thr[k]});
  }
  sc.sort((p,q)=>p.d-q.d);
  const ratio=sc.length>1?sc[0].d/sc[1].d:0;
  return{word:sc[0].word,d:sc[0].d,thr:sc[0].thr,conf:1-ratio,ok:sc[0].d<sc[0].thr&&ratio<.8};
}
const WINS=[SPAN,800];
function bestPred(b,tn){
  let best=null;
  for(const sp of WINS){
    const s=sp===SPAN?b:b.filter(e=>tn-e.t<=sp);
    if(s.length<6)continue;
    const x=resample(s),p=predict(x);if(!p)continue;
    const q=(p.ok?0:10)+p.d/p.thr;
    if(!best||q<best.q)best={p,x,q};
  }
  return best;
}
const STILL=.07;
let SCv=-1,SCl=[],SCvec=[],SCthr=[],candS=null,candSince=0;
const isStatic=w=>HURUF.includes(w)||ANGKA.includes(w);
function buildStatic(){
  SCl=[];SCvec=[];SCthr=[];
  for(const w in SD){
    if(!isStatic(w))continue;
    const vs=(SD[w]||[]).filter(x=>x&&x.length===D).map(x=>{
      const m=new Float64Array(130);
      for(let k=0;k<W;k++)for(let j=0;j<130;j++)m[j]+=x[k*130+j]/W;
      return m;
    });
    if(!vs.length)continue;
    const m=Math.min(vs.length,8);let rad=0.2;
    if(m>1){
      let tot=0;
      for(let i=0;i<m;i++){
        let best=Infinity;
        for(let k=0;k<m;k++){if(k===i)continue;
          let s2=0;for(let j=0;j<130;j++){const d=vs[i][j]-vs[k][j];s2+=d*d}
          const dd=Math.sqrt(s2/130);if(dd<best)best=dd}
        tot+=best;
      }
      rad=tot/m;
    }
    SCl.push(w);SCthr.push(Math.min(0.6,Math.max(0.25,1.6*rad)));
    SCvec.push(vs);
  }
  SCv=SV;
}
// satu frame tangan -> huruf/angka terdekat (vektor 130 = rata-rata pose)
function staticPredict(f){
  if(SCv!==SV)buildStatic();
  if(!SCl.length)return null;
  const sc=[];
  for(let k=0;k<SCl.length;k++){
    let best=Infinity;
    for(const v of SCvec[k]){
      let s2=0;for(let j=0;j<130;j++){const d=f[j]-v[j];s2+=d*d}
      const dd=Math.sqrt(s2/130);if(dd<best)best=dd;
    }
    sc.push({k,word:SCl[k],d:best});
  }
  sc.sort((p,q)=>p.d-q.d);
  const top=sc[0],ratio=sc.length>1?top.d/sc[1].d:0;
  return{word:top.word,d:top.d,ok:top.d<SCthr[top.k]&&ratio<.8};
}
function staticFrame(hs,now){
  if(!hs.length){candS=null;return}
  const nb=buf.length;
  if(nb>=4&&dist(buf[nb-1].f,buf[nb-4].f)>STILL){candS=null;return}
  const p=staticPredict(feat(hs).slice(0,130));
  if(!p||!p.ok){candS=null;return}
  if(candS!==p.word){candS=p.word;candSince=now;return}
  if(now-candSince>=100){candS=null;commit(p.word)}
}
function onHands(hs,now){
  if(hs.length){lastSeen=now;buf.push({t:now,f:feat(hs)});while(buf.length&&now-buf[0].t>SPAN)buf.shift()}
  else{released=true;if(now-lastSeen>LOST)buf=[]}
}
function commit(w){
  const now=Date.now();
  if(w===lastWord&&(!released||now-lastT<COOL))return false;
  lastWord=w;lastT=now;released=false;buf=[];cand=null;streak=0;
  words.push(w);if(words.length>6)words.shift();
  const sent=matchSentence();
  const txt=sent||words.map(shownWord).join(' ');
  $('#lt').textContent=txt;$('#rs').textContent=txt;
  speak(sent||shownWord(w),tlang());
  testHit(w);
  setDt('Gerakan terdeteksi');lowUntil=now+80;
  clearTimeout(idleT);idleT=setTimeout(()=>{words=[]},6000);
  return true;
}
function tick(){
  const t=Date.now();
  if(rec){
    if(t<rec.go){setLst('Siap membuat isyarat “'+rec.w+'” dalam '+Math.ceil((rec.go-t)/1000)+'…');return}
    if(t>rec.end){const n=(SD[rec.w]||[]).length;rec=null;saveSD();renderCounts();setLst('Rekaman selesai ('+n+' contoh)');return}
    if(buf.length>=8){const a=SD[rec.w]||(SD[rec.w]=[]);if(a.length<MAXS)a.push(resample(buf));setLst('Merekam “'+rec.w+'”: '+a.length+' contoh')}
    return;
  }
  const live=cur==='penerjemah';
  if(!live&&!(cur==='latih'))return;
  if(live&&t<lowUntil)return;
  if(buf.length<6){cand=null;streak=0;if(live)setDt(buf.length?'Mendeteksi gerakan…':'Arahkan tangan ke kamera');return}
  const bp=nSamples()?bestPred(buf,performance.now()):null;
  const p=bp&&bp.p,x=bp?bp.x:resample(buf);
  if(p&&p.ok){
    if(cand===p.word)streak++;else{cand=p.word;streak=1}
    if(live)setDt('“'+shownWord(p.word)+'”… tahan');
    if(streak>=2){
      if((live||T)&&commit(p.word))learn(p.word,x,p.d);
    }
  }else{
    cand=null;streak=0;
    pendingUnknown=x;
    if(live)setDt(nSamples()?'Gerakan belum dikenal':'Belum ada data latih');
  }
}
let saveT=0;
function learn(w,x,d){
  if(d>STRICT*0.6)return;
  SV++;
  const a=SD[w]||(SD[w]=[]);
  if(a.length>=MAXS)a.shift();
  a.push(x);clearTimeout(saveT);saveT=setTimeout(saveSD,3000);
}
/* ===== MediaPipe HandLandmarker (kode dari index-3.html) ===== */
async function initHL(){
  try{
    MOD={FilesetResolver,HandLandmarker};
    FS=await MOD.FilesetResolver.forVisionTasks(BASE+'wasm');
    const opt=d=>({baseOptions:{modelAssetPath:MODEL_URL,delegate:d},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5});
    const p=(async()=>{try{const h=await MOD.HandLandmarker.createFromOptions(FS,opt('GPU'));HLD='GPU';return h}catch(e){HLD='CPU';return await MOD.HandLandmarker.createFromOptions(FS,opt('CPU'))}})();
    HL=await Promise.race([p,new Promise((_,j)=>setTimeout(()=>j(new Error('waktu habis')),120000))]);
    hlReady=true;
  }catch(e){hlErr=(e&&e.message)||String(e)}
}
/* untuk foto & frame video dari berkas: mode IMAGE (tanpa pelacakan antarframe) */
async function imageLandmarker(){
  if(!HLimg)HLimg=await MOD.HandLandmarker.createFromOptions(FS,{baseOptions:{modelAssetPath:MODEL_URL,delegate:'CPU'},runningMode:'IMAGE',numHands:2,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5});
  return HLimg;
}
let camTick=0,lastDet=0,frameN=0;
const DET_W=256;
let detCv=null,detCx=null;
function detectFrame(v){
  const ts=Math.max(performance.now(),lastTs+1);lastTs=ts;
  try{const r=HL.detectForVideo(v,ts);return r.landmarks||[]}catch(e){return[]}
}
function camLoop(){
  requestAnimationFrame(camLoop);
  if(!HL||!stream||mjob)return;
  if(cur!=='penerjemah'&&cur!=='latih')return;
  const v=cur==='latih'?$('#pv'):$('#cv');
  if(v.readyState<2||v.currentTime===lastVT)return;
  lastVT=v.currentTime;
  const t0=performance.now();
  const hs=detectFrame(v);
  DETMS=DETMS*0.9+(performance.now()-t0)*0.1;dfc++;
  onHands(hs,performance.now());
  if(cur==='penerjemah'){staticFrame(hs,performance.now());drawHands(hs,v.videoWidth,v.videoHeight,ov,ox);tick()}
  else{
    drawHands(hs,v.videoWidth,v.videoHeight,pov,pox);
    if(recItems){recFrames++;if(hs.length)recHands++;recItems.push({t:performance.now()-recT0,hs:hs.map(h=>h.map(p=>({x:p.x,y:p.y,z:p.z})))})}
    else{const now=Date.now();if(now-lastTick>=150){lastTick=now;tick()}}
  }
}

const ov=$('#ov'),ox=ov.getContext('2d'),pov=$('#pov'),pox=pov.getContext('2d');
let pendingUnknown=null;
const HAND_LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function fitOv(ov,ox){
  const dpr=devicePixelRatio||1,w=ov.clientWidth,h=ov.clientHeight;
  if(ov.width!==Math.round(w*dpr)||ov.height!==Math.round(h*dpr)){ov.width=Math.round(w*dpr);ov.height=Math.round(h*dpr)}
  ox.setTransform(dpr,0,0,dpr,0,0);
}
function drawHands(hs,vw,vh,ov,ox){
  fitOv(ov,ox);
  const W=ov.clientWidth,H=ov.clientHeight;
  ox.clearRect(0,0,W,H);
  if(!hs.length||!vw||!vh)return;
  const sc=Math.max(W/vw,H/vh),dx=(W-vw*sc)/2,dy=(H-vh*sc)/2;
  const mirror=facing==='user';
  const P=p=>{let x=p.x*vw*sc+dx;if(mirror)x=W-x;return[x,p.y*vh*sc+dy]};
  for(const h of hs){
    ox.strokeStyle='rgba(53,182,194,.95)';ox.lineWidth=2;
    for(const[a,b]of HAND_LINKS){
      const[x1,y1]=P(h[a]),[x2,y2]=P(h[b]);
      ox.beginPath();ox.moveTo(x1,y1);ox.lineTo(x2,y2);ox.stroke();
    }
    h.forEach((pt,i)=>{
      const[x,y]=P(pt);
      ox.beginPath();ox.arc(x,y,i===0?6:4,0,Math.PI*2);
      ox.fillStyle=i===0?'#f2b705':'#ffffff';ox.fill();
      ox.strokeStyle='rgba(0,0,0,.6)';ox.lineWidth=1;ox.stroke();
    });
  }
}
/* ===== kamera ===== */
function paint(){
  const on=!!stream;
  $('#cam').classList.toggle('live',on);
  $('#cam').classList.toggle('off',!camOn);
  $('#cb').classList.toggle('off',!camOn);
  $('#ct').textContent=camOn?'Live':'Mati';
  $('#pill').style.visibility=camOn?'visible':'hidden';
  $('#cv').classList.toggle('mir',facing==='user');$('#pv').classList.toggle('mir',facing==='user');
  $('#stl').textContent=camOn?'Stop Kamera':'Mulai Kamera';
  $('#sti').innerHTML=camOn?STOPI:PLAYI;
  const c=$('#sc');if(c)c.setAttribute('aria-checked',camOn);
}
async function startStream(force){
  if((!camOn&&!force)||stream)return paint();
  try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:facing,width:{ideal:640},height:{ideal:480},frameRate:{ideal:30,max:30}},audio:false})}
  catch(e){stream=null;setDt('Kamera ditolak');}
  if(stream){buf=[];lastVT=-1;for(const id of['#cv','#pv']){const v=$(id);v.srcObject=stream;v.play().catch(()=>{})}}
  paint();
}
function stopStream(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}
  for(const id of['#cv','#pv'])$(id).srcObject=null;
  buf=[];paint();
}
function setCam(v){
  camOn=v;
  if(v){if(cur==='penerjemah'||cur==='latih')startStream();else paint()}
  else{stopStream();setDt('Kamera dimatikan')}
}
$('#stp').onclick=()=>setCam(!camOn);
async function flipCam(){if(recBusy)return setLst('Selesaikan rekaman dulu sebelum memutar kamera.');facing=facing==='user'?'environment':'user';if(stream){stopStream();await startStream(true)}else paint();$('#pcam').textContent=stream?'Matikan kamera':'Nyalakan kamera'}
$('#flip').onclick=flipCam;$('#pflip').onclick=flipCam;

/* ===== gambar & video dari berkas (Panduan & Galeri) ===== */
function seekTo(v,t){
  return new Promise(res=>{
    if(Math.abs(v.currentTime-t)<0.01){res();return}
    const done=()=>{clearTimeout(tmo);v.removeEventListener('seeked',done);res()};
    const tmo=setTimeout(done,1500);
    v.addEventListener('seeked',done);
    v.currentTime=Math.max(0,Math.min(t,v.duration-0.05));
  });
}
async function eachFrame(file,cb,fps=MFPS,maxW=480){
  if(file.type.startsWith('image/')){
    const img=new Image();const u=URL.createObjectURL(file);img.src=u;
    await img.decode();
    await cb(img,0,1);URL.revokeObjectURL(u);return;
  }
  const v=document.createElement('video');
  v.muted=true;v.playsInline=true;v.preload='auto';
  const src=URL.createObjectURL(file);v.src=src;
  await new Promise((res,rej)=>{
    const tmo=setTimeout(()=>rej(new Error('video tidak termuat')),15000);
    v.onloadeddata=()=>{clearTimeout(tmo);res()};
    v.onerror=()=>{clearTimeout(tmo);rej(new Error('format video tidak didukung peramban ini. Coba MP4 (H.264)'))};
  });
  const vw=v.videoWidth,vh=v.videoHeight;
  if(!vw||!vh)throw new Error('video tidak bisa dibaca. Coba MP4 (H.264)');
  const w=Math.min(vw,maxW),h=Math.round(w*vh/vw);
  const c=document.createElement('canvas');c.width=w;c.height=h;const g=c.getContext('2d');
  // rekaman MediaRecorder sering tidak punya durasi: paksa peramban menghitungnya
  if(!isFinite(v.duration)){
    await new Promise(res=>{const tm=setTimeout(res,2000);v.ontimeupdate=()=>{clearTimeout(tm);v.ontimeupdate=null;res()};v.currentTime=1e101});
    v.currentTime=0;
    await new Promise(res=>{const tm=setTimeout(res,1000);v.onseeked=()=>{clearTimeout(tm);v.onseeked=null;res()}});
  }
  const dur=isFinite(v.duration)&&v.duration>0?v.duration:0;
  const total=dur?Math.min(300,Math.max(1,Math.floor(dur*fps))):1;
  for(let i=0;i<total;i++){
    if(dur)await seekTo(v,i/fps);
    g.drawImage(v,0,0,w,h);
    await cb(c,i,total);
  }
  URL.revokeObjectURL(src);
}
/* item = {t (ms), hs (landmark tangan)} per frame */
function samplesFromItems(items,isImg){
  if(isImg){
    const it=items[0];if(!it||!it.hs.length)return[];
    return[Array(W).fill(feat(it.hs)).flat()];
  }
  const out=[];let seg=[];
  const flush=()=>{
    if(seg.length>=8){
      out.push(resample(seg));
      const t0=seg[0].t,t1=seg[seg.length-1].t;
      for(let s=t0;s<=t1;s+=400){
        const win=seg.filter(x=>x.t>=s&&x.t<s+SPAN);
        if(win.length>=8)out.push(resample(win));
      }
    }
    seg=[];
  };
  for(const it of items){
    if(it.hs.length){
      if(seg.length&&it.t-seg[seg.length-1].t>LOST)flush();
      seg.push({t:it.t,f:feat(it.hs)});
    }else if(seg.length&&it.t-seg[seg.length-1].t>LOST)flush();
  }
  flush();
  return out;
}
/* mengulang logika commit() pada rekaman, mengembalikan deretan kata */
function analyzeItems(items,isImg){
  if(isImg){
    const it=items[0];if(!it||!it.hs.length)return[];
    const p=predict(Array(W).fill(feat(it.hs)).flat());
    return p&&p.ok?[{w:p.word,t:0}]:[];
  }
  const outW=[];let b=[],cnd=null,stk=0,lw=null,lt=-1e9,seen=-1e9,rel=true;
  for(const it of items){
    if(it.hs.length){seen=it.t;b.push({t:it.t,f:feat(it.hs)});while(b.length&&it.t-b[0].t>SPAN)b.shift()}
    else{rel=true;if(it.t-seen>LOST)b=[]}
    if(b.length<8){cnd=null;stk=0;continue}
    const bp=bestPred(b,it.t),p=bp&&bp.p;
    if(!p)continue;
    if(p.ok){
      if(cnd===p.word)stk++;else{cnd=p.word;stk=1}
      if(stk>=3){
        if(!(p.word===lw&&(!rel||it.t-lt<COOL))){outW.push({w:p.word,t:it.t});lw=p.word;lt=it.t;rel=false}
        b=[];cnd=null;stk=0;
      }
    }else{cnd=null;stk=0}
  }
  return outW;
}
const pvc=$('#pvc'),pvx=pvc.getContext('2d');
function previewFrame(src,hs){
  const w=src.naturalWidth||src.width,h=src.naturalHeight||src.height;
  if(!w||!h)return;
  if(pvc.width!==w||pvc.height!==h){pvc.width=w;pvc.height=h}
  pvc.style.display='block';
  pvx.drawImage(src,0,0,w,h);
  const r=Math.max(2,Math.round(w/120));
  for(const lm of hs){
    lm.forEach((p,i)=>{
      pvx.beginPath();pvx.arc(p.x*w,p.y*h,i===0?r*1.6:r,0,Math.PI*2);
      pvx.fillStyle=i===0?'#f2b705':'#ffffff';pvx.fill();
    });
  }
}
function renderMres(lines){
  const b=$('#mres');b.textContent='';
  if(!lines.length){b.style.display='none';return}
  b.style.display='flex';
  lines.forEach(t=>{const d=document.createElement('div');d.textContent=t;b.append(d)});
}
$('#mf').onchange=e=>{
  MF=[...e.target.files];
  const nv=MF.filter(f=>!f.type.startsWith('image/')).length;
  setMp(MF.length?(MF.length+' berkas dipilih ('+(MF.length-nv)+' foto, '+nv+' video).'):'Belum ada berkas dipilih.');
};
// Analisis cepat: 10 fps, 360 px, mode VIDEO (pelacakan antarframe)
const AFPS=10;
let vidLM=null,vts=0;
async function videoLandmarker(){
  if(!vidLM){
    const mk=d=>HandLandmarker.createFromOptions(FS,{baseOptions:{modelAssetPath:MODEL_URL,delegate:d},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5});
    try{vidLM=await mk(HLD==='CPU'?'CPU':'GPU')}catch(e){vidLM=await mk('CPU')}
  }
  return vidLM;
}
async function runAnalyze(){
  if(mjob)return;
  if(!MF.length)return setMp('Pilih video dulu.');
  if(!hlReady)return setMp(hlErr?'Model gagal dimuat: '+hlErr:'Model belum siap, tunggu sebentar.');
  mjob=true;
  const lines=[],errs=[],allW=[];
  renderMres([]);
  const t0=performance.now();
  try{
    const vlm=await videoLandmarker();
    const ilm=await imageLandmarker();
    for(let fi=0;fi<MF.length;fi++){
      const f=MF[fi],isImg=f.type.startsWith('image/');
      try{
        const items=[];
        await eachFrame(f,async(src,i,n)=>{
          let hs;
          if(isImg){hs=ilm.detect(src).landmarks||[]}
          else{const ts=vts;vts+=Math.round(1000/AFPS);hs=vlm.detectForVideo(src,ts).landmarks||[]}
          items.push({t:i*1000/AFPS,hs});
          if(i%3===0||i===n-1)previewFrame(src,hs);
          setMp('Menganalisis '+(fi+1)+'/'+MF.length+': '+f.name+' — '+Math.round((i+1)/n*100)+'%');
        },AFPS,360);
        const ws=analyzeItems(items,isImg);
        const hands=items.filter(x=>x.hs.length).length;
        lines.push(f.name+': '+(ws.length?ws.map(x=>shownWord(x.w)+' ('+(x.t/1000).toFixed(1)+' dtk)').join(', '):'tidak ada gerakan yang dikenali')+' ('+items.length+' frame, tangan terlihat di '+hands+')');
        ws.forEach(x=>allW.push(x.w));
      }catch(e){errs.push(f.name+': '+(e.message||e))}
    }
  }catch(e){errs.push('model: '+(e.message||e))}
  finally{mjob=false}
  const sec=((performance.now()-t0)/1000).toFixed(1);
  renderMres(lines.concat(['Waktu proses: '+sec+' dtk'],errs.map(e=>'Gagal: '+e)));
  setMp('Analisis selesai.');
  if(allW.length)speak(allW.map(x=>shownWord(x.w)).join(' '),tlang());
}
$('#manal').onclick=async()=>{
  if(mjob||recBusy)return;
  const vids=MF.filter(f=>f.type.startsWith('video/'));
  if(!vids.length)return setMp('Pilih video dulu.');
  if(!hlReady)return setMp(hlErr?'Model gagal dimuat: '+hlErr:'Model belum siap, tunggu sebentar.');
  const label=curLabel();
  if(!label)return setMp($('#mcat').value==='Kalimat'?'Kalimat dikenali dari urutan kata. Latih kata penyusunnya dulu.':'Tulis dulu nama kata, huruf, atau angka di bagian rekam.');
  const before=(SD[label]||[]).length;
  for(let k=0;k<vids.length;k++){
    setMp('Melatih '+(k+1)+'/'+vids.length+': '+vids[k].name);
    await trainFromBlob(vids[k],label,m=>setMp('Video '+(k+1)+'/'+vids.length+' · '+m));
  }
  const after=(SD[label]||[]).length;
  setMp('Selesai. '+(after-before)+' contoh ditambahkan untuk “'+label+'”. Total: '+after+'.');
  MF=[];$('#mf').value='';
};
function showHasil(ok,text){
  $('#hs').textContent=ok?'Terjemahan Berhasil!':'Gerakan Belum Dikenali';
  $('#okc').style.display=ok?'':'none';
  $('#rs').textContent=text;
  location.hash='hasil';
  if(ok)speak(text,tlang());
}
// Galeri: analisis video yang dipilih, hasilnya tampil di halaman Hasil
async function analyzeVideoFile(f){
  const vlm=await videoLandmarker();
  const items=[];
  await eachFrame(f,async(src,i,n)=>{
    const ts=vts;vts+=Math.round(1000/AFPS);
    const hs=vlm.detectForVideo(src,ts).landmarks||[];
    items.push({t:i*1000/AFPS,hs});
  },AFPS,360);
  return analyzeItems(items,false);
}
$('#gal').onchange=async e=>{
  const f=e.target.files[0];e.target.value='';if(!f)return;
  if(!f.type.startsWith('video/'))return showHasil(false,'Pilih file video.');
  if(!hlReady)return showHasil(false,hlErr?'Model gagal dimuat: '+hlErr:'Model belum siap, coba lagi sebentar.');
  if(mjob)return;
  mjob=true;
  let ws=[];
  try{ws=await analyzeVideoFile(f)}
  catch(err){mjob=false;return showHasil(false,'Video tidak bisa dibaca: '+(err.message||err))}
  mjob=false;
  if(ws.length)showHasil(true,ws.map(x=>shownWord(x.w)).join(' '));
  else showHasil(false,'Gerakan belum dikenali. Coba video dengan tangan terlihat jelas.');
};

/* ===== latih kata (halaman #latih) ===== */
function renderCounts(){
  const box=$('#lcnt');box.textContent='';
  const keys=[...new Set([...ALLW,...Object.keys(SD)])];
  keys.forEach(k=>{
    const n=(SD[k]||[]).length;
    const d=document.createElement('div');d.className='row';
    const a=document.createElement('span');a.textContent=k;
    const b=document.createElement('span');b.style.flex='none';b.style.color='var(--mut)';b.textContent=n+' contoh';
    d.append(a,b);
    if(n>0){
      const x=document.createElement('button');x.className='out';x.textContent='Hapus';
      x.style.cssText='width:auto;margin:0 0 0 10px;padding:4px 12px;font-size:12px;flex:none';
      x.onclick=()=>{
        if(!confirm('Hapus semua contoh untuk “'+k+'”?'))return;
        delete SD[k];delete RES[k];saveSD();saveRES();renderCounts();tShow();
        setLst('Contoh untuk “'+k+'” dihapus.');
      };
      d.append(x);
    }
    box.append(d);
  });
  const s=$('#tsel'),cur0=s.value;s.innerHTML='';
  Object.keys(SD).filter(w=>SD[w].length).forEach(w=>{const o=document.createElement('option');o.textContent=w;s.append(o)});
  if(cur0)s.value=cur0;
  mstat();
}
$('#lsave').onclick=()=>{
  if(!pendingUnknown){setLst('Belum ada gerakan yang belum dikenal. Lakukan gerakan di kamera dulu.');return}
  const w=curLabel();
  if(!w){setLst('Tulis nama kata, huruf, atau angka dulu.');return}
  const a=SD[w]||(SD[w]=[]);
  a.push(pendingUnknown);if(a.length>MAXS)a.shift();
  pendingUnknown=null;saveSD();renderCounts();
  setLst('Gerakan disimpan sebagai “'+w+'”.');
};
$('#lexp').onclick=()=>{
  const b=new Blob([JSON.stringify(SD)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='bisindo-data-latih.json';a.click();
};
$('#limp').onchange=async e=>{
  const f=e.target.files[0];e.target.value='';if(!f)return;
  try{
    const j=JSON.parse(await f.text());
    let n=0;
    for(const w in j){
      if(!Array.isArray(j[w]))continue;
      const ok=j[w].filter(s=>Array.isArray(s)&&s.length===W*130);
      const a=SD[w]||(SD[w]=[]);
      ok.forEach(s=>{if(a.length<MAXS){a.push(s);n++}});
    }
    saveSD();renderCounts();setLst('Diimpor '+n+' contoh.');
  }catch(err){setLst('File tidak valid.')}
};
$('#lhapus').onclick=()=>{if(!confirm('Hapus semua data latih?'))return;SD={};saveSD();renderCounts();setLst('Data latih dihapus.')};

/* ===== uji akurasi (halaman #latih) ===== */
function tShow(){
  const b=$('#tbody');b.innerHTML='';
  for(const w in RES){
    const r=RES[w],tr=document.createElement('tr');
    [w,r.n,r.ok,Math.round(r.ok/r.n*100)+'%'].forEach(x=>{const td=document.createElement('td');td.textContent=x;td.style.padding='6px 4px';td.style.borderTop='1px solid var(--line)';tr.append(td)});
    b.append(tr);
  }
  $('#tst').textContent=T?'Menguji “'+T.w+'”: percobaan '+T.n+'/10, berhasil '+T.ok:'Pilih kata lalu tekan Mulai uji.';
}
function tStep(ok){if(!T)return;T.n++;if(ok)T.ok++;if(T.n>=10){RES[T.w]={n:T.n,ok:T.ok};saveRES();T=null}tShow()}
function testHit(w){if(T)tStep(w===T.w)}
$('#tgo').onclick=()=>{
  const w=$('#tsel').value;
  if(!w){$('#tst').textContent='Rekam kata dulu di atas.';return}
  if(!stream){$('#tst').textContent='Nyalakan kamera dulu.';return}
  T={w,n:0,ok:0};lastWord=null;tShow();
};
$('#tmiss').onclick=()=>tStep(false);
$('#tclr').onclick=()=>{RES={};saveRES();tShow()};
$('#tcopy').onclick=async()=>{
  const t=Object.keys(RES).map(w=>w+'\t'+RES[w].n+'\t'+RES[w].ok+'\t'+Math.round(RES[w].ok/RES[w].n*100)+'%').join('\n');
  try{await navigator.clipboard.writeText('Kata\tPercobaan\tBerhasil\tAkurasi\n'+t);$('#tst').textContent='Tabel disalin.'}catch(e){$('#tst').textContent='Gagal menyalin.'}
};

/* ===== status model ===== */
function mstat(){
  $('#mstat').textContent='Status model: '+(hlReady?'siap':(hlErr?'gagal dimuat (periksa internet)':'memuat…'))+' · Pelacak: '+HLD+' · Dataset: '+nSamples()+' contoh'+(stream?' · Deteksi: '+DETMS.toFixed(0)+' ms/frame ('+DFPS+' fps)':'');
  const nw=performance.now();if(nw-dfT>=1000){DFPS=dfc;dfc=0;dfT=nw}
}

/* ===== tema & navigasi ===== */
const SUN='<svg class="i" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.500 1.500M17.500 17.500L19 19M5 19l1.500-1.500M17.500 6.500L19 5"/></svg>',MOON='<svg class="i" viewBox="0 0 24 24" style="fill:currentColor"><path d="M20 14A8 8 0 1110 4a6 6 0 0010 10z"/></svg>',GEAR=$('#pengaturan svg').outerHTML;
const STOPI='<svg viewBox="0 0 24 24" fill="#fff"><rect x="3" y="3" width="18" height="18" rx="4"/></svg>',PLAYI='<svg viewBox="0 0 24 24" fill="#fff"><path d="M7 4l14 8-14 8z"/></svg>';
const L=[['Tolong','s0'],['Terima Kasih','s1'],['Maaf','s2'],['Selamat','s3'],['Ya','s4'],['Tidak','s5'],['Toilet','s6'],['Makan','s7']],IM={"s0": "img/img4.jpg", "s1": "img/img5.jpg", "s2": "img/img6.jpg", "s3": "img/img7.jpg", "s4": "img/img8.jpg", "s5": "img/img9.jpg", "s6": "img/img10.jpg", "s7": "img/img11.jpg"};
const WORDS=(()=>{const seen=new Set(),o=[];for(const w of [...L.map(x=>x[0]),'Tolong','Terima kasih','Toilet','Halo','Maaf','Ya','Tidak','Makan','Minum','Air','Sakit','Rumah','Sekolah','Nama','Saya']){const k=w.toLowerCase();if(!seen.has(k)){seen.add(k);o.push(w)}}return o})();
function dark(){const r=document.documentElement,t=r.dataset.theme;return t?t=='dark':matchMedia('(prefers-color-scheme:dark)').matches}
function icon(p){$('#th').innerHTML=dark()?SUN:MOON;const g=$('#gb'),on=p=='pengaturan';g.innerHTML=GEAR;g.style.background=on?'var(--pri)':'';g.style.color=on?'#fff':''}
$('#th').onclick=()=>{document.documentElement.dataset.theme=dark()?'light':'dark';icon(cur)};
function go(){
  clearTimeout(tm);let p=location.hash.slice(1);if(!R.includes(p))p='beranda';cur=p;
  R.forEach(r=>$('#'+r).classList.toggle('on',r==p));
  ['beranda','panduan','latih','tentang'].forEach(r=>{const a=$('#n-'+r);a.classList.toggle('on',r==p);a.classList.toggle('s',r!='beranda')});
  icon(p);scrollTo(0,0);
  if(p!=='latih'&&p!=='panduan'){rec=null}
  if(p!=='latih')cancelRec();
  if(p==='penerjemah'){buf=[];words=[];}
  if(p==='penerjemah'||p==='latih'){if(camOn)startStream();else paint()}else stopStream();
  if(!stream)$('#pcam').textContent='Nyalakan kamera';
}
addEventListener('hashchange',go);
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>location.hash=b.dataset.go);
$('#pl').onclick=()=>talk($('#rs').textContent,tlang());
$('#cp').onclick=async()=>{try{await navigator.clipboard.writeText($('#rs').textContent)}catch(e){}$('#cpt').textContent='Tersalin!';setTimeout(()=>$('#cpt').textContent='Salin Teks',1500)};
function sw(id,f){$(id).onclick=e=>{const b=e.currentTarget,v=b.getAttribute('aria-checked')!='true';b.setAttribute('aria-checked',v);f(v)}}
sw('#sv',v=>snd=v);sw('#sc',v=>setCam(v));
$('#sp').onclick=()=>talk($('#lt').textContent,tlang());
document.querySelectorAll('.op select').forEach(s=>s.onchange=()=>s.parentNode.querySelector('.sv').textContent='('+s.value+')');
$('#sapp').onchange=e=>{const s=tgtSel();s.value=e.target.value;s.onchange()};
$('#gd').innerHTML=L.map(([n,k])=>`<figure><img src="${IM[k]}" alt="Isyarat ${n}"><figcaption>${n}</figcaption></figure>`).join('');
const HURUF=Array.from({length:26},(_,i)=>String.fromCharCode(65+i));
const ANGKA=['0','1','2','3','4','5','6','7','8','9'];
const ALLW=[...HURUF,...ANGKA,...WORDS];
const CATS={Huruf:HURUF,Angka:ANGKA,Kata:WORDS};
const KKEY='bisindo-kalimat-v1';
let SENT=[];
try{SENT=JSON.parse(localStorage.getItem(KKEY)||'[]')||[]}catch(e){SENT=[]}
function saveSENT(){try{localStorage.setItem(KKEY,JSON.stringify(SENT))}catch(e){}}
function matchSentence(){
  for(const k of SENT){
    const n=k.words.length;
    if(n&&words.length>=n&&k.words.every((x,i)=>x.toLowerCase()===String(words[words.length-n+i]).toLowerCase()))return k.text;
  }
  return null;
}
function addOpt(sel,text,value){const o=document.createElement('option');o.textContent=text;if(value!==undefined)o.value=value;sel.append(o)}
function fillPilih(){$('#mkrow').style.display=$('#mcat').value==='Kalimat'?'':'none'}
function curLabel(){return $('#mc').value.trim()}
async function trainFromBlob(blob,label,prog=setLst){
  if(!hlReady){setLst(hlErr?'Model gagal dimuat, rekaman tidak bisa diproses.':'Model belum siap, tunggu sebentar.');return}
  mjob=true;
  try{
    prog('Memproses “'+label+'”… 0%');
    const vlm=await videoLandmarker();
    const items=[];
    await eachFrame(blob,async(src,i,n)=>{
      const ts=vts;vts+=Math.round(1000/AFPS);
      const hs=vlm.detectForVideo(src,ts).landmarks||[];
      items.push({t:i*1000/AFPS,hs});
      if(i%3===0||i===n-1)previewFrame(src,hs);
      prog('Memproses “'+label+'”… '+Math.round((i+1)/n*100)+'%');
    },AFPS,360);
    storeFromItems(items,label);
  }catch(e){setLst('Gagal memproses rekaman: '+(e.message||e))}
  finally{mjob=false}
}
function spread(a,n){if(a.length<=n)return a;const o=[];for(let i=0;i<n;i++)o.push(a[Math.round(i*(a.length-1)/(n-1||1))]);return o}
function storeFromItems(items,label){
  const hands=items.filter(x=>x.hs.length).length,pct=Math.round(100*hands/Math.max(1,items.length));
  const samples=samplesFromItems(items,false);
  if(!samples.length){setLst('Tangan tidak terdeteksi cukup ('+hands+' dari '+items.length+' frame). Pastikan tangan terlihat penuh di kamera, lalu rekam ulang.');return}
  const a=SD[label]||(SD[label]=[]);
  const room=Math.max(0,MAXS-a.length),take=Math.min(samples.length,room);
  spread(samples,take).forEach(x=>a.push(x));
  saveSD();renderCounts();mstat();
  setLst('Tersimpan '+take+' contoh untuk “'+label+'”. Total: '+a.length+(room<samples.length?' (batas '+MAXS+' tercapai)':'')+'. Tangan terlihat di '+hands+' dari '+items.length+' frame ('+pct+'%)'+(pct<50?'. Tangan sering keluar layar: dekatkan kamera dan pastikan tangan terlihat penuh.':'.'));
}
let recItems=null,recT0=0,recTimer=0,recCap=0,recLabel='',recFrames=0,recHands=0;
function setRecBtn(t){$('#prec').textContent=t}
function cancelRec(){clearInterval(recTimer);clearTimeout(recCap);recItems=null;recBusy=false;setRecBtn('Mulai rekam');$('#rtag').hidden=true;$('#rcd').hidden=true}
function finishRec(){if(!recItems)return;const items=recItems;cancelRec();storeFromItems(items,recLabel)}
async function toggleRec(label){
  if(recBusy||mjob)return;
  if(!label){setLst('Tulis dulu nama kata, huruf, atau angka.');return}
  if(!stream){setLst('Nyalakan kamera dulu.');return}
  recBusy=true;setRecBtn('Batal');
  const cd=$('#rcd');
  for(let c=3;c>0&&recBusy;c--){cd.textContent=c;cd.hidden=false;setLst('Bersiap membuat isyarat “'+label+'”…');await sleep(1000)}
  cd.hidden=true;
  if(!recBusy){setRecBtn('Mulai rekam');return}
  recLabel=label;recFrames=0;recHands=0;recT0=performance.now();recItems=[];
  setRecBtn('Berhenti & simpan');$('#rtag').hidden=false;
  recTimer=setInterval(()=>{
    const s=(performance.now()-recT0)/1000;
    $('#rtm').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0');
    setLst('Merekam “'+recLabel+'”… tangan terlihat '+Math.round(100*recHands/Math.max(1,recFrames))+'%');
  },250);
  recCap=setTimeout(finishRec,30000);
}
$('#mcat').onchange=fillPilih;
$('#msave').onclick=()=>{
  const text=$('#mkt').value.trim();
  const ws=$('#mks').value.split(',').map(x=>x.trim()).filter(Boolean);
  if(!text||!ws.length){setMp('Isi kalimat dan kata penyusunnya.');return}
  const miss=ws.filter(w=>!(SD[w]&&SD[w].length));
  SENT.push({text:text,words:ws});saveSENT();fillPilih();
  setMp('Kalimat disimpan.'+(miss.length?' Belum ada contoh untuk: '+miss.join(', ')+'.':''));
};
$('#pcam').onclick=async()=>{
  if(stream)stopStream();else await startStream(true);
  $('#pcam').textContent=stream?'Matikan kamera':'Nyalakan kamera';
};
$('#prec').onclick=()=>{
  if(recItems)return finishRec();
  if(recBusy){recBusy=false;return}
  if($('#mcat').value==='Kalimat'){setLst('Kalimat dikenali dari urutan kata. Latih kata penyusunnya dulu.');return}
  toggleRec(curLabel());
};
fillPilih();
renderCounts();tShow();
setInterval(mstat,1000);
initHL().then(mstat);
camLoop();go();


if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(BASE + 'sw.js');
}
