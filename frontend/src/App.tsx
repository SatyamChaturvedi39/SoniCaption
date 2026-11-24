import { useEffect, useRef, useState } from 'react'
import { AudioLines, Upload, Mic, Square, Play, Pause, FileAudio, Activity, GitBranch, Network, BrainCircuit, Download, Menu, CheckCircle2, AlertCircle } from 'lucide-react'
import { analyze, getLab, exportResult } from './api'
import type { Analysis } from './types'

type Tab = 'workspace'|'signal'|'pattern'|'hmm'|'model'
const stages = ['Preparing audio','Analyzing speech','Identifying environmental sounds','Building synchronized captions','Preparing results']

export default function App() {
  const [tab,setTab]=useState<Tab>('workspace')
  const [inputMode,setInputMode]=useState<'record'|'upload'>('record')
  const [recording,setRecording]=useState(false)
  const [seconds,setSeconds]=useState(0)
  const [processing,setProcessing]=useState(false)
  const [stage,setStage]=useState(0)
  const [error,setError]=useState('')
  const [file,setFile]=useState<File|null>(null)
  const [audioUrl,setAudioUrl]=useState('')
  const [result,setResult]=useState<Analysis|null>(null)
  const [lab,setLab]=useState<any>(null)
  const [playing,setPlaying]=useState(false)
  const [currentTime,setCurrentTime]=useState(0)
  const mediaRecorder=useRef<MediaRecorder|null>(null)
  const chunks=useRef<BlobPart[]>([])
  const streamRef=useRef<MediaStream|null>(null)
  const audioRef=useRef<HTMLAudioElement|null>(null)
  const timer=useRef<number|undefined>(undefined)
  const [drag,setDrag]=useState(false)

  useEffect(()=>{ if(tab!=='workspace' && tab!=='model'){ const p=tab==='signal'?'/api/v1/labs/signal':tab==='pattern'?'/api/v1/labs/dtw':'/api/v1/labs/hmm/viterbi'; getLab(p).then(setLab).catch(e=>setError(e.message)) } },[tab])
  useEffect(()=>()=>{streamRef.current?.getTracks().forEach(t=>t.stop()); if(audioUrl) URL.revokeObjectURL(audioUrl)},[audioUrl])

  function chooseFile(f?:File) {
    if(!f)return
    setError('')
    if(!f.name.toLowerCase().endsWith('.wav')) {setError('Please choose a .wav file.');return}
    setFile(f); setResult(null)
    const u=URL.createObjectURL(f); setAudioUrl(u)
  }
  async function startRecording() {
    setError('')
    if(!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {setError('Recording is not supported in this browser. Try a recent version of Chrome or Edge.');return}
    try {
      const stream=await navigator.mediaDevices.getUserMedia({audio:true})
      streamRef.current=stream
      const rec=new MediaRecorder(stream); mediaRecorder.current=rec; chunks.current=[]
      rec.ondataavailable=e=>{if(e.data.size)chunks.current.push(e.data)}
      rec.onstop=()=>{
        stream.getTracks().forEach(t=>t.stop()); streamRef.current=null
        const blob=new Blob(chunks.current,{type:rec.mimeType||'audio/webm'})
        if(!blob.size){setError('No audio was recorded. Please try again.');return}
        const f=new File([blob],`recording-${Date.now()}.webm`,{type:blob.type})
        setFile(f); setAudioUrl(URL.createObjectURL(blob))
      }
      rec.start(); setRecording(true); setSeconds(0)
      timer.current=window.setInterval(()=>setSeconds(s=>s+1),1000)
    } catch {setError('Microphone permission was denied or no microphone is available. Check browser permissions and try again.')}
  }
  function stopRecording() {
    if(timer.current)clearInterval(timer.current)
    if(mediaRecorder.current?.state==='recording')mediaRecorder.current.stop()
    setRecording(false)
  }
  async function processAudio() {
    if(!file)return
    setError(''); setProcessing(true); setStage(0)
    const tick=window.setInterval(()=>setStage(s=>Math.min(s+1,stages.length-1)),900)
    try {
      // Backend currently validates WAV. Browser MediaRecorder output may be WebM;
      // show a clear message rather than silently mislabelling the recorded format.
      if(!file.name.toLowerCase().endsWith('.wav')) throw new Error('Your browser recorded WebM audio. The current mock API accepts WAV uploads. Please use Upload WAV for this demo, or integrate browser-side audio conversion before live recording analysis.')
      const data=await analyze(file); setResult(data); setTab('workspace')
    } catch(e:any) {setError(e.message||'Unable to process audio')}
    finally {clearInterval(tick);setProcessing(false)}
  }
  function seek(t:number){if(audioRef.current){audioRef.current.currentTime=t;setCurrentTime(t)}}
  const fmt=(s:number)=>`${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}`
  const nav=[['workspace','Workspace',AudioLines],['signal','Signal Lab',Activity],['pattern','Pattern Lab',GitBranch],['hmm','HMM Trellis',Network],['model','Model Card',BrainCircuit]] as const

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><AudioLines size={22}/></div><div><b>SonicCaption</b><small>Speech + Sound Intelligence</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      {nav.map(([id,label,Icon])=><button key={id} className={`nav-item ${tab===id?'active':''}`} onClick={()=>{setTab(id);setError('')}}><Icon size={18}/>{label}</button>)}
      <div className="sidebar-bottom"><span className="demo-dot"/> DEMONSTRATION MODE</div>
    </aside>
    <main className="main">
      <header className="topbar"><div><span className="eyebrow">SONICCAPTION STUDIO</span><h1>{nav.find(n=>n[0]===tab)?.[1]}</h1></div><span className="mode-pill">● Mock data enabled</span></header>
      {error&&<div className="alert"><AlertCircle size={18}/>{error}<button onClick={()=>setError('')}>Dismiss</button></div>}
      {tab==='workspace'&&<div className="content">
        {!result ? <>
          <section className="intro"><div><span className="eyebrow">AUDIO WORKSPACE</span><h2>Turn audio into <em>understanding.</em></h2><p>Record a moment or upload a WAV file to explore speech and environmental sound captioning.</p></div><div className="intro-art"><AudioLines size={78}/></div></section>
          <div className="tabs"><button className={inputMode==='record'?'selected':''} onClick={()=>{setInputMode('record');setFile(null);setAudioUrl('')}}><Mic size={17}/> Live Recording</button><button className={inputMode==='upload'?'selected':''} onClick={()=>{setInputMode('upload');setFile(null);setAudioUrl('')}}><Upload size={17}/> Upload WAV File</button></div>
          <section className="panel input-panel">
            {inputMode==='record'?<div className="record-area">
              <div className={`mic-orb ${recording?'is-recording':''}`}><Mic size={34}/></div>
              <h3>{recording?'Recording in progress':file?'Recording ready':'Ready to record?'}</h3>
              <p>{recording?'Speak clearly. Your microphone is active.':file?'Your recording is ready for preview.':'Allow microphone access when your browser asks.'}</p>
              <div className="record-time">{fmt(seconds)}</div>
              <div className={`level-bars ${recording?'live':''}`}>{Array.from({length:31},(_,i)=><i key={i} style={{height:`${recording?15+Math.abs(Math.sin(i*2.2+seconds))*48:8+Math.abs(Math.sin(i))*9}px`}}/>)}</div>
              {!recording?<button className="primary-btn" onClick={startRecording}><Mic size={17}/> Start Recording</button>:<button className="danger-btn" onClick={stopRecording}><Square size={15}/> Stop Recording</button>}
              <small className="hint">Microphone access is required. Recording uses your browser.</small>
            </div>:<div className={`dropzone ${drag?'dragging':''}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);chooseFile(e.dataTransfer.files[0])}}>
              <div className="upload-icon"><Upload size={28}/></div><h3>Drop your WAV file here</h3><p>or browse files from your device</p>
              <label className="secondary-btn">Browse Files<input type="file" accept=".wav,audio/wav" hidden onChange={e=>chooseFile(e.target.files?.[0])}/></label>
              <small className="hint">WAV audio only · Choose a valid .wav file</small>
            </div>}
            {file&&<div className="selected-file"><FileAudio size={19}/><div><b>{file.name}</b><small>{(file.size/1024).toFixed(1)} KB · {file.type||'audio file'}</small></div><button className="text-btn" onClick={()=>{setFile(null);setAudioUrl('')}}>Remove</button></div>}
            {audioUrl&&<div className="preview"><audio ref={audioRef} src={audioUrl} controls onTimeUpdate={e=>setCurrentTime(e.currentTarget.currentTime)} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)}/></div>}
            {file&&<button className="primary-btn process-btn" onClick={processAudio} disabled={processing}>{processing?'Processing…':'Analyze Audio'} <span>→</span></button>}
            {processing&&<div className="processing"><div className="spinner"/><div><b>{stages[stage]}</b><small>Running demonstration workflow · no ML model is connected</small></div></div>}
          </section>
          <div className="info-row"><div><CheckCircle2 size={18}/><span><b>Private by design</b><small>Audio is sent to your local backend only.</small></span></div><div><Activity size={18}/><span><b>Multi-layer analysis</b><small>Speech and sound events in one timeline.</small></span></div></div>
        </>:<Results data={result} audioUrl={audioUrl} audioRef={audioRef} currentTime={currentTime} seek={seek} onBack={()=>setResult(null)} />}
      </div>}
      {tab==='signal'&&<Lab title="Signal Lab" subtitle="Explore time-domain and spectral audio features." data={lab} kind="signal"/>}
      {tab==='pattern'&&<Lab title="Pattern Lab" subtitle="Explore Dynamic Time Warping alignment." data={lab} kind="dtw"/>}
      {tab==='hmm'&&<Lab title="HMM Trellis" subtitle="Explore state transitions and Viterbi decoding." data={lab} kind="hmm"/>}
      {tab==='model'&&<div className="content"><div className="page-intro"><span className="eyebrow">MODEL OVERVIEW</span><h2>Environmental Sound Classifier</h2><p>Planned model card for the team's future ESC-50 classifier.</p></div><div className="panel"><span className="demo-label">ILLUSTRATIVE PLACEHOLDER</span><h3>SonicCaption Environmental CNN</h3><p className="muted">Architecture, dataset, trained weights, and verified evaluation metrics will be added by Member 2 after model development.</p><div className="metric-grid"><Metric label="Dataset" value="ESC-50 planned"/><Metric label="Model" value="CNN planned"/><Metric label="Status" value="Not integrated"/><Metric label="Metrics" value="Pending"/></div></div></div>}
    </main>
  </div>
}

function Results({data,audioUrl,audioRef,currentTime,seek,onBack}:any){
  return <div className="results">
    <div className="results-heading"><div><span className="eyebrow">ANALYSIS COMPLETE</span><h2>{data.filename}</h2><p>{data.duration.toFixed(1)} sec · {data.sample_rate} Hz · <span className="demo-label">DEMO RESULTS</span></p></div><button className="secondary-btn" onClick={onBack}>← New analysis</button></div>
    <div className="metric-grid"><Metric label="Speech segments" value={data.speech_segments.length}/><Metric label="Sound events" value={data.sound_events.length}/><Metric label="Speech ratio" value={`${Math.round((data.metrics.vad_speech_ratio||0)*100)}%`}/><Metric label="Analysis mode" value="Mock"/></div>
    <section className="panel"><h3>Audio player</h3>{audioUrl?<audio ref={audioRef} src={audioUrl} controls className="audio-full"/>:<p className="muted">Audio preview is not available.</p>}<small className="muted">Click any timeline entry below to jump to its timestamp.</small></section>
    <section className="panel"><div className="section-head"><div><h3>Unified audio timeline</h3><p className="muted">Speech and environmental events · synchronized timestamps</p></div><ExportMenu data={data}/></div>
      <div className="timeline">{data.timeline.map((item:any)=><button className={`timeline-item ${item.category}`} key={item.id} onClick={()=>seek(item.start)}><span className="time">{item.start.toFixed(2)}s – {item.end.toFixed(2)}s</span><b>{item.category==='speech'?'Speech':item.label}</b><span>{item.display_text}</span>{item.confidence!=null&&<small>Confidence {(item.confidence*100).toFixed(0)}%</small>}</button>)}</div>
    </section>
    <section className="panel"><h3>Speech transcription</h3>{data.speech_segments.map((s:any)=><div className="transcript" key={s.id}><span>{s.start.toFixed(2)}s – {s.end.toFixed(2)}s</span><p>{s.text}</p></div>)}</section>
    <div className="disclaimer">These are synthetic demonstration results. They are not generated by a trained speech or sound-recognition model.</div>
  </div>
}
function Metric({label,value}:any){return <div className="metric"><small>{label}</small><b>{value}</b></div>}
function ExportMenu({data}:any){return <div className="export-buttons"><Download size={15}/>{['srt','vtt','json','csv','txt'].map(f=><button key={f} onClick={()=>exportResult(f,data).catch(()=>alert('Export failed'))}>{f.toUpperCase()}</button>)}</div>}
function Lab({title,subtitle,data,kind}:any){
  return <div className="content"><div className="page-intro"><span className="eyebrow">LEARNING LAB · DEMO DATA</span><h2>{title}</h2><p>{subtitle}</p></div>
    <div className="panel"><span className="demo-label">SYNTHETIC DEMONSTRATION</span>
    {kind==='signal'?<><h3>Signal feature overview</h3><div className="wave-visual">{(data?.waveform||[0,.2,.6,.3,-.3,-.7,0,.4,.1]).map((v:number,i:number)=><i key={i} style={{height:`${15+Math.abs(v)*105}px`}}/>)}</div><div className="metric-grid"><Metric label="RMS samples" value={data?.rms?.length||5}/><Metric label="ZCR samples" value={data?.zcr?.length||5}/><Metric label="MFCC" value="Illustrative"/></div><p className="muted">FFT, STFT, MFCC, LPC and VQ panels are integration-ready placeholders. Connect Member 1's DSP output to replace these visuals.</p></>:kind==='dtw'?<><h3>Accumulated cost matrix · sample</h3><div className="matrix">{(data?.cost_matrix_sample||[[.1,1.2,2.5],[1.4,.3,1.8],[2.8,1.5,.4]]).flat().map((v:number,i:number)=><div key={i} style={{opacity:.25+Math.min(v/3,.7)}}>{v.toFixed(1)}</div>)}</div><div className="metric-grid"><Metric label="DTW distance" value={data?.distance??'—'}/><Metric label="Normalized" value={data?.normalized_distance??'—'}/><Metric label="Path length" value={data?.path_length??'—'}/></div><p className="muted">Illustrative values only. The actual DTW path and cost matrix will come from Member 1's implementation.</p></>:<><h3>Viterbi best path</h3><div className="state-path">{(data?.best_path||['Sunny','Sunny','Rainy','Rainy','Sunny']).map((s:string,i:number)=><div key={i}><small>Step {i+1}</small><b>{s}</b></div>)}</div><div className="metric-grid"><Metric label="Log likelihood" value={data?.log_likelihood??'—'}/><Metric label="States" value="2"/><Metric label="Mode" value="Mock"/></div><p className="muted">Illustrative HMM trellis. Replace with Member 2's HMM and Viterbi output when available.</p></>}
    </div>
  </div>
}
