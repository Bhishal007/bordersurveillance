import { useState, useEffect, useRef } from 'react';
import Globe from 'react-globe.gl';
import { io } from 'socket.io-client';
import { 
  Menu, X, Home, LayoutGrid, FileVideo, ShieldCheck, 
  ChevronDown, ChevronRight, Activity, Terminal, UploadCloud, Search, Send, Bot, MessageSquare
} from 'lucide-react';
import { LineChart, Line, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = 'http://localhost:5000';
const CV_STREAM_BASE = 'http://localhost:5001/video_feed';
const ML_ENGINE_BASE = 'http://localhost:5001'; // Used for forensic uploads

// Helper function to render bold markdown text (**text**)
const formatAgentMessage = (text) => {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => 
    part.startsWith('**') && part.endsWith('**') ? 
      <strong key={i} className="text-white font-bold">{part.slice(2, -2)}</strong> : part
  );
};

// ==========================================
// 1. DASHBOARD VIEW (Omni-Agent Removed)
// ==========================================
const DashboardView = ({ alerts, sysLogs, isFeedOffline, setIsFeedOffline, streamKey, setStreamKey, isIrMode, setIsIrMode, isTracking, setIsTracking, filterTarget, setFilterTarget, loading }) => {
  const globeEl = useRef();
  const [globeSize, setGlobeSize] = useState(0);
  const [isRadarActive, setIsRadarActive] = useState(true);
  const [presetIndex, setPresetIndex] = useState(0);
  const [locationIndex, setLocationIndex] = useState(0);

  const GLOBE_PRESETS = [
    { name: "NIGHT VISION", url: "//unpkg.com/three-globe/example/img/earth-night.jpg" },
    { name: "SATELLITE", url: "//unpkg.com/three-globe/example/img/earth-blue-marble.jpg" },
    { name: "POLITICAL BORDERS", url: "//unpkg.com/three-globe/example/img/earth-day.jpg" }
  ];

  const SECTORS = [
    { name: "HQ COMMAND (KOLKATA)", lat: 22.5726, lng: 88.3639, alt: 2.0 },
    { name: "NORTHERN COMMAND (KASHMIR)", lat: 34.0837, lng: 74.7973, alt: 0.8 },
    { name: "EASTERN BORDER (ASSAM)", lat: 27.0986, lng: 93.6053, alt: 0.8 },
    { name: "WESTERN BORDER (RAJASTHAN)", lat: 27.0238, lng: 74.2179, alt: 0.8 }
  ];

  const handleVisualPreset = () => setPresetIndex((presetIndex + 1) % GLOBE_PRESETS.length);
  const handleLocationChange = () => {
    const nextIndex = (locationIndex + 1) % SECTORS.length;
    setLocationIndex(nextIndex);
    const loc = SECTORS[nextIndex];
    if (globeEl.current) globeEl.current.pointOfView({ lat: loc.lat, lng: loc.lng, altitude: loc.alt }, 2000);
  };

  const handleAcknowledge = async (id) => {
    try {
      await fetch(`${API_BASE}/api/v1/alerts/${id}/acknowledge`, { method: 'PUT' });
    } catch (error) { console.error("Failed to acknowledge"); }
  };

  useEffect(() => {
    const updateSize = () => {
      if (window.innerWidth < 1024) setGlobeSize(Math.min(window.innerWidth * 0.9, window.innerHeight * 0.45));
      else setGlobeSize(window.innerHeight * 0.75);
    };
    window.addEventListener('resize', updateSize);
    updateSize();
    setTimeout(updateSize, 100);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  const displayedAlerts = alerts.filter(a => {
    if (filterTarget === 'ALL EVENTS') return true;
    if (filterTarget === 'INTRUSION ONLY') return a.eventType === 'INTRUSION';
    if (filterTarget === 'VEHICLE MATCH') return a.eventType === 'VEHICLE_MATCH';
    return true;
  });

  const activeCameras = [...new Set(displayedAlerts.map(a => a.cameraId))];
  const globeRings = activeCameras.map(cameraId => {
    const isCritical = displayedAlerts.filter(a => a.cameraId === cameraId).some(a => a.threatLevel === 'CRITICAL' || a.threatLevel === 'HIGH');
    return { lat: 24.0, lng: 80.0, type: isCritical ? 'critical' : 'normal', maxR: isCritical ? 5 : 3, propagationSpeed: 1.2, repeatPeriod: 1200 };
  });

  const chartData = [
    { time: 'T-4H', threats: 1 }, { time: 'T-3H', threats: 3 }, { time: 'T-2H', threats: 0 },
    { time: 'T-1H', threats: 2 }, { time: 'NOW', threats: alerts.length }
  ];

  return (
    <div className="w-full h-full flex flex-col lg:flex-row overflow-hidden bg-[#050a0f]">
      {/* LEFT PANEL */}
      <div className="w-full lg:w-1/4 h-auto lg:h-full p-4 flex flex-col border-b lg:border-r border-cyan-500/30 bg-[#020508]/80 order-2 lg:order-1">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 shrink-0 rounded-full border-2 border-cyan-500 flex items-center justify-center shadow-[0_0_10px_rgba(56,189,248,0.5)]">
            <div className="w-5 h-3 border border-cyan-500 rounded-[50%] flex items-center justify-center"><div className="w-1.5 h-1.5 bg-cyan-500 rounded-full"></div></div>
          </div>
          <div><h1 className="text-xl font-bold text-cyan-400 tracking-[0.2em]">TACTICAL COMMS</h1><p className="text-[8px] opacity-70">NODE STATUS: ACTIVE</p></div>
        </div>
        <div className="mb-6 space-y-1 opacity-80 text-[9px]">
          <p className={`font-bold text-sm mt-2 ${alerts.some(a => (a.threatLevel === 'CRITICAL' || a.threatLevel === 'HIGH')) ? 'text-red-500 animate-pulse' : 'text-cyan-400'}`}>
            {alerts.some(a => (a.threatLevel === 'CRITICAL' || a.threatLevel === 'HIGH')) ? 'THREAT DETECTED' : 'NORMAL STATUS'}
          </p>
          <div className="mt-4 border-t border-cyan-500/20 pt-2 h-24 overflow-hidden flex flex-col justify-end space-y-1">
            {sysLogs.map((log, i) => (<p key={i} className={i === 0 ? "text-cyan-400 font-bold" : "opacity-50"}>{log}</p>))}
          </div>
        </div>
        <div className="mt-auto border border-cyan-500/50 bg-[#020508] p-1 rounded-sm">
          <div className="mb-1 p-1 bg-black border-b border-cyan-500/20 text-center flex justify-between px-2">
             <span className="text-cyan-400 font-bold">ML ENGINE FEED</span><span className={isFeedOffline ? "text-red-500 animate-pulse" : "text-green-400"}>{isFeedOffline ? "OFFLINE" : "ONLINE"}</span>
          </div>
          <div className="aspect-video bg-black relative border border-cyan-500/20 flex items-center justify-center overflow-hidden">
            {isFeedOffline ? (
              <div className="text-center p-2 text-red-400">
                <p className="font-bold animate-pulse">STREAM DISCONNECTED</p>
                <button onClick={() => { setStreamKey(Date.now()); setIsFeedOffline(false); }} className="mt-2 text-[8px] border border-cyan-500/40 px-2 py-0.5 text-cyan-400 hover:bg-cyan-500/20 cursor-pointer">RECONNECT</button>
              </div>
            ) : (
              <img src={`${CV_STREAM_BASE}?t=${streamKey}`} alt="AI Live Feed" onError={() => setIsFeedOffline(true)} className={`absolute inset-0 w-full h-full object-cover transition-all duration-500 ${isIrMode ? 'invert sepia saturate-200 hue-rotate-180 opacity-80' : 'opacity-70 mix-blend-luminosity'}`} />
            )}
            <div className="absolute inset-0 bg-cyan-500/10 mix-blend-overlay pointer-events-none"></div>
            <p className="z-10 absolute bottom-1 right-1 bg-black/80 px-2 py-1 text-cyan-400 font-bold border border-cyan-500/30 text-[8px]">{isIrMode ? 'YOLO-FLIR' : 'YOLO-LIVE'}</p>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <button onClick={() => setIsTracking(!isTracking)} className={`py-1 border border-cyan-500/50 cursor-pointer transition-colors ${isTracking ? 'bg-cyan-500 text-black font-bold shadow-[0_0_10px_rgba(56,189,248,0.5)]' : 'bg-transparent text-cyan-400 hover:bg-cyan-500/20'}`}>TRACK {isTracking ? 'ON' : 'OFF'}</button>
            <button onClick={() => setIsIrMode(!isIrMode)} className={`py-1 border border-cyan-500/50 cursor-pointer transition-colors ${isIrMode ? 'bg-cyan-500 text-black font-bold shadow-[0_0_10px_rgba(56,189,248,0.5)]' : 'bg-transparent text-cyan-400 hover:bg-cyan-500/20'}`}>FLIR IR</button>
          </div>
        </div>
      </div>

      {/* CENTER PANEL */}
      <div className="flex-1 h-full relative flex items-center justify-center order-1 lg:order-2 overflow-hidden bg-transparent">
        <div className="absolute top-6 z-30 w-full px-8 flex justify-center gap-6 pointer-events-none">
          <div className="bg-[#020508]/80 border border-cyan-500/50 px-6 py-2 rounded-md backdrop-blur-sm flex flex-col items-center min-w-[140px] shadow-[0_0_15px_rgba(56,189,248,0.2)]">
            <span className="text-[8px] text-cyan-500/70 tracking-widest">TOTAL ALERTS</span>
            <span className="text-2xl font-bold text-cyan-400">{alerts.length}</span>
          </div>
          <div className="bg-[#020508]/80 border border-red-500/50 px-6 py-2 rounded-md backdrop-blur-sm flex flex-col items-center min-w-[140px] shadow-[0_0_15px_rgba(255,0,0,0.2)]">
            <span className="text-[8px] text-red-500/70 tracking-widest">CRITICAL THREATS</span>
            <span className="text-2xl font-bold text-red-500">{alerts.filter(a => a.threatLevel === 'CRITICAL' || a.threatLevel === 'HIGH').length}</span>
          </div>
          <div className="bg-[#020508]/80 border border-green-500/50 px-6 py-2 rounded-md backdrop-blur-sm flex flex-col items-center min-w-[140px] shadow-[0_0_15px_rgba(34,197,94,0.2)]">
            <span className="text-[8px] text-green-500/70 tracking-widest">RESOLVED</span>
            <span className="text-2xl font-bold text-green-400">{alerts.filter(a => a.threatLevel === 'RESOLVED').length}</span>
          </div>
        </div>

        <div className={`w-[75vh] h-[75vh] rounded-full border border-cyan-400 overflow-hidden relative shadow-[0_0_80px_rgba(56,189,248,0.15)] flex items-center justify-center transition-opacity duration-1000 ${isRadarActive ? 'opacity-100' : 'opacity-30'}`}>
          <div className="absolute inset-0 z-10 flex items-center justify-center cursor-move">
            {globeSize > 0 && (
              <Globe ref={globeEl} width={globeSize} height={globeSize} backgroundColor="rgba(0,0,0,0)" globeImageUrl={GLOBE_PRESETS[presetIndex].url} bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png" ringsData={isTracking ? globeRings : []} ringColor={(ring) => (t) => `rgba(${ring.type === 'critical' ? '255, 0, 60' : '56, 189, 248'}, ${1 - Math.pow(t, 0.5)})`} ringMaxRadius="maxR" ringPropagationSpeed="propagationSpeed" ringRepeatPeriod="repeatPeriod" showAtmosphere={true} atmosphereColor="#38bdf8" atmosphereAltitude={0.15} animateIn={false} onGlobeReady={() => { if(globeEl.current) globeEl.current.pointOfView({ lat: SECTORS[0].lat, lng: SECTORS[0].lng, altitude: SECTORS[0].alt }, 3000); }} />
            )}
          </div>
          <div className={`absolute inset-0 z-20 bg-[linear-gradient(rgba(56,189,248,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(56,189,248,0.1)_1px,transparent_1px)] bg-[size:40px_40px] transition-transform duration-[10s] pointer-events-none ${isRadarActive ? 'scale-110 rotate-1' : 'scale-100 rotate-0'}`}></div>
        </div>
        <div className="absolute bottom-4 z-20 flex gap-4 px-2 w-full justify-center">
          <button onClick={handleLocationChange} className="bg-[#020508] border border-cyan-500/50 px-6 py-2 rounded-md hover:bg-cyan-500/20 cursor-pointer flex flex-col items-center"><span>LOCATION</span><span className="text-[7px] text-cyan-500/50">{SECTORS[locationIndex].name}</span></button>
          <button onClick={() => setIsRadarActive(!isRadarActive)} className={`px-6 py-2 rounded-md font-bold transition-all border border-cyan-400 cursor-pointer ${isRadarActive ? 'bg-cyan-500 text-black shadow-[0_0_15px_rgba(56,189,248,0.5)]' : 'bg-transparent text-cyan-400 opacity-50'}`}>RADAR {isRadarActive ? 'ON' : 'OFF'}</button>
          <button onClick={handleVisualPreset} className="bg-[#020508] border border-cyan-500/50 px-6 py-2 rounded-md hover:bg-cyan-500/20 cursor-pointer flex flex-col items-center"><span>VISUALS</span><span className="text-[7px] text-cyan-500/50">{GLOBE_PRESETS[presetIndex].name}</span></button>
        </div>
      </div>

      {/* RIGHT PANEL (ANALYTICS & FULL ALERTS) */}
      <div className="w-full lg:w-1/4 h-full p-4 flex flex-col border-l border-cyan-500/30 z-10 bg-[#020508]/80 order-3">
        
        <div className="border border-cyan-500/30 p-2 bg-[#020508] mb-2 rounded-sm flex flex-col">
          <span className="text-[8px] text-cyan-500/70 tracking-widest mb-1 px-1">THREAT FREQUENCY (ROLLING)</span>
          <div className="w-full h-16">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <Line type="monotone" dataKey="threats" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2, fill: '#38bdf8' }} />
                <Tooltip contentStyle={{ backgroundColor: '#020508', border: '1px solid #38bdf8', fontSize: '10px' }} itemStyle={{ color: '#38bdf8' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="border border-cyan-500/30 p-2 bg-[#020508] mb-2 rounded-sm">
          <div className="flex justify-between items-center">
            <span className="text-cyan-400">TARGET:</span>
            <select value={filterTarget} onChange={(e) => setFilterTarget(e.target.value)} className="bg-black border border-cyan-500/50 text-cyan-400 text-[10px] p-1 outline-none cursor-pointer">
              <option value="ALL EVENTS">ALL EVENTS</option>
              <option value="INTRUSION ONLY">INTRUSION ONLY</option>
            </select>
          </div>
        </div>

        {/* ALERTS FEED EXPANDED */}
        <div className="flex-1 overflow-y-auto pr-1 custom-scrollbar mb-2">
          <div className="border border-cyan-500 p-1.5 bg-cyan-500/10 text-cyan-400 mb-2 font-bold sticky top-0 backdrop-blur-md z-10 text-[9px] flex justify-between">
            <span>⚠ ALERTS ({displayedAlerts.length})</span>
            <span className="text-green-400">AUTONOMOUS</span>
          </div>
          {loading ? <p className="text-center opacity-50 mt-6 animate-pulse">FETCHING DB RECORDS...</p> : (
            <div className="space-y-2">
              {displayedAlerts.map(alert => (
                <div key={alert._id} className={`border p-2 bg-black text-[9px] cursor-pointer hover:bg-cyan-500/10 ${(alert.threatLevel === 'CRITICAL' || alert.threatLevel === 'HIGH') ? 'border-red-500/50 shadow-[0_0_10px_rgba(255,0,0,0.2)]' : 'border-cyan-500/30'}`}>
                  <div className="flex justify-between text-cyan-400 mb-1">
                    <span className="font-bold flex items-center gap-1">
                      {alert.cameraId}
                      {alert.evaluatedByAgent && (
                        <span className="text-[7px] text-cyan-400 bg-cyan-900/40 border border-cyan-500/30 px-1 rounded-sm">AI TRIAGED</span>
                      )}
                    </span>
                    <span>CONF: {(alert.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-[8px] opacity-80">
                    <p>TYPE: {alert.eventType.replace('_', ' ')}</p>
                    <p className={(alert.threatLevel === 'CRITICAL' || alert.threatLevel === 'HIGH') ? 'text-red-400 font-bold' : ''}>LVL: {alert.threatLevel}</p>
                  </div>
                  {alert.sitrep && (
                    <div className="mt-1 p-1 bg-cyan-950/40 border-l-2 border-cyan-400 text-cyan-300 text-[8px] font-mono leading-tight">
                      <span className="text-cyan-500 font-bold block mb-0.5">SITREP REPORT:</span>
                      {formatAgentMessage(alert.sitrep)}
                    </div>
                  )}
                  <div className="mt-2 pt-1 border-t border-cyan-500/20 flex justify-between items-center text-[8px]">
                    <div className="flex gap-2 items-center">
                      {alert.blockchain?.verified ? (<><span className="text-green-400 font-bold tracking-widest">✓ ON-CHAIN</span></>) : (alert.threatLevel === 'CRITICAL' || alert.threatLevel === 'HIGH') ? (<span className="text-yellow-400 animate-pulse tracking-widest">⏳ ANCHORING...</span>) : (<span className="opacity-50 tracking-widest">STANDARD LOG</span>)}
                    </div>
                    {alert.threatLevel !== 'RESOLVED' ? (
                      <button onClick={(e) => { e.stopPropagation(); handleAcknowledge(alert._id); }} className="bg-red-500/20 text-red-400 border border-red-500/50 px-2 py-0.5 font-bold hover:bg-red-500 hover:text-white transition-colors cursor-pointer">
                        [ ACKNOWLEDGE ]
                      </button>
                    ) : (<span className="text-cyan-500/50 font-bold tracking-widest">RESOLVED</span>)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 2. LIVE RTSP MULTI-CAMERA GRID (Updated WebCam)
// ==========================================
const LiveGridView = () => {
  const [rtspUrl, setRtspUrl] = useState('');
  const [activeStream, setActiveStream] = useState(false);
  return (
    <div className="flex-1 flex flex-col p-4 bg-[#050a0f]">
      <div className="mb-4 flex gap-4 bg-[#020508] p-3 border border-cyan-500/30 items-center">
        <Terminal size={18} className="text-cyan-500 shrink-0" />
        <input type="text" placeholder="ENTER RTSP STREAM URL (e.g., rtsp://camera-ip:554/stream)" className="flex-1 bg-black border border-cyan-500/50 p-2 text-cyan-400 text-[10px] outline-none placeholder:text-cyan-500/30" value={rtspUrl} onChange={(e) => setRtspUrl(e.target.value)} />
        <button onClick={() => setActiveStream(true)} className="px-6 py-2 bg-cyan-500 text-black font-bold hover:bg-cyan-400 transition-colors cursor-pointer shadow-[0_0_10px_rgba(56,189,248,0.3)]">INITIALIZE UPLINK</button>
      </div>
      <div className="flex-1 grid grid-cols-2 gap-4">
        {/* NODE 01 - CONNECTED TO LIVE WEBCAM FEED */}
        <div className={`border ${activeStream ? 'border-red-500 shadow-[0_0_20px_rgba(255,0,0,0.1)]' : 'border-cyan-500/30'} bg-black relative flex flex-col overflow-hidden group`}>
          <div className={`absolute top-2 left-2 bg-black/80 px-2 py-1 text-[8px] font-bold border z-10 ${activeStream ? 'text-red-500 border-red-500 animate-pulse' : 'text-cyan-500 border-cyan-500/50'}`}>NODE 01: ACTIVE ML ENGINE</div>
          {activeStream ? (
            <img src={CV_STREAM_BASE} alt="Live WebCam Grid" className="w-full h-full object-cover opacity-80" />
          ) : (
            <div className="flex-1 flex items-center justify-center text-cyan-500/30 flex-col gap-2"><Search size={32} className="opacity-20" /><span>AWAITING RTSP SOURCE</span></div>
          )}
          <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] pointer-events-none opacity-50"></div>
        </div>
        {[2, 3, 4].map(num => (
          <div key={num} className="border border-cyan-500/30 bg-[#020508] relative flex items-center justify-center overflow-hidden">
            <div className="absolute top-2 left-2 bg-black/80 px-2 py-1 text-[8px] text-cyan-500/50 border border-cyan-500/20">NODE 0{num}: OFFLINE</div>
            <div className="text-cyan-500/10 text-4xl font-bold tracking-widest flex flex-col items-center">NO SIGNAL</div>
            <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] pointer-events-none opacity-20"></div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ==========================================
// 3. FORENSIC VIDEO UPLOAD (Updated Logic with Drag & Drop & Visual Proof)
// ==========================================
const ForensicView = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setAnalysisResult(null);
    
    const formData = new FormData();
    formData.append('video', file);

    try {
      const res = await fetch(`${ML_ENGINE_BASE}/analyze_video`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setAnalysisResult(data);
    } catch (error) {
      setAnalysisResult({ error: "Failed to connect to ML Engine on port 5001. Ensure Python server is active." });
    } finally {
      setUploading(false);
    }
  };

  // Drag and Drop Handlers
  const handleDragOver = (e) => {
    e.preventDefault(); // Stops the browser from opening the file
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); // Stops the browser from opening the file
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      // Check if it's an mp4 or video file
      if (droppedFile.type.startsWith('video/')) {
        setFile(droppedFile);
      }
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 bg-[#050a0f]">
      <div 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`w-full max-w-2xl border-2 border-dashed ${isDragging ? 'border-cyan-400 bg-cyan-900/20' : 'border-cyan-500/50 bg-[#020508]'} p-12 flex flex-col items-center text-center relative overflow-hidden group transition-all duration-300`}
      >
        <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
        <UploadCloud size={64} className={`mb-6 transition-transform duration-500 ${isDragging ? 'text-cyan-300 scale-125' : 'text-cyan-500 group-hover:scale-110'}`} />
        <h2 className="text-xl font-bold text-cyan-400 mb-2 tracking-widest">SECURE FORENSIC DROPZONE</h2>
        <p className="text-cyan-500/50 mb-8 text-[10px] tracking-[0.2em]">DRAG & DROP OR UPLOAD .MP4 CLIPS FOR YOLO BATCH PROCESSING</p>
        
        <input type="file" accept="video/mp4" className="hidden" id="video-upload" onChange={(e) => setFile(e.target.files[0])} />
        
        <label htmlFor="video-upload" className="px-8 py-3 bg-black border border-cyan-500 text-cyan-400 font-bold hover:bg-cyan-500 hover:text-black transition-all cursor-pointer shadow-[0_0_15px_rgba(56,189,248,0.2)] z-10">
          {file ? `[ FILE SELECTED: ${file.name.substring(0,20)}... ]` : "SELECT VIDEO FILE"}
        </label>
        
        {file && !uploading && !analysisResult && (
          <button onClick={handleUpload} className="mt-6 px-12 py-3 bg-red-500 text-white font-bold tracking-widest cursor-pointer hover:bg-red-600 transition-colors shadow-[0_0_20px_rgba(255,0,0,0.4)] z-10">
            INITIATE AI ANALYSIS
          </button>
        )}
        
        {uploading && (
          <div className="w-full mt-8 max-w-md z-10">
            <div className="flex justify-between text-[8px] text-cyan-400 mb-1">
              <span>YOLO11 ENGINE RUNNING</span>
              <span className="animate-pulse">ANALYZING FRAMES...</span>
            </div>
            <div className="w-full h-1 bg-black border border-cyan-500/30 relative overflow-hidden"><div className="absolute top-0 left-0 h-full w-[100%] bg-cyan-500 shadow-[0_0_10px_rgba(56,189,248,1)] animate-pulse"></div></div>
          </div>
        )}

        {analysisResult && (
          <div className="mt-8 w-full max-w-md p-4 bg-black border border-cyan-500/50 text-left z-10">
            <h3 className="text-cyan-400 font-bold mb-2">ANALYSIS REPORT</h3>
            {analysisResult.error ? (
              <p className="text-red-500 text-[10px]">{analysisResult.error}</p>
            ) : (
              <div className="flex gap-4 items-center">
                <div className="text-[10px] text-cyan-300 space-y-1 flex-1">
                  <p>STATUS: <span className="text-green-400">COMPLETE</span></p>
                  <p>THREATS DETECTED: {analysisResult.threats || 0}</p>
                  <p>CONFIDENCE: {analysisResult.confidence || "N/A"}</p>
                  <p>LOGGED TO BLOCKCHAIN: {analysisResult.onChain ? "YES" : "NO"}</p>
                </div>
                
                {/* NEW: Displays the visual proof snapshot from the ML Engine */}
                {analysisResult.snapshot && (
                  <div className="w-32 h-24 border border-cyan-500/50 overflow-hidden shrink-0 relative group">
                    <img 
                      src={analysisResult.snapshot} 
                      alt="Threat Snapshot" 
                      className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-150" 
                    />
                    <div className="absolute inset-0 border border-red-500 animate-pulse pointer-events-none opacity-50"></div>
                  </div>
                )}
              </div>
            )}
            <button onClick={() => { setFile(null); setAnalysisResult(null); }} className="mt-4 border border-cyan-500/50 px-4 py-1 text-cyan-500 hover:bg-cyan-500/10 cursor-pointer text-[8px]">NEW UPLOAD</button>
          </div>
        )}
      </div>
    </div>
  );
};

// ==========================================
// 4. BLOCKCHAIN AUDIT VIEW
// ==========================================
const BlockchainAuditView = () => {
  const [selectedTx, setSelectedTx] = useState(null);
  const [ledgerData, setLedgerData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLedger = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/alerts`);
        const data = await response.json();
        const verifiedAlerts = data.filter(alert => alert.blockchain && alert.blockchain.verified);
        setLedgerData(verifiedAlerts);
      } catch (error) { console.error("Error fetching ledger:", error); } finally { setLoading(false); }
    };
    fetchLedger();
  }, []);

  return (
    <div className="flex-1 flex p-4 gap-4 relative bg-[#050a0f]">
      <div className="flex-1 border border-cyan-500/30 bg-[#020508] p-6 flex flex-col">
        <div className="flex items-center gap-3 border-b border-cyan-500/30 pb-4 mb-4">
          <ShieldCheck size={24} className="text-cyan-500" />
          <h2 className="text-cyan-400 font-bold tracking-widest text-lg">IMMUTABLE THREAT LEDGER</h2>
        </div>
        <div className="grid grid-cols-4 text-cyan-500/50 text-[8px] font-bold pb-2 mb-2 border-b border-cyan-500/10 px-2">
          <span>SOURCE NODE</span><span>CLASSIFICATION</span><span>TIMESTAMP</span><span className="text-right">STATUS</span>
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
          {loading ? ( <div className="text-cyan-500/50 text-center mt-10 animate-pulse">SYNCING WITH LEDGER...</div>
          ) : ledgerData.length === 0 ? ( <div className="text-cyan-500/50 text-center mt-10">NO ON-CHAIN RECORDS FOUND</div>
          ) : (
            ledgerData.map(record => (
              <div key={record._id} onClick={() => setSelectedTx(record)} className={`grid grid-cols-4 items-center border border-cyan-500/20 bg-black p-3 mb-2 cursor-pointer transition-colors ${selectedTx?._id === record._id ? 'bg-cyan-500/20 border-cyan-500' : 'hover:bg-cyan-500/10'}`}>
                <div className="text-cyan-400 font-bold">{record.cameraId}</div>
                <div className="text-red-400">{record.eventType.replace('_', ' ')}</div>
                <div className="text-cyan-500/70">{new Date(record.createdAt).toLocaleTimeString()}</div>
                <div className="text-right"><span className="bg-green-500/10 text-green-400 border border-green-500/30 px-2 py-1 rounded-sm tracking-widest font-bold">✓ SECURED</span></div>
              </div>
            ))
          )}
        </div>
      </div>
      {selectedTx && (
        <div className="w-[400px] border border-cyan-500 bg-[#020508] shadow-[0_0_30px_rgba(56,189,248,0.15)] p-6 flex flex-col shrink-0 animate-in slide-in-from-right-10 duration-300 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-16 h-16 bg-cyan-500/10 blur-xl"></div>
          <div className="absolute bottom-0 left-0 w-16 h-16 bg-cyan-500/10 blur-xl"></div>
          <div className="flex justify-between items-center border-b border-cyan-500/30 pb-4 mb-6">
            <span className="text-cyan-400 font-bold tracking-[0.2em]">CRYPTOGRAPHIC RECEIPT</span>
            <button onClick={() => setSelectedTx(null)} className="text-cyan-500 hover:text-white cursor-pointer bg-cyan-500/10 p-1"><X size={16}/></button>
          </div>
          <div className="space-y-6 text-[10px] flex-1 z-10">
            <div><div className="text-cyan-500/50 mb-1 tracking-widest">TRANSACTION HASH</div><div className="text-white break-all bg-black border border-cyan-500/20 p-2 font-mono">{selectedTx.blockchain.txHash}</div></div>
            <div className="grid grid-cols-2 gap-4">
              <div><div className="text-cyan-500/50 mb-1 tracking-widest">BLOCK NUMBER</div><div className="text-white text-lg font-bold">{selectedTx.blockchain.blockNumber || 'PENDING'}</div></div>
              <div><div className="text-cyan-500/50 mb-1 tracking-widest">NETWORK</div><div className="text-green-400 font-bold bg-green-500/10 border border-green-500/30 px-2 py-1 inline-block">HARDHAT LOCAL</div></div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div><div className="text-cyan-500/50 mb-1 tracking-widest">AI CONFIDENCE</div><div className="text-white">{(selectedTx.confidence * 100).toFixed(1)}%</div></div>
              <div><div className="text-cyan-500/50 mb-1 tracking-widest">TIMESTAMP</div><div className="text-white">{new Date(selectedTx.createdAt).toLocaleString()}</div></div>
            </div>
          </div>
          <button className="mt-8 border border-cyan-500 text-cyan-500 py-3 hover:bg-cyan-500 hover:text-black font-bold tracking-[0.2em] cursor-pointer transition-colors z-10 shadow-[0_0_15px_rgba(56,189,248,0.2)]">VERIFY ON EXPLORER</button>
        </div>
      )}
    </div>
  );
};

// ==========================================
// 5. OMNI-AGENT FULL VIEW (New Dedicated UI)
// ==========================================
const OmniAgentView = ({ chatMessages, chatInput, setChatInput, handleSendChat, chatLoading }) => {
  const chatEndRef = useRef(null);
  
  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  return (
    <div className="flex-1 flex p-4 bg-[#050a0f]">
      <div className="w-full max-w-4xl mx-auto border border-cyan-500/50 bg-[#020508] p-6 flex flex-col shadow-[0_0_30px_rgba(56,189,248,0.1)]">
        
        <div className="flex items-center gap-4 border-b border-cyan-500/30 pb-4 mb-4">
          <div className="w-12 h-12 rounded-full border border-cyan-500 flex items-center justify-center bg-cyan-500/10 shadow-[0_0_15px_rgba(56,189,248,0.4)]">
            <Bot size={24} className="text-cyan-400" />
          </div>
          <div>
            <h2 className="text-cyan-400 font-bold tracking-[0.3em] text-lg">OMNI-AGENT TERMINAL</h2>
            <p className="text-[10px] text-cyan-500/50">LOCAL LLM SECURE UPLINK ACTIVE</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar mb-4">
          {chatMessages.map((msg, i) => (
            <div key={i} className={`p-4 rounded-sm max-w-[80%] ${msg.sender === 'user' ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 ml-auto' : 'bg-black text-cyan-400 border border-cyan-500/40 mr-auto'}`}>
              <div className="flex items-center gap-2 mb-2 border-b border-cyan-500/20 pb-1">
                {msg.sender === 'user' ? <Terminal size={12}/> : <Bot size={12}/>}
                <span className="opacity-70 text-[10px] tracking-widest">{msg.sender === 'user' ? 'COMMANDER' : 'AI INTEL'}</span>
              </div>
              <div className="text-[11px] leading-relaxed whitespace-pre-wrap">
                {formatAgentMessage(msg.text)}
              </div>
            </div>
          ))}
          {chatLoading && (
            <div className="p-4 rounded-sm max-w-[80%] bg-black text-cyan-500/50 border border-cyan-500/20 animate-pulse">
              [ PROCESSING TACTICAL QUERY VIA LLOCAL LLM... ]
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSendChat} className="mt-auto flex gap-2 pt-4 border-t border-cyan-500/30 relative">
          <input 
            type="text" 
            placeholder="ENTER DIRECTIVE OR QUERY INTEL AGENT..." 
            value={chatInput} 
            onChange={(e) => setChatInput(e.target.value)}
            className="flex-1 bg-black border border-cyan-500/50 p-4 text-[12px] text-cyan-400 outline-none placeholder:text-cyan-500/30 focus:border-cyan-400 focus:shadow-[0_0_10px_rgba(56,189,248,0.2)] transition-all"
          />
          <button type="submit" disabled={chatLoading} className="px-8 bg-cyan-500/20 border border-cyan-500 text-cyan-400 hover:bg-cyan-500 hover:text-black cursor-pointer flex items-center justify-center transition-all disabled:opacity-50">
            <Send size={18} />
          </button>
        </form>

      </div>
    </div>
  );
};

// ==========================================
// 6. MAIN APP SHELL
// ==========================================
export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState('dashboard');
  const [isSurveillanceOpen, setIsSurveillanceOpen] = useState(true);

  // Lifted Dashboard States
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isIrMode, setIsIrMode] = useState(false);
  const [isTracking, setIsTracking] = useState(true);
  const [filterTarget, setFilterTarget] = useState('ALL EVENTS');
  const [isFeedOffline, setIsFeedOffline] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [sysLogs, setSysLogs] = useState(["[SYS] WAITING FOR CV STREAM...", "[SYS] UPLINK ESTABLISHED..."]);

  // Lifted Chat States (so history persists across tabs)
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([{ sender: 'agent', text: 'OMNI-AGENT ONLINE. Awaiting tactical queries.' }]);
  const [chatLoading, setChatLoading] = useState(false);

  const logAction = (action) => {
    const timestamp = new Date().toISOString().split('T')[1].substring(0, 8);
    setSysLogs(prev => [`[${timestamp}Z] ${action}`, ...prev].slice(0, 5));
  };

  // Socket & Alerts Logic
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/alerts`);
        const data = await response.json();
        setAlerts(data);
      } catch (error) { logAction("ERROR: DB SYNC FAILED"); } finally { setLoading(false); }
    };
    fetchAlerts();

    const socket = io(API_BASE);
    socket.on('NEW_ALERT', (newAlert) => {
      setAlerts(prev => [newAlert, ...prev]);
      logAction(`WARNING: NEW INSTANT ALERT FROM ${newAlert.cameraId}`);
    });
    socket.on('ALERT_VERIFIED', (verifiedAlert) => {
      setAlerts(prev => prev.map(a => a._id === verifiedAlert._id ? verifiedAlert : a));
      logAction(`[SECURE] BLOCKCHAIN ANCHOR CONFIRMED: ${verifiedAlert.cameraId}`);
    });
    return () => { socket.off('NEW_ALERT'); socket.off('ALERT_VERIFIED'); socket.disconnect(); };
  }, []);

  // Chat Send Logic
  const handleSendChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMsg = chatInput.trim();
    setChatMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, { sender: 'agent', text: data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { sender: 'agent', text: '[ERR]: Agent response timeout.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#050a0f] text-cyan-400 font-mono overflow-hidden selection:bg-cyan-500 selection:text-black uppercase text-[10px] tracking-widest">
      {/* SIDEBAR NAVIGATION */}
      <div className={`${isSidebarOpen ? 'w-64' : 'w-16'} transition-all duration-300 ease-in-out border-r border-cyan-500/30 bg-[#020508] flex flex-col z-50 shrink-0`}>
        <div className="h-16 flex items-center justify-between px-4 border-b border-cyan-500/30">
          {isSidebarOpen && (<div className="flex items-center gap-2 overflow-hidden"><Terminal size={18} className="text-cyan-500 shrink-0" /><span className="font-bold tracking-[0.2em] text-xs">SSB C2 NODE</span></div>)}
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-1 hover:bg-cyan-500/20 rounded transition-colors text-cyan-500 cursor-pointer mx-auto">{isSidebarOpen ? <X size={18} /> : <Menu size={18} />}</button>
        </div>
        
        <div className="flex-1 overflow-y-auto py-4 flex flex-col gap-1 px-2 custom-scrollbar">
          <button onClick={() => setActiveView('dashboard')} className={`flex items-center gap-3 px-3 py-3 rounded-md transition-colors cursor-pointer w-full text-left ${activeView === 'dashboard' ? 'bg-cyan-500/20 text-white shadow-[0_0_10px_rgba(56,189,248,0.2)] border border-cyan-500/50' : 'hover:bg-cyan-500/10 text-cyan-600'}`}>
            <Home size={18} className="shrink-0" />{isSidebarOpen && <span>Dashboard</span>}
          </button>
          
          <button onClick={() => setActiveView('agent')} className={`flex items-center gap-3 px-3 py-3 mt-1 rounded-md transition-colors cursor-pointer w-full text-left ${activeView === 'agent' ? 'bg-cyan-500/20 text-white shadow-[0_0_10px_rgba(56,189,248,0.2)] border border-cyan-500/50' : 'hover:bg-cyan-500/10 text-cyan-600'}`}>
            <MessageSquare size={18} className="shrink-0" />{isSidebarOpen && <span>Omni-Agent Comm</span>}
          </button>

          <div className="mt-2">
            <button onClick={() => { setIsSurveillanceOpen(!isSurveillanceOpen); if (!isSidebarOpen) setIsSidebarOpen(true); }} className="flex items-center justify-between px-3 py-3 rounded-md transition-colors hover:bg-cyan-500/10 text-cyan-600 cursor-pointer w-full text-left">
              <div className="flex items-center gap-3"><LayoutGrid size={18} className="shrink-0" />{isSidebarOpen && <span>Surveillance</span>}</div>
              {isSidebarOpen && (isSurveillanceOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />)}
            </button>
            {isSidebarOpen && isSurveillanceOpen && (
              <div className="flex flex-col gap-1 pl-9 pr-2 mt-1 border-l border-cyan-500/20 ml-5">
                <button onClick={() => setActiveView('live')} className={`py-2 px-2 text-left rounded transition-colors cursor-pointer ${activeView === 'live' ? 'text-white bg-cyan-500/10 font-bold border-l-2 border-cyan-500' : 'text-cyan-600 hover:text-cyan-400'}`}>Live RTSP Grid</button>
                <button onClick={() => setActiveView('forensics')} className={`py-2 px-2 text-left rounded transition-colors cursor-pointer ${activeView === 'forensics' ? 'text-white bg-cyan-500/10 font-bold border-l-2 border-cyan-500' : 'text-cyan-600 hover:text-cyan-400'}`}>Forensic Upload</button>
              </div>
            )}
          </div>

          <button onClick={() => setActiveView('audit')} className={`flex items-center gap-3 px-3 py-3 mt-2 rounded-md transition-colors cursor-pointer w-full text-left ${activeView === 'audit' ? 'bg-cyan-500/20 text-white shadow-[0_0_10px_rgba(56,189,248,0.2)] border border-cyan-500/50' : 'hover:bg-cyan-500/10 text-cyan-600'}`}>
            <ShieldCheck size={18} className="shrink-0" />{isSidebarOpen && <span>Blockchain Ledger</span>}
          </button>
        </div>

        {isSidebarOpen && (
          <div className="p-4 border-t border-cyan-500/30 text-[8px] flex flex-col gap-1 shrink-0">
            <div className="flex justify-between"><span className="opacity-60">ML ENGINE</span><span className="text-green-400">ONLINE</span></div>
            <div className="flex justify-between"><span className="opacity-60">OMNI-AGENT</span><span className="text-cyan-400">OLLAMA L3</span></div>
            <div className="flex justify-between"><span className="opacity-60">BLOCKCHAIN</span><span className="text-green-400">SYNCED</span></div>
          </div>
        )}
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col bg-[#050a0f] relative overflow-hidden">
        <div className="h-16 border-b border-cyan-500/30 flex items-center px-6 justify-between bg-[#020508]/80 backdrop-blur-sm z-40 shrink-0">
          <h1 className="text-lg font-bold text-white tracking-[0.3em]">
            {activeView === 'dashboard' && 'SYSTEM DASHBOARD'}
            {activeView === 'agent' && 'OMNI-AGENT TERMINAL'}
            {activeView === 'live' && 'TACTICAL MULTI-GRID'}
            {activeView === 'forensics' && 'FORENSIC VIDEO ANALYSIS'}
            {activeView === 'audit' && 'ON-CHAIN THREAT LEDGER'}
          </h1>
          <div className="flex items-center gap-4"><span className="text-cyan-600 animate-pulse">● LIVE UPLINK</span><span className="opacity-50">{new Date().toISOString().split('T')[1].substring(0, 8)}Z</span></div>
        </div>
        <div className="flex-1 flex flex-col overflow-hidden relative">
          {activeView === 'dashboard' && <DashboardView alerts={alerts} loading={loading} filterTarget={filterTarget} setFilterTarget={setFilterTarget} isIrMode={isIrMode} setIsIrMode={setIsIrMode} isTracking={isTracking} setIsTracking={setIsTracking} isFeedOffline={isFeedOffline} setIsFeedOffline={setIsFeedOffline} streamKey={streamKey} setStreamKey={setStreamKey} sysLogs={sysLogs} />}
          {activeView === 'agent' && <OmniAgentView chatMessages={chatMessages} chatInput={chatInput} setChatInput={setChatInput} handleSendChat={handleSendChat} chatLoading={chatLoading} />}
          {activeView === 'live' && <LiveGridView />}
          {activeView === 'forensics' && <ForensicView />}
          {activeView === 'audit' && <BlockchainAuditView />}
        </div>
      </div>
    </div>
  );
}