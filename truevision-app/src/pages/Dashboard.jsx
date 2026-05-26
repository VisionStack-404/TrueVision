/* Dashboard.jsx — Main workspace (right panel) */
import { useState, useCallback } from 'react';
import { Loader2, Upload, AlertCircle, Wifi, WifiOff } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ProcessingView from '../components/ProcessingView';
import ResultsPanel from '../components/ResultsPanel';
import Sidebar from '../components/Sidebar';
import { processFile, checkHealth } from '../api';
import { useHistory } from '../context/HistoryContext';
import { useToast } from '../context/ToastContext';
import { useEffect } from 'react';

const STATE = { IDLE: 'idle', PROCESSING: 'processing', RESULTS: 'results', NO_FACE: 'no_face', ERROR: 'error' };

export default function Dashboard() {
  const [view, setView] = useState(STATE.IDLE);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [results, setResults] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [noFaceData, setNoFaceData] = useState(null);
  const [proxyOnline, setProxyOnline] = useState(null);
  const { addEntry } = useHistory();
  const { addToast } = useToast();

  // Health-check proxy
  useEffect(() => {
    checkHealth()
      .then(() => setProxyOnline(true))
      .catch(() => setProxyOnline(false));
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setView(STATE.PROCESSING);
    setUploadPct(0);

    try {
      const res = await processFile(selectedFile, (evt) => {
        if (evt.total) setUploadPct(Math.round((evt.loaded / evt.total) * 100));
      });
      const data = res.data;

      if (data.status === 'error') {
        setErrorMsg(data.message ?? 'Unknown error from server.');
        setView(STATE.ERROR);
        return;
      }

      if (data.status === 'no_face') {
        setNoFaceData(data);
        setView(STATE.NO_FACE);
        addToast('No faces detected in the media.', 'warning');
        return;
      }

      setResults(data);
      setView(STATE.RESULTS);

      // Save to history
      addEntry({
        fileName: data.file,
        fileType: selectedFile.type,
        prediction: data.final_result?.prediction,
        confidence: data.final_result?.confidence_pct,
        data,
      });
      addToast(
        `Analysis complete — ${data.final_result?.prediction} (${data.final_result?.confidence_pct}%)`,
        data.final_result?.prediction === 'FAKE' ? 'error' : 'success'
      );
    } catch (err) {
      setErrorMsg(err.response?.data?.detail ?? err.message ?? 'Request failed.');
      setView(STATE.ERROR);
      addToast('Analysis failed. Check backend connection.', 'error');
    }
  }, [selectedFile, addEntry, addToast]);

  const reset = () => {
    setView(STATE.IDLE);
    setSelectedFile(null);
    setResults(null);
    setErrorMsg('');
    setNoFaceData(null);
    setUploadPct(0);
  };

  const handleSelectHistory = (entry) => {
    if (entry.data) {
      setResults(entry.data);
      setView(STATE.RESULTS);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#060610]">
      {/* Mesh background */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-700/6 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-purple-700/4 rounded-full blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage: 'linear-gradient(rgba(139,92,246,0.5) 1px,transparent 1px),linear-gradient(90deg,rgba(139,92,246,0.5) 1px,transparent 1px)',
            backgroundSize: '64px 64px',
          }}
        />
      </div>

      {/* Sidebar */}
      <Sidebar onNewScan={reset} onSelectHistory={handleSelectHistory} />

      {/* Main panel */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Topbar */}
        <header className="flex items-center justify-between px-8 py-4 border-b border-white/5 bg-[#060610]/80 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <span className="text-xs text-white/30">
              {view === STATE.IDLE && 'New Analysis'}
              {view === STATE.PROCESSING && 'Processing...'}
              {view === STATE.RESULTS && 'Analysis Report'}
              {view === STATE.NO_FACE && 'No Face Found'}
              {view === STATE.ERROR && 'Analysis Failed'}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-white/20">
            {proxyOnline === true && <><Wifi size={12} className="text-emerald-400"/><span className="text-emerald-400">Proxy Online</span></>}
            {proxyOnline === false && <><WifiOff size={12} className="text-red-400"/><span className="text-red-400">Proxy Offline</span></>}
            {proxyOnline === null && <span className="text-white/15">Connecting...</span>}
          </div>
        </header>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {/* IDLE — Upload */}
          {view === STATE.IDLE && (
            <div className="max-w-2xl mx-auto animate-fade-up">
              <div className="mb-8">
                <h1 className="text-3xl font-bold text-white tracking-tight mb-2">
                  Detect Manipulations
                </h1>
                <p className="text-sm text-white/35 leading-relaxed">
                  Upload a video or image to run forensic-grade deepfake analysis using our 3-model AI ensemble.
                </p>
              </div>

              <UploadZone onFile={setSelectedFile} disabled={false} />

              {selectedFile && (
                <button
                  onClick={handleAnalyze}
                  className="mt-5 w-full flex items-center justify-center gap-3 py-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-bold text-sm transition-all shadow-xl shadow-violet-500/20 hover:-translate-y-0.5"
                >
                  <Upload size={16} />
                  Run Analysis
                </button>
              )}

              {/* Capability pills */}
              <div className="mt-8 glass rounded-2xl p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/20 mb-4">Detection Capabilities</p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { model: 'CNN', desc: 'Face-swap & GAN blending', color: 'violet' },
                    { model: 'CViT', desc: 'Edge & boundary artifacts', color: 'purple' },
                    { model: 'ETCNN', desc: 'Texture synthesis (online)', color: 'indigo' },
                  ].map(m => (
                    <div key={m.model} className="bg-white/3 rounded-xl p-3 border border-white/5">
                      <p className="text-xs font-bold text-violet-300 mb-1 font-mono">{m.model}</p>
                      <p className="text-[10px] text-white/25 leading-relaxed">{m.desc}</p>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] font-mono text-white/15 mt-4 text-center">
                  Weighted ensemble · CNN×33% + CViT×33% + ETCNN×34%
                </p>
              </div>
            </div>
          )}

          {/* PROCESSING */}
          {view === STATE.PROCESSING && (
            <div className="max-w-xl mx-auto">
              <ProcessingView fileName={selectedFile?.name ?? ''} uploadProgress={uploadPct} />
            </div>
          )}

          {/* RESULTS */}
          {view === STATE.RESULTS && results && (
            <div className="max-w-4xl mx-auto">
              <ResultsPanel data={results} onReset={reset} />
            </div>
          )}

          {/* NO FACE */}
          {view === STATE.NO_FACE && noFaceData && (
            <NoFaceState data={noFaceData} onRetry={reset} />
          )}

          {/* ERROR */}
          {view === STATE.ERROR && (
            <ErrorState message={errorMsg} onRetry={reset} />
          )}
        </div>
      </main>
    </div>
  );
}

function NoFaceState({ data, onRetry }) {
  return (
    <div className="max-w-lg mx-auto mt-16 text-center animate-fade-up">
      <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-5">
        <AlertCircle size={28} className="text-amber-400" />
      </div>
      <h2 className="text-xl font-bold text-white mb-2">No Faces Detected</h2>
      <p className="text-sm text-white/35 mb-6 leading-relaxed">{data.message}</p>
      {data.user_tip && (
        <div className="glass rounded-xl p-4 text-left mb-6 border border-amber-500/10">
          <p className="text-xs text-amber-400 font-semibold mb-2">Tips for better results:</p>
          <p className="text-xs text-white/35 leading-relaxed">{data.user_tip}</p>
        </div>
      )}
      <button onClick={onRetry} className="px-6 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-colors">
        Try Another File
      </button>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="max-w-lg mx-auto mt-16 text-center animate-fade-up">
      <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-5">
        <AlertCircle size={28} className="text-red-400" />
      </div>
      <h2 className="text-xl font-bold text-white mb-2">Analysis Failed</h2>
      <p className="text-sm text-white/35 mb-2 leading-relaxed">{message}</p>
      <p className="text-xs text-white/20 mb-6">Make sure the local backend is running on port 8000.</p>
      <div className="glass rounded-xl p-3 text-left mb-6 border border-red-500/10 font-mono text-xs text-red-300/50">
        <code>uvicorn app:app --reload (in /backend)</code>
      </div>
      <button onClick={onRetry} className="px-6 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-colors">
        Try Again
      </button>
    </div>
  );
}
