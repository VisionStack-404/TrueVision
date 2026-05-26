import { useEffect, useRef } from 'react';

const LOGS = [
  '[SYS] Initializing ingestion matrix...',
  '[SYS] Artifact secured — forwarding to EC2 node',
  '[PROC] Handshake with inference cluster... OK',
  '[SYS] Loading CNN anomaly detection core... Active',
  '[SYS] Loading CViT structural boundary mapper... Active',
  '[SYS] Loading ETCNN texture synthesis analyzer... Active',
  '[PROC] Extracting temporal keyframes & facial biometrics...',
  '[MODEL] Spatial-frequency analysis on extracted faces...',
  '[MODEL] Computing weighted ensemble: CNN×0.33 + CViT×0.33 + ETCNN×0.34',
  '[PROC] Awaiting inference results from EC2...',
];

export default function ProcessingView({ fileName, uploadProgress }) {
  const logRef = useRef(null);
  const timerRef = useRef(null);
  const iRef = useRef(0);

  useEffect(() => {
    iRef.current = 0;
    if (logRef.current) logRef.current.innerHTML = '';

    const print = () => {
      if (!logRef.current || iRef.current >= LOGS.length) return;
      const div = document.createElement('div');
      const line = LOGS[iRef.current];
      const color = line.includes('[SYS]') ? 'text-white/30'
        : line.includes('[MODEL]') ? 'text-violet-300'
        : 'text-emerald-400';
      div.className = `text-xs font-mono ${color} animate-fade-in`;
      div.textContent = '> ' + line;
      logRef.current.appendChild(div);
      logRef.current.scrollTop = logRef.current.scrollHeight;
      iRef.current++;
      timerRef.current = setTimeout(print, Math.random() * 450 + 200);
    };
    print();
    return () => clearTimeout(timerRef.current);
  }, []);

  const progress = Math.min(uploadProgress ?? 0, 95);

  return (
    <div className="flex flex-col items-center gap-8 py-8 animate-fade-up">
      {/* Radar */}
      <div className="flex flex-col items-center gap-4">
        <div className="radar-container">
          <div className="radar-beam" />
          <div className="radar-dot" />
        </div>
        <div className="text-center">
          <p className="text-sm font-semibold text-white/80">Deep-level Analysis</p>
          <p className="text-xs text-white/30 font-mono mt-1 truncate max-w-[200px]">{fileName}</p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between text-xs text-white/30 font-mono mb-2">
          <span>Processing</span>
          <span>{progress}%</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-violet-600 to-purple-400 transition-all duration-700"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Terminal */}
      <div className="w-full max-w-lg glass rounded-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 bg-white/[0.02]">
          <span className="w-3 h-3 rounded-full bg-red-500/60" />
          <span className="w-3 h-3 rounded-full bg-amber-500/60" />
          <span className="w-3 h-3 rounded-full bg-emerald-500/60" />
          <span className="ml-2 text-xs font-mono text-white/20">system.log</span>
        </div>
        <div
          ref={logRef}
          className="p-4 h-48 overflow-y-auto flex flex-col gap-2"
        />
      </div>
    </div>
  );
}
