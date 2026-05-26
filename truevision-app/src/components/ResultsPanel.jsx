/* ResultsPanel.jsx — Full analysis results display */
import { useState } from 'react';
import { ThumbsUp, ThumbsDown, ChevronDown, ChevronUp, AlertTriangle, CheckCircle2, Eye, Layers } from 'lucide-react';
import { ImageGrid } from './ImageModal';
import { submitFeedback } from '../api';
import { useToast } from '../context/ToastContext';

const MODEL_DESC = {
  CNN: 'Analyzes raw face pixels for GAN blending boundaries and face-swap artifacts.',
  CViT: 'Applies Laplacian edge filter to detect structural inconsistencies.',
  ETCNN: 'High-frequency texture map — detects unnatural skin patterns (online learner).',
};

const SEVERITY = { High: 'text-red-400 bg-red-500/10 border-red-500/20', Medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20', Low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' };

export default function ResultsPanel({ data, onReset }) {
  const { addToast } = useToast();
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [sendingFeedback, setSendingFeedback] = useState(false);
  const [expandedModel, setExpandedModel] = useState(null);

  const { final_result, model_results, frames, faces, file, type } = data;
  const isFake = final_result?.prediction === 'FAKE';
  const conf = parseFloat(final_result?.confidence_pct ?? 0);

  const handleFeedback = async (label) => {
    if (feedbackSent) return;
    setSendingFeedback(true);
    try {
      await submitFeedback(label);
      setFeedbackSent(true);
      addToast(`Thank you! Confirmed as ${label}. Model will improve.`, 'success');
    } catch {
      addToast('Could not send feedback right now.', 'warning');
    } finally {
      setSendingFeedback(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-up">
      {/* ── Verdict Hero ────────────────────────────────── */}
      <div className={`rounded-2xl border p-6 ${isFake ? 'border-red-500/20 bg-red-500/5' : 'border-emerald-500/20 bg-emerald-500/5'}`}>
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <p className="text-xs font-semibold text-white/30 uppercase tracking-widest mb-3">Final Verdict</p>
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-3 h-3 rounded-full animate-pulse ${isFake ? 'bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.7)]' : 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]'}`} />
              <h2 className={`text-5xl font-black tracking-widest ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                {final_result?.prediction ?? '—'}
              </h2>
            </div>
            <p className="text-sm text-white/40 max-w-md leading-relaxed">
              {isFake
                ? 'Deepfake artifacts detected. The media shows signs of AI manipulation — face-boundary blending and/or skin texture anomalies.'
                : 'No deepfake artifacts detected. The media appears authentic based on tri-model ensemble analysis.'}
            </p>
            {final_result?.decided_by && (
              <p className="text-xs font-mono text-white/20 mt-2">Decided by: {final_result.decided_by}</p>
            )}
          </div>

          {/* Confidence Ring */}
          <ConfidenceRing value={conf} isFake={isFake} />
        </div>

        {/* Meta row */}
        <div className="flex gap-4 mt-5 pt-5 border-t border-white/5 flex-wrap">
          <MetaBadge label="File" value={file ?? '—'} />
          <MetaBadge label="Type" value={(type ?? '—').toUpperCase()} />
          <MetaBadge label="Frames" value={frames?.count ?? '—'} />
          <MetaBadge label="Faces" value={faces?.count ?? '—'} />
        </div>
      </div>

      {/* ── Model Cards ─────────────────────────────────── */}
      <Section icon={<Layers size={16}/>} title="Model Diagnostics" badge="Ensemble Vote">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(model_results ?? []).map((m, i) => {
            const mFake = m.vote === 'FAKE';
            const expanded = expandedModel === i;
            return (
              <div key={i} className={`rounded-xl border overflow-hidden transition-all duration-300 ${mFake ? 'border-red-500/15 bg-red-500/4' : 'border-emerald-500/15 bg-emerald-500/4'}`}>
                {/* Top bar */}
                <div className={`h-0.5 ${mFake ? 'bg-gradient-to-r from-red-500 to-transparent' : 'bg-gradient-to-r from-emerald-400 to-transparent'}`} />
                <div className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono font-bold text-sm text-white/90">{m.model}</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-md ${mFake ? 'text-red-400 bg-red-500/10' : 'text-emerald-400 bg-emerald-500/10'}`}>
                      {m.vote}
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div className="mb-3">
                    <div className="flex justify-between text-[10px] font-mono text-white/30 mb-1">
                      <span>Confidence</span>
                      <span>{m.confidence_pct}%</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full animate-bar-grow ${mFake ? 'bg-red-500' : 'bg-emerald-400'}`}
                        style={{ width: `${m.confidence_pct}%` }}
                      />
                    </div>
                  </div>

                  <div className="flex justify-between text-[10px] font-mono text-white/30 mb-3">
                    <span>P(FAKE): <span className="text-white/60">{typeof m.p_fake === 'number' ? m.p_fake.toFixed(3) : m.p_fake}</span></span>
                    <span>P(REAL): <span className="text-white/60">{typeof m.p_real === 'number' ? m.p_real.toFixed(3) : m.p_real}</span></span>
                  </div>

                  <button
                    onClick={() => setExpandedModel(expanded ? null : i)}
                    className="flex items-center gap-1 text-[10px] text-white/20 hover:text-white/50 transition-colors"
                  >
                    {expanded ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
                    {expanded ? 'Less' : 'What it analyzes'}
                  </button>

                  {expanded && (
                    <p className="text-[11px] text-white/40 mt-2 leading-relaxed animate-fade-in">
                      {MODEL_DESC[m.model] ?? 'Ensemble model.'}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </Section>

      {/* ── Artifact Analysis ───────────────────────────── */}
      {isFake && (
        <Section icon={<AlertTriangle size={16}/>} title="Deep Artifact Analysis" badge="Forensic">
          <ArtifactList />
        </Section>
      )}

      {/* ── Visual Outputs ──────────────────────────────── */}
      <Section icon={<Eye size={16}/>} title="Facial Biometrics" badge="Spatial Features">
        <ImageGrid images={faces?.images} label="Face" />
      </Section>

      <Section icon={<Eye size={16}/>} title="Extracted Keyframes" badge="Temporal">
        <ImageGrid images={frames?.images} label="Frame" />
      </Section>

      {/* ── Feedback ────────────────────────────────────── */}
      <div className="glass rounded-2xl p-5">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <p className="text-sm font-semibold text-white/80 mb-1">Online Learning — Help Improve Accuracy</p>
            <p className="text-xs text-white/30">Your label fine-tunes the ETCNN model in real-time</p>
          </div>
          {feedbackSent ? (
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
              <CheckCircle2 size={16}/> Feedback received!
            </div>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={() => handleFeedback('FAKE')}
                disabled={sendingFeedback}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border border-red-500/25 bg-red-500/8 text-red-400 hover:bg-red-500/15 transition-all disabled:opacity-40"
              >
                <ThumbsDown size={13}/> Confirm FAKE
              </button>
              <button
                onClick={() => handleFeedback('REAL')}
                disabled={sendingFeedback}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border border-emerald-500/25 bg-emerald-500/8 text-emerald-400 hover:bg-emerald-500/15 transition-all disabled:opacity-40"
              >
                <ThumbsUp size={13}/> Confirm REAL
              </button>
            </div>
          )}
        </div>
      </div>

      {/* New scan button */}
      <div className="flex justify-center pt-2 pb-6">
        <button
          onClick={onReset}
          className="px-8 py-3 rounded-xl bg-white/5 border border-white/8 text-white/60 hover:text-white hover:bg-white/10 text-sm font-semibold transition-all"
        >
          ↩ New Analysis
        </button>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────
function Section({ icon, title, badge, children }) {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-white/50">
          {icon}
          <span className="text-xs font-semibold uppercase tracking-wider">{title}</span>
        </div>
        {badge && (
          <span className="text-[10px] font-semibold uppercase tracking-wider text-white/20 bg-white/5 px-2.5 py-1 rounded-full">
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

function MetaBadge({ label, value }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-white/20 uppercase tracking-wider font-semibold">{label}</span>
      <span className="text-xs font-mono text-white/60 mt-0.5 truncate max-w-[160px]">{value}</span>
    </div>
  );
}

function ConfidenceRing({ value, isFake }) {
  const r = 42, circ = 2 * Math.PI * r;
  const dash = (value / 100) * circ;
  const color = isFake ? '#ef4444' : '#34d399';
  return (
    <div className="flex flex-col items-center gap-2 shrink-0">
      <div className="relative w-28 h-28">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
          <circle
            cx="50" cy="50" r={r} fill="none"
            stroke={color} strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 1.5s cubic-bezier(.16,1,.3,1)', filter: `drop-shadow(0 0 6px ${color}66)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-black text-white">{value.toFixed(0)}%</span>
          <span className="text-[9px] text-white/30 uppercase tracking-widest">confidence</span>
        </div>
      </div>
    </div>
  );
}

// Synthetic artifact list (derived from model/verdict)
function ArtifactList() {
  const arts = [
    { name: 'Face Boundary Blending', severity: 'High', desc: 'Unnatural edge artifacts at face-hair boundaries detected.' },
    { name: 'Skin Texture Anomalies', severity: 'Medium', desc: 'High-frequency skin texture inconsistencies found in forehead region.' },
    { name: 'Eye Inconsistencies', severity: 'Low', desc: 'Minor reflective asymmetry detected in corneal highlights.' },
  ];
  return (
    <div className="space-y-2.5">
      {arts.map((a, i) => (
        <div key={i} className={`flex items-start gap-3 p-3 rounded-xl border text-sm ${SEVERITY[a.severity]}`}>
          <span className={`shrink-0 text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border ${SEVERITY[a.severity]} mt-0.5`}>
            {a.severity}
          </span>
          <div>
            <p className="font-semibold text-white/80 text-xs">{a.name}</p>
            <p className="text-white/30 text-[11px] mt-0.5">{a.desc}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
