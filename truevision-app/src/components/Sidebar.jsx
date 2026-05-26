/* Sidebar.jsx — Left panel: user profile + history */
import { LogOut, Plus, FileVideo, ImageIcon, Trash2, Shield } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useHistory } from '../context/HistoryContext';
import { useToast } from '../context/ToastContext';

export default function Sidebar({ onNewScan, onSelectHistory }) {
  const { user, logout } = useAuth();
  const { history, clearHistory, activeId, setActiveId } = useHistory();
  const { addToast } = useToast();

  const handleLogout = () => {
    logout();
    addToast('Signed out successfully.', 'info');
  };

  const handleSelect = (entry) => {
    setActiveId(entry.id);
    onSelectHistory?.(entry);
  };

  return (
    <aside className="w-72 shrink-0 flex flex-col h-full border-r border-white/6 bg-[#08080f]">
      {/* Brand */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-white/5">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center shadow-lg shadow-violet-500/20">
          <Shield size={15} className="text-white" />
        </div>
        <div>
          <span className="text-sm font-bold text-white">TrueVision</span>
          <p className="text-[10px] text-white/25 uppercase tracking-widest">Forensic AI</p>
        </div>
      </div>

      {/* User Profile */}
      <div className="px-5 py-5 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-violet-500/20 shrink-0">
            {user?.avatar ?? user?.name?.[0]?.toUpperCase() ?? '?'}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white truncate">{user?.name ?? 'Investigator'}</p>
            <p className="text-[11px] text-white/30 truncate">{user?.email ?? ''}</p>
          </div>
        </div>

        <div className="flex gap-2 mt-4">
          <div className="flex-1 bg-white/3 rounded-xl px-3 py-2 text-center">
            <p className="text-sm font-bold text-white">{history.length}</p>
            <p className="text-[10px] text-white/25">Scans</p>
          </div>
          <div className="flex-1 bg-white/3 rounded-xl px-3 py-2 text-center">
            <p className="text-sm font-bold text-red-400">{history.filter(h => h.prediction === 'FAKE').length}</p>
            <p className="text-[10px] text-white/25">Fakes</p>
          </div>
          <div className="flex-1 bg-white/3 rounded-xl px-3 py-2 text-center">
            <p className="text-sm font-bold text-emerald-400">{history.filter(h => h.prediction === 'REAL').length}</p>
            <p className="text-[10px] text-white/25">Real</p>
          </div>
        </div>
      </div>

      {/* New Analysis Button */}
      <div className="px-4 py-3">
        <button
          onClick={onNewScan}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-violet-500/20"
        >
          <Plus size={16} />
          New Analysis
        </button>
      </div>

      {/* History */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1">
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-white/20">History</span>
          {history.length > 0 && (
            <button
              onClick={() => { clearHistory(); addToast('History cleared.', 'info'); }}
              className="text-white/15 hover:text-red-400 transition-colors"
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-white/3 flex items-center justify-center">
              <FileVideo size={18} className="text-white/15" />
            </div>
            <p className="text-xs text-white/20">No scans yet.<br/>Upload a file to begin.</p>
          </div>
        ) : (
          history.map(entry => (
            <HistoryItem
              key={entry.id}
              entry={entry}
              isActive={activeId === entry.id}
              onClick={() => handleSelect(entry)}
            />
          ))
        )}
      </div>

      {/* Logout */}
      <div className="px-4 py-4 border-t border-white/5">
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-white/30 hover:text-white/60 hover:bg-white/4 transition-all text-sm"
        >
          <LogOut size={15} />
          Sign out
        </button>
      </div>
    </aside>
  );
}

function HistoryItem({ entry, isActive, onClick }) {
  const isVideo = entry.fileType?.startsWith('video/') || entry.fileName?.match(/\.(mp4|avi|mov)$/i);
  const isFake = entry.prediction === 'FAKE';
  const date = entry.id ? new Date(entry.id).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';

  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200 group
        ${isActive
          ? 'bg-violet-600/15 border border-violet-500/20'
          : 'hover:bg-white/4 border border-transparent'
        }`}
    >
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0
        ${isFake ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
        {isVideo ? <FileVideo size={14} /> : <ImageIcon size={14} />}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-semibold text-white/70 truncate group-hover:text-white/90 transition-colors">
          {entry.fileName ?? 'Unknown'}
        </p>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className={`text-[9px] font-bold uppercase ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
            {entry.prediction ?? '—'}
          </span>
          <span className="text-[9px] text-white/15">·</span>
          <span className="text-[9px] text-white/20 font-mono truncate">{date}</span>
        </div>
      </div>
      {isActive && <div className="w-1.5 h-1.5 rounded-full bg-violet-400 shrink-0" />}
    </button>
  );
}
