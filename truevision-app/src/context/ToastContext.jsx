import { createContext, useContext, useState, useCallback } from 'react';

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);

function ToastContainer({ toasts, onRemove }) {
  if (!toasts.length) return null;
  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-3 pointer-events-none">
      {toasts.map(t => (
        <Toast key={t.id} toast={t} onRemove={onRemove} />
      ))}
    </div>
  );
}

function Toast({ toast, onRemove }) {
  const colors = {
    success: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
    error:   'border-red-500/30   bg-red-500/10   text-red-300',
    info:    'border-violet-500/30 bg-violet-500/10 text-violet-300',
    warning: 'border-amber-500/30  bg-amber-500/10  text-amber-300',
  };
  const icons = {
    success: '✓', error: '✕', info: 'ℹ', warning: '⚠',
  };
  return (
    <div
      className={`animate-toast pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl border glass shadow-2xl min-w-[280px] max-w-sm text-sm font-medium ${colors[toast.type] || colors.info}`}
      onClick={() => onRemove(toast.id)}
    >
      <span className="text-base shrink-0">{icons[toast.type] || icons.info}</span>
      <span>{toast.message}</span>
    </div>
  );
}
