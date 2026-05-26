import { createContext, useContext, useState, useCallback } from 'react';

const HistoryContext = createContext(null);

export function HistoryProvider({ children }) {
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem('tv_history');
      return stored ? JSON.parse(stored) : [];
    } catch { return []; }
  });

  const [activeId, setActiveId] = useState(null);

  const addEntry = useCallback((entry) => {
    setHistory(prev => {
      const next = [{ id: Date.now(), ...entry }, ...prev].slice(0, 50);
      localStorage.setItem('tv_history', JSON.stringify(next));
      return next;
    });
  }, []);

  const clearHistory = useCallback(() => {
    localStorage.removeItem('tv_history');
    setHistory([]);
  }, []);

  return (
    <HistoryContext.Provider value={{ history, addEntry, clearHistory, activeId, setActiveId }}>
      {children}
    </HistoryContext.Provider>
  );
}

export const useHistory = () => useContext(HistoryContext);
