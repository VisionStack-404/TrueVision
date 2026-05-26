import { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import { HistoryProvider } from './context/HistoryContext';
import LoginPage, { SignupPage } from './pages/AuthPage';
import Dashboard from './pages/Dashboard';
import './index.css';

function AppRouter() {
  const { isLoggedIn } = useAuth();
  const [authMode, setAuthMode] = useState('login'); // 'login' | 'signup'

  if (!isLoggedIn) {
    return authMode === 'login'
      ? <LoginPage onSwitch={() => setAuthMode('signup')} />
      : <SignupPage onSwitch={() => setAuthMode('login')} />;
  }

  return <Dashboard />;
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <HistoryProvider>
          <AppRouter />
        </HistoryProvider>
      </ToastProvider>
    </AuthProvider>
  );
}
