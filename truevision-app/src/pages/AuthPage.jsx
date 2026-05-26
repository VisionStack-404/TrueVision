import { useState, useEffect } from 'react';
import { Eye, EyeOff, Shield, ArrowRight, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';

export default function LoginPage({ onSwitch }) {
  const { login } = useAuth();
  const { addToast } = useToast();
  const [form, setForm] = useState({ email: '', password: '' });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const { addToast: toast } = useToast();

  useEffect(() => {
    // Initialize Facebook SDK
    const initFB = () => {
      if (window.FB) {
        window.FB.init({
          appId: '761397640298692',
          cookie: true,
          xfbml: true,
          version: 'v19.0'
        });
      } else {
        setTimeout(initFB, 100);
      }
    };
    initFB();

    const handleGoogleCredential = (response) => {
      try {
        const token = response.credential;
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        const payload = JSON.parse(jsonPayload);

        let existingUser = null;
        try {
          const stored = localStorage.getItem('tv_user');
          if (stored) existingUser = JSON.parse(stored);
        } catch (err) {}

        login({
          ...(existingUser?.email === payload.email ? existingUser : {}),
          name: payload.name || payload.email.split('@')[0],
          email: payload.email,
          avatar: (payload.name || payload.email)[0].toUpperCase(),
          picture: payload.picture
        });
        toast('Logged in via Google.', 'success');
      } catch (err) {
        console.error("Google Auth Error", err);
        toast('Failed to parse Google credentials.', 'error');
      }
    };

    const initGoogle = () => {
      if (window.google && document.getElementById("google-btn-login")) {
        window.google.accounts.id.initialize({
          client_id: '1007092696181-vk9tqq0gauarejcefks7oui706c62scm.apps.googleusercontent.com',
          callback: handleGoogleCredential,
          auto_select: true, // Automatically sign in if the user has only one session
          cancel_on_tap_outside: false
        });
        window.google.accounts.id.renderButton(
          document.getElementById("google-btn-login"),
          { theme: "filled_black", size: "large", text: "continue_with", shape: "rectangular", width: "100%" }
        );
        // Show the One Tap prompt
        window.google.accounts.id.prompt();
      } else {
        setTimeout(initGoogle, 100);
      }
    };
    initGoogle();
  }, [login, toast]);

  const handleFacebookAuth = () => {
    if (!window.FB) return toast('Facebook SDK is loading, please wait.', 'warning');
    window.FB.login((response) => {
      if (response.authResponse) {
        window.FB.api('/me', {fields: 'name,email,picture'}, (res) => {
          let existingUser = null;
          try {
            const stored = localStorage.getItem('tv_user');
            if (stored) existingUser = JSON.parse(stored);
          } catch (err) {}
          
          login({
            ...(existingUser?.email === res.email ? existingUser : {}),
            name: res.name,
            email: res.email || `${res.id}@facebook.com`,
            avatar: res.name ? res.name[0].toUpperCase() : 'F',
            picture: res.picture?.data?.url
          });
          toast('Logged in via Facebook!', 'success');
        });
      } else {
        toast('Facebook login cancelled.', 'warning');
      }
    }, {scope: 'public_profile,email'});
  };

  const validate = () => {
    const e = {};
    if (!form.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) e.email = 'Enter a valid email';
    if (form.password.length < 6) e.password = 'Password must be at least 6 characters';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    // Simulate auth (replace with real API call if backend has auth)
    await new Promise(r => setTimeout(r, 1200));

    // Retrieve existing user to keep details if they exist
    let existingUser = null;
    try {
      const stored = localStorage.getItem('tv_user');
      if (stored) existingUser = JSON.parse(stored);
    } catch (err) {}

    let finalName = existingUser?.name;
    if (!finalName || existingUser?.email !== form.email) {
       finalName = form.email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    login({
      ...(existingUser?.email === form.email ? existingUser : {}),
      name: finalName,
      email: form.email,
      avatar: finalName[0].toUpperCase(),
    });
    addToast('Welcome back! You are now logged in.', 'success');
    setLoading(false);
  };

  return (
    <AuthLayout>
      <div className="animate-fade-up">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center shadow-lg shadow-violet-500/20">
            <Shield size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">TrueVision</h1>
            <p className="text-[11px] text-white/40 uppercase tracking-widest">Forensic Engine</p>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-1">Welcome back</h2>
        <p className="text-sm text-white/40 mb-8">Sign in to your forensic workspace</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="Email address" error={errors.email}>
            <input
              type="email"
              placeholder="analyst@truevision.ai"
              value={form.email}
              onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
              className="input-field"
              autoComplete="email"
            />
          </Field>

          <Field label="Password" error={errors.password}>
            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'}
                placeholder="••••••••"
                value={form.password}
                onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
                className="input-field pr-12"
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
              >
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </Field>

          <div className="flex justify-end">
            <button type="button" className="text-xs text-violet-400 hover:text-violet-300 transition-colors">
              Forgot password?
            </button>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-semibold text-sm transition-all duration-200 shadow-lg shadow-violet-500/20 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="mt-6 flex items-center gap-4">
          <div className="flex-1 h-px bg-white/5" />
          <span className="text-xs text-white/20">or</span>
          <div className="flex-1 h-px bg-white/5" />
        </div>

        <div className="mt-4 flex flex-col gap-3">
          <div id="google-btn-login" className="w-full flex justify-center"></div>
          
          <button
            type="button"
            onClick={handleFacebookAuth}
            className="w-full flex items-center justify-center gap-3 py-3 px-6 rounded-xl border border-white/8 bg-[#1877F2]/10 hover:bg-[#1877F2]/20 text-white/90 hover:text-white text-sm font-medium transition-all duration-200 border-[#1877F2]/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#1877F2"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.249h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            Continue with Facebook
          </button>
          
          <button
            type="button"
            onClick={() => toast('Twitter login requires backend configuration.', 'warning')}
            className="w-full flex items-center justify-center gap-3 py-3 px-6 rounded-xl border border-white/8 bg-[#1DA1F2]/10 hover:bg-[#1DA1F2]/20 text-white/90 hover:text-white text-sm font-medium transition-all duration-200 border-[#1DA1F2]/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#1DA1F2"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
            Continue with Twitter
          </button>
        </div>

        <p className="mt-8 text-center text-sm text-white/30">
          Don&apos;t have an account?{' '}
          <button onClick={onSwitch} className="text-violet-400 hover:text-violet-300 font-semibold transition-colors">
            Sign up
          </button>
        </p>
      </div>
    </AuthLayout>
  );
}

export function SignupPage({ onSwitch }) {
  const { login } = useAuth();
  const { addToast, addToast: toast } = useToast();
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = 'Full name is required';
    if (!form.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) e.email = 'Enter a valid email';
    if (form.password.length < 6) e.password = 'Password must be at least 6 characters';
    if (form.password !== form.confirm) e.confirm = 'Passwords do not match';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  useEffect(() => {
    // Initialize Facebook SDK
    const initFB = () => {
      if (window.FB) {
        window.FB.init({
          appId: '761397640298692',
          cookie: true,
          xfbml: true,
          version: 'v19.0'
        });
      } else {
        setTimeout(initFB, 100);
      }
    };
    initFB();

    const handleGoogleCredential = (response) => {
      try {
        const token = response.credential;
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        const payload = JSON.parse(jsonPayload);

        let existingUser = null;
        try {
          const stored = localStorage.getItem('tv_user');
          if (stored) existingUser = JSON.parse(stored);
        } catch (err) {}

        login({
          ...(existingUser?.email === payload.email ? existingUser : {}),
          name: payload.name || payload.email.split('@')[0],
          email: payload.email,
          avatar: (payload.name || payload.email)[0].toUpperCase(),
          picture: payload.picture
        });
        toast('Account created! Welcome via Google.', 'success');
      } catch (err) {
        console.error("Google Auth Error", err);
        toast('Failed to parse Google credentials.', 'error');
      }
    };

    const initGoogle = () => {
      if (window.google && document.getElementById("google-btn-signup")) {
        window.google.accounts.id.initialize({
          client_id: '1007092696181-vk9tqq0gauarejcefks7oui706c62scm.apps.googleusercontent.com',
          callback: handleGoogleCredential,
          auto_select: true,
          cancel_on_tap_outside: false
        });
        window.google.accounts.id.renderButton(
          document.getElementById("google-btn-signup"),
          { theme: "filled_black", size: "large", text: "signup_with", shape: "rectangular", width: "100%" }
        );
        // Show the One Tap prompt
        window.google.accounts.id.prompt();
      } else {
        setTimeout(initGoogle, 100);
      }
    };
    initGoogle();
  }, [login, toast]);

  const handleFacebookAuth = () => {
    if (!window.FB) return toast('Facebook SDK is loading, please wait.', 'warning');
    window.FB.login((response) => {
      if (response.authResponse) {
        window.FB.api('/me', {fields: 'name,email,picture'}, (res) => {
          let existingUser = null;
          try {
            const stored = localStorage.getItem('tv_user');
            if (stored) existingUser = JSON.parse(stored);
          } catch (err) {}
          
          login({
            ...(existingUser?.email === res.email ? existingUser : {}),
            name: res.name,
            email: res.email || `${res.id}@facebook.com`,
            avatar: res.name ? res.name[0].toUpperCase() : 'F',
            picture: res.picture?.data?.url
          });
          toast('Account created! Welcome via Facebook.', 'success');
        });
      } else {
        toast('Facebook login cancelled.', 'warning');
      }
    }, {scope: 'public_profile,email'});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    await new Promise(r => setTimeout(r, 1400));
    login({
      name: form.name.trim(),
      email: form.email,
      avatar: form.name.trim()[0].toUpperCase(),
    });
    addToast('Account created! Welcome to TrueVision.', 'success');
    setLoading(false);
  };

  return (
    <AuthLayout>
      <div className="animate-fade-up">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center shadow-lg shadow-violet-500/20">
            <Shield size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white leading-none">TrueVision</h1>
            <p className="text-[11px] text-white/40 uppercase tracking-widest">Forensic Engine</p>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-1">Create account</h2>
        <p className="text-sm text-white/40 mb-8">Join the forensic analysis platform</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="Full name" error={errors.name}>
            <input
              type="text"
              placeholder="Alex Investigator"
              value={form.name}
              onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
              className="input-field"
              autoComplete="name"
            />
          </Field>

          <Field label="Email address" error={errors.email}>
            <input
              type="email"
              placeholder="analyst@truevision.ai"
              value={form.email}
              onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
              className="input-field"
              autoComplete="email"
            />
          </Field>

          <Field label="Password" error={errors.password}>
            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'}
                placeholder="Min. 6 characters"
                value={form.password}
                onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
                className="input-field pr-12"
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
              >
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </Field>

          <Field label="Confirm password" error={errors.confirm}>
            <input
              type={showPw ? 'text' : 'password'}
              placeholder="Repeat your password"
              value={form.confirm}
              onChange={e => setForm(p => ({ ...p, confirm: e.target.value }))}
              className="input-field"
            />
          </Field>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-semibold text-sm transition-all duration-200 shadow-lg shadow-violet-500/20 disabled:opacity-60 disabled:cursor-not-allowed mt-2"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        <div className="mt-6 flex items-center gap-4">
          <div className="flex-1 h-px bg-white/5" />
          <span className="text-xs text-white/20">or sign up with</span>
          <div className="flex-1 h-px bg-white/5" />
        </div>

        <div className="mt-4 flex flex-col gap-3">
          <div id="google-btn-signup" className="w-full flex justify-center"></div>
          
          <button
            type="button"
            onClick={handleFacebookAuth}
            className="w-full flex items-center justify-center gap-3 py-3 px-6 rounded-xl border border-white/8 bg-[#1877F2]/10 hover:bg-[#1877F2]/20 text-white/90 hover:text-white text-sm font-medium transition-all duration-200 border-[#1877F2]/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#1877F2"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.249h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            Continue with Facebook
          </button>
          
          <button
            type="button"
            onClick={() => toast('Twitter login requires backend configuration.', 'warning')}
            className="w-full flex items-center justify-center gap-3 py-3 px-6 rounded-xl border border-white/8 bg-[#1DA1F2]/10 hover:bg-[#1DA1F2]/20 text-white/90 hover:text-white text-sm font-medium transition-all duration-200 border-[#1DA1F2]/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#1DA1F2"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
            Continue with Twitter
          </button>
        </div>

        <p className="mt-8 text-center text-sm text-white/30">
          Already have an account?{' '}
          <button onClick={onSwitch} className="text-violet-400 hover:text-violet-300 font-semibold transition-colors">
            Sign in
          </button>
        </p>
      </div>
    </AuthLayout>
  );
}

// ── Shared layout ────────────────────────────────────────────────
function AuthLayout({ children }) {
  return (
    <div className="min-h-screen flex h-screen overflow-hidden bg-[#060610]">
      {/* Left decorative panel */}
      <div className="hidden lg:flex w-1/2 relative overflow-hidden items-center justify-center">
        <div className="absolute inset-0 bg-gradient-to-br from-violet-950/80 via-[#060610] to-purple-950/50" />
        {/* Grid */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: 'linear-gradient(rgba(139,92,246,0.3) 1px,transparent 1px),linear-gradient(90deg,rgba(139,92,246,0.3) 1px,transparent 1px)',
            backgroundSize: '60px 60px',
          }}
        />
        {/* Radial glow */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl" />
        <div className="relative z-10 text-center px-12">
          <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-violet-500/30">
            <Shield size={36} className="text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4 tracking-tight">
            AI-Powered<br />Deepfake Detection
          </h2>
          <p className="text-sm text-white/40 max-w-xs mx-auto leading-relaxed">
            Tri-model ensemble — CNN, CViT & ETCNN — delivering forensic-grade analysis of digital media.
          </p>
          <div className="flex justify-center gap-6 mt-10">
            {['99.2% Accuracy','3 AI Models','Real-time'].map(stat => (
              <div key={stat} className="text-center">
                <div className="text-xs text-violet-300 font-semibold">{stat}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right form panel */}
      <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  );
}

function Field({ label, children, error }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-white/50 uppercase tracking-wider">{label}</label>
      {children}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  );
}
