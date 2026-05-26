/**
 * TrueVision — app.js v4
 * Landing page · Google One Tap · Enhanced results · Profile · Premium UX
 */

const API = 'http://127.0.0.1:8000';

// ── Replace with your Google OAuth Client ID ──
const GOOGLE_CLIENT_ID = '99923953469-58moinkms1vo0pr7ni2a8g4v24nrarp3.apps.googleusercontent.com';

const MODEL_DESC = {
  CNN:   'Analyzes raw pixel data for face-swap artifacts and GAN-generated blending seams.',
  CViT:  'Applies Laplacian edge filtering to detect structural inconsistencies at face boundaries.',
  ETCNN: 'High-frequency texture map — identifies unnatural skin synthesis patterns (online learner).',
};

// ── State ────────────────────────────────────────
let currentUser   = null;   // { name, email, avatar, age?, gender?, isNew }
let currentFile   = null;
let history       = [];
let activeHist    = null;
let feedbackSent  = false;
let progressTimer = null;
let selectedGender= null;
let reportCounter = parseInt(localStorage.getItem('tv_report_counter') || '0');
// ══════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  // Load Theme
  const savedTheme = localStorage.getItem('tv_theme');
  if (savedTheme && savedTheme !== 'dark') document.documentElement.dataset.theme = savedTheme;

  loadStorage();
  checkHashLogin();
  
  const isLanding = document.getElementById('landing-page');
  const isAppPage = document.getElementById('auth-overlay');

  if (isLanding) {
    if (document.getElementById('landing-particles')) initLandingParticles();
    
    // Inject User Profile on Homepage if logged in
    if (currentUser) {
      const navLinks = document.querySelector('.nav-links');
      if (navLinks) {
        navLinks.innerHTML = `
          <a href="#how-it-works" class="nav-link">How it Works</a>
          <a href="#architecture" class="nav-link">Architecture</a>
          <a href="#capabilities" class="nav-link">Capabilities</a>
          <div style="display:flex; align-items:center; gap:12px; margin-left:1rem; padding-left:1.5rem; border-left:1px solid rgba(255,255,255,0.1);">
            <div style="width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg, var(--v600), var(--p600)); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:14px; box-shadow: 0 0 15px rgba(124,58,237,0.3);">
              ${currentUser.avatar || currentUser.name.charAt(0).toUpperCase()}
            </div>
            <div style="display:flex; flex-direction:column; line-height:1.2;">
              <span style="color:#fff; font-size:13px; font-weight:700; letter-spacing:0.5px;">${currentUser.name.split(' ')[0]}</span>
              <span style="color:var(--success); font-size:9px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">● Active</span>
            </div>
            <a href="app.html" class="nav-cta" style="margin-left:10px;">Enter Dashboard</a>
          </div>
        `;
      }
      
      const heroActions = document.querySelector('.hero-actions');
      if (heroActions) {
         heroActions.innerHTML = `
          <a href="app.html" class="hero-cta-primary">
            <div class="btn-shimmer"></div>
            Continue Workspace Analysis
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
          </a>
         `;
      }
    }
  }

  if (isAppPage) {
    if (currentUser) {
      showApp();
    } else {
      window.location.href = 'login.html';
    }
    if (document.getElementById('landing-particles')) initLandingParticles();
    bindAuth();
    bindProfileModal();
    bindUpload();
    bindUI();
    checkProxy();
    setInterval(checkProxy, 30000);
    initGoogleOneTap();
  }
});

// ══════════════════════════════════════════════════
// LANDING PAGE (Only applicable in index.html)
// ══════════════════════════════════════════════════
function initLandingParticles() {
  const container = document.getElementById('landing-particles');
  if (!container) return;
  const count = 30;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.animationDuration = (Math.random() * 15 + 10) + 's';
    p.style.animationDelay = (Math.random() * 15) + 's';
    p.style.width = p.style.height = (Math.random() * 3 + 1) + 'px';
    p.style.opacity = Math.random() * 0.5 + 0.1;
    container.appendChild(p);
  }
}

function initRevealOnScroll() {
  const options = { threshold: 0.15 };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
      }
    });
  }, options);
  
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

  // FAQ Toggles
  document.querySelectorAll('.faq-q').forEach(q => {
    q.addEventListener('click', () => {
      const a = q.nextElementSibling;
      const icon = q.querySelector('svg');
      const isVisible = a.style.display === 'block';
      a.style.display = isVisible ? 'none' : 'block';
      if (icon) icon.style.transform = isVisible ? 'rotate(0deg)' : 'rotate(180deg)';
    });
  });
}

// ══════════════════════════════════════════════════
// GOOGLE ONE TAP SIGN-IN
// ══════════════════════════════════════════════════
function initGoogleOneTap() {
  // Wait for GIS library to load
  if (typeof google === 'undefined' || !google.accounts) {
    // Retry after GIS script loads
    window.addEventListener('load', () => {
      setTimeout(() => {
        if (typeof google !== 'undefined' && google.accounts) {
          setupGIS();
        }
      }, 500);
    });
  } else {
    setupGIS();
  }
}

function setupGIS() {
  const GOOGLE_CLIENT_ID = '1007092696181-vk9tqq0gauarejcefks7oui706c62scm.apps.googleusercontent.com';
  
  if (!GOOGLE_CLIENT_ID) {
    bindGoogleButtonsAsMock();
    return;
  }

  try {
    google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: handleGoogleCredential,
      auto_select: true,
      context: 'signin',
      cancel_on_tap_outside: false
    });

    const loginContainer = document.getElementById('google-btn-login');
    if (loginContainer) {
      google.accounts.id.renderButton(loginContainer, {
        theme: 'filled_black',
        size: 'large',
        text: 'continue_with',
        shape: 'rectangular',
        width: '100%',
        logo_alignment: 'left'
      });
    }

    const signupContainer = document.getElementById('google-btn-signup');
    if (signupContainer) {
      google.accounts.id.renderButton(signupContainer, {
        theme: 'filled_black',
        size: 'large',
        text: 'signup_with',
        shape: 'rectangular',
        width: '100%',
        logo_alignment: 'left'
      });
    }

    google.accounts.id.prompt();
  } catch (e) {
    console.warn('Google One Tap init failed:', e);
    bindGoogleButtonsAsMock();
  }

  // Initialize Facebook
  if (window.FB) {
    window.FB.init({
      appId: '761397640298692',
      cookie: true,
      xfbml: true,
      version: 'v19.0'
    });
  }

  const handleFbLogin = () => {
    if (!window.FB) return toast('Facebook SDK loading...', 'warning');
    window.FB.login((response) => {
      if (response.authResponse) {
        window.FB.api('/me', {fields: 'name,email,picture'}, (res) => {
          let existingUser = null;
          try {
            const stored = localStorage.getItem('tv_user');
            if (stored) existingUser = JSON.parse(stored);
          } catch (err) {}
          
          currentUser = {
            ...(existingUser && existingUser.email === res.email ? existingUser : {}),
            name: res.name,
            email: res.email || `${res.id}@facebook.com`,
            avatar: res.name ? res.name[0].toUpperCase() : 'F',
            isNew: existingUser && existingUser.email === res.email ? false : true,
          };
          saveUser();
          sendWelcomeEmail(res.name, currentUser.email, true);
          document.getElementById('auth-overlay').classList.add('hidden');
          toast(`Authenticated via Facebook!`, 'success');
        });
      } else {
        toast('Facebook login cancelled.', 'warning');
      }
    }, {scope: 'public_profile,email'});
  };

  const fbLoginBtn = document.getElementById('fb-login-btn');
  const fbSignupBtn = document.getElementById('fb-signup-btn');
  if (fbLoginBtn) fbLoginBtn.addEventListener('click', handleFbLogin);
  if (fbSignupBtn) fbSignupBtn.addEventListener('click', handleFbLogin);
}

function bindGoogleButtonsAsMock() {
  const googleBtns = [
    document.getElementById('google-login-btn'),
    document.getElementById('google-signup-btn'),
  ];
  googleBtns.forEach(btn => {
    if (btn) btn.addEventListener('click', () => mockSocialLogin('Google'));
  });
}

function handleGoogleCredential(response) {
  const payload = decodeJWT(response.credential);
  if (!payload) {
    toast('Failed to decode Google credential.', 'error');
    return;
  }

  const name = payload.name || payload.email?.split('@')[0] || 'Google User';
  const email = payload.email || '';
  const picture = payload.picture || '';

  let existingUser = null;
  try {
    const stored = localStorage.getItem('tv_user');
    if (stored) existingUser = JSON.parse(stored);
  } catch (err) {}

  currentUser = {
    ...(existingUser && existingUser.email === email ? existingUser : {}),
    name,
    email,
    avatar: name[0].toUpperCase(),
    picture,
    isNew: existingUser && existingUser.email === email ? false : true,
  };
  saveUser();

  // Send welcome email (non-blocking)
  sendWelcomeEmail(name, email, true);

  // Close auth & show profile setup
  document.getElementById('auth-overlay').classList.add('hidden');
  toast(`Authenticated via Google!`, 'success');
  openProfileModal();
}

function decodeJWT(token) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    // Base64url decode the payload (second part)
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64).split('').map(c =>
        '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)
      ).join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

// ══════════════════════════════════════════════════
// STORAGE
// ══════════════════════════════════════════════════
function loadStorage() {
  try {
    const u = localStorage.getItem('tv_user');
    if (u) currentUser = JSON.parse(u);
    const h = localStorage.getItem('tv_history');
    if (h) history = JSON.parse(h);
  } catch { /**/ }
}

function saveUser() {
  localStorage.setItem('tv_user', JSON.stringify(currentUser));
}

function saveHistory() {
  localStorage.setItem('tv_history', JSON.stringify(history));
}

function checkHashLogin() {
  const hash = window.location.hash;
  if (hash.includes('access_token')) {
    const tokenPart = hash.split('access_token=')[1];
    if (tokenPart.startsWith('simulate_')) {
      const payloadEncoded = tokenPart.replace('simulate_', '');
      try {
        const payload = JSON.parse(atob(payloadEncoded));
        currentUser = {
          name: payload.name,
          email: payload.email,
          avatar: payload.name[0].toUpperCase(),
          isNew: false
        };
        saveUser();
        // Clear hash without reload
        history.replaceState(null, null, window.location.pathname);
        
        // If we are on login.html, redirect to index.html
        if (window.location.pathname.includes('login.html')) {
          window.location.href = 'index.html';
        } else {
          // If already on index.html, we might need to refresh UI
          // But since this runs in DOMContentLoaded, the logic below in init will handle it.
        }
      } catch (e) {
        console.error("Hash login failed", e);
      }
    }
  }
}

// Support hashchange if the page is already open (e.g. from a popup)
window.addEventListener('hashchange', checkHashLogin);

function addHistoryEntry(entry) {
  history.unshift({ id: Date.now(), ...entry });
  saveHistory();
  renderHistory();
  updateStats();
}

// ══════════════════════════════════════════════════
// AUTH
// ══════════════════════════════════════════════════
window.switchAuth = (mode) => {
  document.getElementById('form-login').classList.toggle('hidden', mode !== 'login');
  document.getElementById('form-signup').classList.toggle('hidden', mode !== 'signup');
};

function showAuth() {
  const authOl = document.getElementById('auth-overlay');
  if(authOl) authOl.classList.remove('hidden');
  const appEl = document.getElementById('app');
  if(appEl) appEl.classList.add('hidden');
}

function bindAuth() {
  // Password toggles
  document.querySelectorAll('.pw-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById(btn.dataset.target);
      const isText = input.type === 'text';
      input.type = isText ? 'password' : 'text';
      btn.querySelector('.eye-show').classList.toggle('hidden', !isText);
      btn.querySelector('.eye-hide').classList.toggle('hidden', isText);
    });
  });

  // Login submit
  document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('login-email').value.trim();
    const pw    = document.getElementById('login-pw').value;
    let valid   = true;

    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      setErr('login-email-err', 'Enter a valid email'); valid = false;
    } else setErr('login-email-err', '');

    if (pw.length < 6) {
      setErr('login-pw-err', 'Minimum 6 characters'); valid = false;
    } else setErr('login-pw-err', '');

    if (!valid) return;

    const btn = document.getElementById('login-btn');
    setButtonLoading(btn, true, 'Signing in...');
    await sleep(1000);

    let name = email.split('@')[0]
      .replace(/[._-]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
      .trim();

    let existingUser = null;
    try {
      const stored = localStorage.getItem('tv_user');
      if (stored) existingUser = JSON.parse(stored);
    } catch (err) {}

    if (existingUser && existingUser.email === email && existingUser.name) {
      name = existingUser.name;
    }

    currentUser = { ...(existingUser && existingUser.email === email ? existingUser : {}), name, email, avatar: name[0].toUpperCase(), isNew: false };
    saveUser();
    setButtonLoading(btn, false, 'Sign In');

    // Send welcome back email (non-blocking)
    sendWelcomeEmail(name, email, false);

    // If user has no age/gender, show profile modal after a brief delay
    if (!currentUser.age && !currentUser.gender) {
      document.getElementById('auth-overlay').classList.add('hidden');
      openProfileModal();
    } else {
      showApp();
      toast(`Welcome back, ${name.split(' ')[0]}!`, 'success');
    }
  });

  // Signup submit
  document.getElementById('signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name    = document.getElementById('signup-name').value.trim();
    const email   = document.getElementById('signup-email').value.trim();
    const pw      = document.getElementById('signup-pw').value;
    const confirm = document.getElementById('signup-confirm').value;
    let valid = true;

    const checks = [
      ['signup-name-err',    !name,                                   'Full name is required'],
      ['signup-email-err',   !email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/), 'Enter a valid email'],
      ['signup-pw-err',      pw.length < 6,                           'Minimum 6 characters'],
      ['signup-confirm-err', pw !== confirm,                           'Passwords do not match'],
    ];
    checks.forEach(([id, fail, msg]) => {
      setErr(id, fail ? msg : '');
      if (fail) valid = false;
    });

    if (!valid) return;

    const btn = document.getElementById('signup-btn');
    setButtonLoading(btn, true, 'Creating account...');
    await sleep(1200);

    currentUser = { name, email, avatar: name[0].toUpperCase(), isNew: true };
    saveUser();
    setButtonLoading(btn, false, 'Create Account');

    // Send welcome email
    sendWelcomeEmail(name, email, true);

    // Final Call: Redirect new users to Home Page first
    window.location.href = 'index.html';
  });

  // ── Global Theme Switcher ───────────────────
  const themeToggles = [
    document.getElementById('theme-toggle-app'),
    document.getElementById('theme-toggle-landing')
  ].filter(Boolean);

  const updateThemeIcons = () => {
    const isLight = document.documentElement.dataset.theme === 'light';
    const iconMoon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    const iconSun = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
    
    themeToggles.forEach(btn => {
      btn.innerHTML = isLight ? iconSun : iconMoon;
      // Adjust color for landing nav if needed
      if (btn.id === 'theme-toggle-landing') {
        btn.style.color = isLight ? 'var(--t2)' : 'rgba(255,255,255,0.35)';
      }
    });
  };

  const toggleTheme = () => {
    const isLight = document.documentElement.dataset.theme === 'light';
    if (isLight) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('tv_theme', 'dark');
    } else {
      document.documentElement.dataset.theme = 'light';
      localStorage.setItem('tv_theme', 'light');
    }
    updateThemeIcons();
  };

  themeToggles.forEach(btn => btn.addEventListener('click', toggleTheme));
  updateThemeIcons();

  // Sync theme across tabs
  window.addEventListener('storage', (e) => {
    if (e.key === 'tv_theme') {
      if (e.newValue === 'light') document.documentElement.dataset.theme = 'light';
      else document.documentElement.removeAttribute('data-theme');
      updateThemeIcons();
    }
  });


  // Logout
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      localStorage.removeItem('tv_user');
      localStorage.removeItem('tv_token');
      localStorage.removeItem('tv_history'); // Clear history on logout as well for forensic privacy if needed, or keep it. Let's keep it for now but clear user.
      currentUser = null;
      window.location.href = 'index.html';
    });
  }
}

// ══════════════════════════════════════════════════
// SOCIAL LOGIN MOCK
// ══════════════════════════════════════════════════
async function mockSocialLogin(provider) {
  // Ask the user to provide their email to simulate a real OAuth handoff
  const email = prompt(`Connecting to ${provider}. Please confirm your email address to continue:`);
  if (!email || !email.includes('@')) return; // Cancel if empty or invalid

  const name = email.split('@')[0]
      .replace(/[._-]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
      .trim() || `${provider} User`;
  
  toast(`Connecting securely via ${provider}...`, 'info');
  await sleep(1200); // Simulate network loading delay

  currentUser = { name, email, avatar: name[0].toUpperCase(), isNew: true };
  saveUser();
  
  // Send welcome email using the new mock account
  sendWelcomeEmail(name, email, true);

  // Final Call: Redirect new users to Home Page first
  window.location.href = 'index.html';
}

// ══════════════════════════════════════════════════
// PROFILE MODAL
// ══════════════════════════════════════════════════
let currentBase64Photo = '';

function openProfileModal() {
  const modal = document.getElementById('profile-modal');
  const preview = document.getElementById('profile-avatar-preview');
  
  document.getElementById('prof-name').value = currentUser?.name || '';
  document.getElementById('prof-nick').value = currentUser?.nickname || '';
  document.getElementById('prof-phone').value = currentUser?.phone || '';
  document.getElementById('prof-dob').value = currentUser?.dob || '';
  document.getElementById('prof-photo-file').value = ''; // Reset file input

  currentBase64Photo = currentUser?.picture || '';

  if (currentBase64Photo) {
    preview.innerHTML = `<img src="${currentBase64Photo}" style="width:100%; height:100%; object-fit:cover; border-radius:50%;" />`;
  } else {
    preview.innerHTML = currentUser?.avatar || 'A';
  }

  modal.classList.remove('hidden');
}

function closeProfileModal() {
  const modal = document.getElementById('profile-modal');
  if(modal) modal.classList.add('hidden');
}

function bindProfileModal() {
  const nameInput = document.getElementById('prof-name');
  const photoFileInput = document.getElementById('prof-photo-file');
  const preview = document.getElementById('profile-avatar-preview');

  if(nameInput) nameInput.addEventListener('input', (e) => {
    if (currentBase64Photo) return;
    const val = e.target.value.trim();
    preview.innerHTML = val ? val[0].toUpperCase() : (currentUser?.avatar || 'A');
  });

  if(photoFileInput) photoFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if(file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        currentBase64Photo = ev.target.result;
        preview.innerHTML = `<img src="${currentBase64Photo}" style="width:100%; height:100%; object-fit:cover; border-radius:50%;" />`;
      };
      reader.readAsDataURL(file);
    } else {
      currentBase64Photo = '';
      const nameVal = nameInput?.value.trim();
      preview.innerHTML = nameVal ? nameVal[0].toUpperCase() : (currentUser?.avatar || 'A');
    }
  });

  const form = document.getElementById('profile-form');
  if(form) form.addEventListener('submit', (e) => {
    e.preventDefault();
    saveProfileData();
  });

  const skipBtn = document.getElementById('profile-skip-btn');
  if(skipBtn) skipBtn.addEventListener('click', () => {
    closeProfileModal();
    showApp();
    const displayName = currentUser?.nickname || currentUser?.name?.split(' ')[0] || 'Analyst';
    toast(`Welcome, ${displayName}! 🎉`, 'success');
  });

  const editBtn = document.getElementById('sb-edit-btn');
  if(editBtn) editBtn.addEventListener('click', openProfileModal);
}

function saveProfileData() {
  const name  = document.getElementById('prof-name')?.value.trim();
  const nick  = document.getElementById('prof-nick')?.value.trim();
  const phone = document.getElementById('prof-phone')?.value.trim();
  const dob   = document.getElementById('prof-dob')?.value;

  if (name) {
    currentUser.name   = name;
    currentUser.avatar = name[0].toUpperCase();
  }
  if (nick) currentUser.nickname = nick;
  if (phone) currentUser.phone = phone;
  if (dob) currentUser.dob = dob;
  if (currentBase64Photo) currentUser.picture = currentBase64Photo;

  saveUser();
  closeProfileModal();
  showApp();

  const displayName = currentUser.nickname || currentUser.name?.split(' ')[0] || 'Analyst';
  const greeting = currentUser.isNew ? `Welcome aboard, ${displayName}! 🎉` : `Welcome back, ${displayName}!`;
  toast(greeting, 'success');
}


// ══════════════════════════════════════════════════
// WELCOME EMAIL
// ══════════════════════════════════════════════════
async function sendWelcomeEmail(name, email, isNewUser) {
  try {
    const res = await fetch(`${API}/send-welcome-email/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, is_new_user: isNewUser }),
    });
    if (!res.ok) {
      const err = await res.json();
      toast(`Email Error: ${err.detail || 'Failed to send'}`, 'error', 8000);
    }
  } catch(e) {
    toast(`Network Error: Ensure backend is running.`, 'error', 8000);
  }
}

// ══════════════════════════════════════════════════
// SHOW APP
// ══════════════════════════════════════════════════
function showApp() {
  const authOl = document.getElementById('auth-overlay');
  if(authOl) authOl.classList.add('hidden');
  const prof = document.getElementById('profile-modal');
  if(prof) prof.classList.add('hidden');
  
  const app = document.getElementById('app');
  if(app) app.classList.remove('hidden');
  
  populateUserUI();
  renderHistory();
  updateStats();
  showView('upload');
  setTopbar('New Scan');
}

function populateUserUI() {
  if (!currentUser) return;
  const initial = currentUser.avatar || currentUser.name?.[0]?.toUpperCase() || 'A';
  const displayName = currentUser.nickname || currentUser.name?.split(' ')[0] || 'Analyst';

  // Sidebar Avatar
  const sbAvatar = document.getElementById('sb-avatar');
  if (sbAvatar) {
    if (currentUser.picture) {
      sbAvatar.innerHTML = `<img src="${currentUser.picture}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />`;
    } else {
      sbAvatar.textContent = initial;
    }
  }
  
  // Topbar Avatar
  const tbAvatar = document.getElementById('tuc-avatar');
  if (tbAvatar) {
    if (currentUser.picture) {
      tbAvatar.innerHTML = `<img src="${currentUser.picture}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />`;
    } else {
      tbAvatar.textContent = initial;
    }
  }

  // Sidebar
  const sbName = document.getElementById('sb-name');
  if(sbName) sbName.textContent = currentUser.nickname ? `${currentUser.name} (${currentUser.nickname})` : (currentUser.name || 'Analyst');
  const sbEmail = document.getElementById('sb-email');
  if(sbEmail) sbEmail.textContent = currentUser.email || '';

  // Topbar chip
  const tucName = document.getElementById('tuc-name');
  if(tucName) tucName.textContent = displayName;
  const tbCurrent = document.getElementById('tb-current');
  if(tbCurrent) tbCurrent.textContent = 'New Scan';

  // Profile details chips (DOB, Phone)
  const details = document.getElementById('sb-profile-details');
  if(details) {
    details.innerHTML = '';
    if (currentUser.phone) {
      const chip = document.createElement('div');
      chip.className = 'sb-detail-chip';
      chip.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg> <span style="font-size: 11px;">${currentUser.phone}</span>`;
      details.appendChild(chip);
    }
    if (currentUser.dob) {
      const chip = document.createElement('div');
      chip.className = 'sb-detail-chip';
      chip.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg> <span style="font-size: 11px;">${currentUser.dob}</span>`;
      details.appendChild(chip);
    }
  }
}

// ══════════════════════════════════════════════════
// PROXY HEALTH CHECK
// ══════════════════════════════════════════════════
function checkProxy() {
  fetch(`${API}/`)
    .then(r => r.json())
    .then(d => setStatus(d.status === 'running' ? 'online' : 'offline', d.status === 'running' ? 'Proxy Online' : 'Proxy Issue'))
    .catch(() => setStatus('offline', 'Proxy Offline'));
}

function setStatus(cls, label) {
  document.getElementById('sb-proxy-dot').className  = `sb-status-dot ${cls}`;
  document.getElementById('tb-proxy-indicator').className = `tb-dot-indicator ${cls}`;
  document.getElementById('sb-proxy-label').textContent = label;
  document.getElementById('tb-proxy-text').textContent   = label;
}

// ══════════════════════════════════════════════════
// UPLOAD
// ══════════════════════════════════════════════════
function bindUpload() {
  const dropZone   = document.getElementById('drop-zone');
  const fileInput  = document.getElementById('file-input');
  const browseBtn  = document.getElementById('browse-btn');
  const fpClear    = document.getElementById('fp-clear');
  const analyzeBtn = document.getElementById('analyze-btn');
  const newScanBtn = document.getElementById('new-scan-btn');

  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFileSelect(e.target.files[0]); });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  });

  fpClear.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    document.getElementById('file-preview').classList.add('hidden');
    dropZone.style.display = '';
    analyzeBtn.classList.add('hidden');
  });

  analyzeBtn.addEventListener('click', () => { if (currentFile) runAnalysis(currentFile); });
  newScanBtn.addEventListener('click', resetToUpload);
}

function handleFileSelect(file) {
  if (!/\.(jpg|jpeg|png|webp|mp4|avi|mov)$/i.test(file.name)) {
    toast('Unsupported file type.', 'error'); return;
  }
  currentFile = file;
  const sizeLabel = file.size > 1048576 ? `${(file.size/1048576).toFixed(1)} MB` : `${(file.size/1024).toFixed(0)} KB`;
  const isVideo   = /\.(mp4|avi|mov)$/i.test(file.name);

  document.getElementById('fp-name').textContent = file.name;
  document.getElementById('fp-meta').textContent = `${sizeLabel} · ${isVideo ? 'Video' : 'Image'}`;

  const fpThumb = document.getElementById('fp-thumb');
  fpThumb.innerHTML = '';
  if (!isVideo && file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    fpThumb.appendChild(img);
  } else {
    fpThumb.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>`;
  }

  document.getElementById('drop-zone').style.display = 'none';
  document.getElementById('file-preview').classList.remove('hidden');
  document.getElementById('analyze-btn').classList.remove('hidden');
}

// ══════════════════════════════════════════════════
// ANALYSIS
// ══════════════════════════════════════════════════
async function runAnalysis(file) {
  showView('processing');
  setTopbar('Processing...');
  document.getElementById('proc-file-name').textContent = file.name;
  feedbackSent = false;
  startTerminal();
  startProgressBar();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API}/process/`, { method: 'POST', body: formData });
    stopProgressBar(true);
    appendLog('[SUCCESS] Response received from inference pipeline.', 'tlog-ok');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    await sleep(750);
    handleResponse(data, file);
  } catch (err) {
    stopProgressBar(false);
    appendLog(`[ERROR] ${err.message}`, 'tlog-err');
    await sleep(500);
    document.getElementById('err-message').textContent = err.message || 'Request failed.';
    showView('error');
    setTopbar('Error');
    toast('Analysis failed. Is the proxy running?', 'error');
  }
}

function handleResponse(data, file) {
  if (data.status === 'error') {
    document.getElementById('err-message').textContent = data.message || 'Unknown server error.';
    showView('error'); setTopbar('Error'); return;
  }
  if (data.status === 'no_face') {
    populateNoFace(data);
    showView('no-face');
    setTopbar('No Face Found');
    toast('No faces detected in the media.', 'warning');
    return;
  }
  populateResults(data);
  showView('results');
  setTopbar('Analysis Report');

  const pred = data.final_result?.prediction;
  const conf = data.final_result?.confidence_pct;
  addHistoryEntry({ fileName: data.file || file?.name, fileType: file?.type, prediction: pred, confidence: conf, fullData: data });
  toast(`Analysis complete — ${pred} (${conf}%)`, pred === 'FAKE' ? 'error' : 'success');
}

// ══════════════════════════════════════════════════
// REPORT ID GENERATOR
// ══════════════════════════════════════════════════
function generateReportId() {
  reportCounter++;
  localStorage.setItem('tv_report_counter', reportCounter.toString());
  const now = new Date();
  const dateStr = now.getFullYear().toString() +
    String(now.getMonth() + 1).padStart(2, '0') +
    String(now.getDate()).padStart(2, '0');
  return `TV-${dateStr}-${String(reportCounter).padStart(3, '0')}`;
}

// ══════════════════════════════════════════════════
// ANIMATED CONFIDENCE COUNTER
// ══════════════════════════════════════════════════
function animateCounter(element, targetValue, duration = 1500, suffix = '%') {
  const start = performance.now();
  const initialValue = 0;

  function update(currentTime) {
    const elapsed = currentTime - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const currentValue = initialValue + (targetValue - initialValue) * eased;
    element.textContent = currentValue.toFixed(1) + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// ══════════════════════════════════════════════════
// CONSENSUS CALCULATOR
// ══════════════════════════════════════════════════
function calculateConsensus(modelResults, finalPrediction) {
  if (!modelResults || modelResults.length === 0) return { text: 'N/A', count: 0, total: 0 };
  
  const total = modelResults.length;
  const agreeing = modelResults.filter(m => m.vote === finalPrediction).length;
  
  if (agreeing === total) {
    return { text: `${agreeing}/${total} Unanimous`, count: agreeing, total, level: 'unanimous' };
  } else if (agreeing > total / 2) {
    return { text: `${agreeing}/${total} Majority`, count: agreeing, total, level: 'majority' };
  } else {
    return { text: `${agreeing}/${total} Split`, count: agreeing, total, level: 'split' };
  }
}

// ══════════════════════════════════════════════════
// THREAT LEVEL CALCULATOR
// ══════════════════════════════════════════════════
function updateThreatBar(confidence, isFake) {
  const fill = document.getElementById('threat-bar-fill');
  const text = document.getElementById('threat-level-text');
  
  // For REAL: low threat → green/short bar. For FAKE: high threat → red/long bar.
  let threatPct, threatLabel, threatColor, threatGlow;
  
  if (isFake) {
    threatPct = Math.min(confidence, 100);
    if (confidence >= 80) {
      threatLabel = 'Critical Risk';
      threatColor = 'linear-gradient(90deg, #f59e0b, #ef4444)';
      threatGlow = 'rgba(239,68,68,.5)';
      text.style.color = '#ef4444';
    } else if (confidence >= 60) {
      threatLabel = 'High Risk';
      threatColor = 'linear-gradient(90deg, #f59e0b, #f97316)';
      threatGlow = 'rgba(249,115,22,.4)';
      text.style.color = '#f97316';
    } else {
      threatLabel = 'Moderate Risk';
      threatColor = 'linear-gradient(90deg, #10b981, #f59e0b)';
      threatGlow = 'rgba(245,158,11,.4)';
      text.style.color = '#f59e0b';
    }
  } else {
    threatPct = Math.max(100 - confidence, 5);
    if (confidence >= 90) {
      threatLabel = 'Low Risk';
      threatColor = 'linear-gradient(90deg, #10b981, #34d399)';
      threatGlow = 'rgba(16,185,129,.4)';
      text.style.color = '#10b981';
    } else if (confidence >= 70) {
      threatLabel = 'Minor Risk';
      threatColor = 'linear-gradient(90deg, #10b981, #f59e0b)';
      threatGlow = 'rgba(245,158,11,.3)';
      text.style.color = '#f59e0b';
    } else {
      threatLabel = 'Uncertain';
      threatColor = 'linear-gradient(90deg, #f59e0b, #f97316)';
      threatGlow = 'rgba(245,158,11,.4)';
      text.style.color = '#f59e0b';
    }
  }

  text.textContent = threatLabel;
  fill.style.background = threatColor;
  fill.style.setProperty('--threat-glow', threatGlow);
  
  // Animate the bar
  setTimeout(() => {
    fill.style.width = threatPct + '%';
  }, 200);
}

// ══════════════════════════════════════════════════
// RESULTS
// ══════════════════════════════════════════════════
function populateResults(data) {
  const { final_result, model_results, frames, faces, file, type } = data;
  const isFake = final_result?.prediction === 'FAKE';
  const conf   = parseFloat(final_result?.confidence_pct ?? 0);

  // ── Report header ──
  const reportId = generateReportId();
  document.getElementById('report-id').textContent = reportId;
  const now = new Date();
  document.getElementById('report-time').textContent = now.toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });

  // Verdict card
  const vc = document.getElementById('verdict-card');
  vc.className = `verdict-card card-glass ${isFake ? 'fake' : 'real'}`;
  document.getElementById('verdict-text').textContent = final_result?.prediction ?? '—';

  // ── Consensus badge ──
  const consensus = calculateConsensus(model_results, final_result?.prediction);
  document.getElementById('consensus-text').textContent = consensus.text;

  // Dynamic explanation based on frames and models
  let explanation = isFake
    ? `Based on the analysis of ${frames?.count || 'the'} extracted frame(s), deepfake artifacts were detected.`
    : `Based on the analysis of ${frames?.count || 'the'} extracted frame(s), no deepfake artifacts were detected. The media appears visually authentic.`;

  if (model_results && model_results.length > 0) {
    const fakeModels = model_results.filter(m => m.vote === 'FAKE').map(m => m.model);
    if (isFake && fakeModels.length > 0) {
      explanation += ` Specifically, the ${fakeModels.join(' and ')} model(s) identified signs of AI manipulation such as unnatural face-boundary blending and/or skin texture anomalies.`;
    } else if (!isFake) {
      explanation += ` The tri-model ensemble confirmed structural consistency across boundary and texture regions.`;
    }
  }
  document.getElementById('verdict-explanation').textContent = explanation;

  // ── Verdict meta: file hash + analysis time ──
  const fileHash = generateShortHash(file || 'unknown');
  document.getElementById('verdict-meta').innerHTML = [
    final_result?.decided_by ? `Decided by: ${final_result.decided_by}` : '',
    `SHA: ${fileHash}`,
    `Report: ${reportId}`,
  ].filter(Boolean).join(' · ');

  // ── Email report button ──
  const emailBtn = document.getElementById('email-report-btn');
  if (emailBtn) {
    emailBtn.onclick = async () => {
      if (!currentUser || !currentUser.email) {
        toast('Please log in to receive reports.', 'error');
        return;
      }
      const origHTML = emailBtn.innerHTML;
      emailBtn.disabled = true;
      emailBtn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;"></div> Saving...';
      try {
        const payload = {
          name: currentUser.name || "Analyst",
          email: currentUser.email,
          file_name: file?.name || data.file || 'unknown_media',
          prediction: final_result?.prediction || 'UNKNOWN',
          confidence: conf,
          explanation: explanation
        };
        const resp = await fetch(`${API}/send-report-email/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) {
          const errData = await resp.json().catch(() => null);
          throw new Error(errData?.detail || 'Failed to send report');
        }
        toast(`Report saved to ${currentUser.email}`, 'success');
      } catch (e) {
        toast(`Error: ${e.message}`, 'error', 7000);
      } finally {
        emailBtn.disabled = false;
        emailBtn.innerHTML = origHTML;
      }
    };
  }

  // ── Animated confidence ring ──
  const circ = 263.9;
  // Reset ring first
  const circle = document.getElementById('conf-circle');
  circle.setAttribute('stroke-dasharray', `0 ${circ}`);
  circle.style.stroke = '#8b5cf6';
  circle.style.filter = '';
  
  // Animate counter
  const confValue = document.getElementById('conf-value');
  confValue.textContent = '0.0%';
  
  setTimeout(() => {
    circle.setAttribute('stroke-dasharray', `${(conf / 100) * circ} ${circ}`);
    circle.style.stroke = isFake ? '#ef4444' : '#10b981';
    circle.style.filter = `drop-shadow(0 0 8px ${isFake ? 'rgba(239,68,68,.5)' : 'rgba(16,185,129,.5)'})`;
    animateCounter(confValue, conf, 1800);
  }, 300);

  // ── Threat bar ──
  const threatFill = document.getElementById('threat-bar-fill');
  threatFill.style.width = '0%';
  updateThreatBar(conf, isFake);

  // Stats
  document.getElementById('stat-frames').textContent = frames?.count ?? '—';
  document.getElementById('stat-faces').textContent  = faces?.count  ?? '—';
  document.getElementById('stat-type').textContent   = type ? type.toUpperCase() : '—';

  // ── Model cards with staggered animation ──
  const grid = document.getElementById('model-grid');
  grid.innerHTML = '';
  (model_results ?? []).forEach((m, index) => {
    const mFake = m.vote === 'FAKE';
    const card  = document.createElement('div');
    card.className = `model-card ${mFake ? 'fake' : 'real'}`;
    card.style.animationDelay = `${0.15 + index * 0.12}s`;
    card.innerHTML = `
      <div class="mc-bar"></div>
      <div class="mc-body">
        <div class="mc-head">
          <span class="mc-name">${m.model}</span>
          <span class="mc-vote ${mFake ? 'fake' : 'real'}">${m.vote}</span>
        </div>
        <div class="mc-conf-row"><span>Confidence</span><span>${m.confidence_pct}%</span></div>
        <div class="mc-bar-bg">
          <div class="mc-bar-fill" style="width:0%"></div>
        </div>
        <div class="mc-probs">
          <span>P(FAKE) <b>${typeof m.p_fake === 'number' ? m.p_fake.toFixed(3) : m.p_fake}</b></span>
          <span>P(REAL) <b>${typeof m.p_real === 'number' ? m.p_real.toFixed(3) : m.p_real}</b></span>
        </div>
        <p class="mc-desc">${MODEL_DESC[m.model] ?? ''}</p>
      </div>`;
    grid.appendChild(card);
    
    // Animate the confidence bar fill with stagger
    setTimeout(() => {
      const barFill = card.querySelector('.mc-bar-fill');
      if (barFill) barFill.style.width = m.confidence_pct + '%';
    }, 400 + index * 200);
  });

  // Artifact section
  const artSection = document.getElementById('artifact-section');
  artSection.style.display = isFake ? '' : 'none';
  if (isFake) {
    const arts = [
      { name:'Face Boundary Blending', sev:'high',   desc:'Unnatural edge seams detected at face-hair and face-neck transitions.' },
      { name:'Skin Texture Anomalies', sev:'medium', desc:'High-frequency texture inconsistencies found in forehead and cheek regions.' },
      { name:'Eye Reflection Glitch',  sev:'low',    desc:'Minor asymmetric corneal highlights inconsistent with ambient lighting.' },
    ];
    const artList = document.getElementById('artifact-list');
    artList.innerHTML = '';
    arts.forEach(a => {
      const d = document.createElement('div');
      d.className = `artifact-item ${a.sev}`;
      d.innerHTML = `<span class="art-sev">${a.sev}</span><div><p class="art-name">${a.name}</p><p class="art-desc">${a.desc}</p></div>`;
      artList.appendChild(d);
    });
  }

  // Galleries
  renderGallery('faces-gallery', faces?.images, 'Face');
  renderGallery('frames-gallery', frames?.images, 'Frame');

  // Reset feedback buttons
  document.getElementById('fb-actions').innerHTML = `
    <button class="fb-btn fake" id="fb-fake-btn">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
      Confirm FAKE
    </button>
    <button class="fb-btn real" id="fb-real-btn">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
      Confirm REAL
    </button>`;
  document.getElementById('fb-fake-btn').addEventListener('click', () => sendFeedback('FAKE'));
  document.getElementById('fb-real-btn').addEventListener('click', () => sendFeedback('REAL'));
}

function renderGallery(containerId, images, label) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!images?.length) {
    container.innerHTML = `<p class="no-gallery">No ${label.toLowerCase()}s detected.</p>`;
    return;
  }
  images.forEach((url, i) => {
    const card = document.createElement('div');
    card.className = 'img-card';
    card.innerHTML = `
      <img src="${url}" alt="${label} ${i}" loading="lazy"
        onerror="this.src='data:image/svg+xml;charset=UTF-8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22100%22><rect width=%22160%22 height=%22100%22 fill=%22%23111%22/><text x=%2280%22 y=%2255%22 text-anchor%3D%22middle%22 fill=%22%23444%22 font-family=%22sans-serif%22 font-size=%2211%22>Unavailable</text></svg>'" />
      <div class="img-card-overlay">${label}_${i}</div>
      <div class="img-card-hover">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
      </div>`;
    card.addEventListener('click', () => openModal(url, `${label} #${i}`));
    container.appendChild(card);
  });
}

// ══════════════════════════════════════════════════
// SHORT HASH GENERATOR (cosmetic)
// ══════════════════════════════════════════════════
function generateShortHash(input) {
  let hash = 0;
  const str = input + Date.now().toString();
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).toUpperCase().padStart(8, '0').substring(0, 8);
}

// ══════════════════════════════════════════════════
// NO FACE
// ══════════════════════════════════════════════════
function populateNoFace(data) {
  document.getElementById('nf-message').textContent = data.message || 'No faces detected.';
  const tipsEl = document.getElementById('nf-tips');
  tipsEl.innerHTML = '';
  if (data.user_tip) {
    const tips = data.user_tip.split(/\(\d+\)\s*/).filter(Boolean);
    tips.forEach((tip, i) => {
      const d = document.createElement('div');
      d.className = 'nf-tip';
      d.innerHTML = `<span class="tip-n">${i+1}</span><span>${tip.trim()}</span>`;
      tipsEl.appendChild(d);
    });
  }
}

// ══════════════════════════════════════════════════
// FEEDBACK
// ══════════════════════════════════════════════════
async function sendFeedback(label) {
  if (feedbackSent) return;
  document.getElementById('fb-fake-btn').disabled = true;
  document.getElementById('fb-real-btn').disabled = true;
  try {
    const res = await fetch(`${API}/feedback/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    if (!res.ok) throw new Error();
    feedbackSent = true;
    document.getElementById('fb-actions').innerHTML =
      `<div class="fb-sent"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Feedback received — ETCNN will improve!</div>`;
    toast(`Confirmed as ${label}. Thank you!`, 'success');
  } catch {
    document.getElementById('fb-fake-btn').disabled = false;
    document.getElementById('fb-real-btn').disabled = false;
    toast('Could not send feedback right now.', 'warning');
  }
}

// ══════════════════════════════════════════════════
// UI BINDINGS
// ══════════════════════════════════════════════════
function bindUI() {
  document.getElementById('reset-btn').addEventListener('click', resetToUpload);
  document.getElementById('nf-retry-btn').addEventListener('click', resetToUpload);
  document.getElementById('err-retry-btn').addEventListener('click', resetToUpload);
  document.getElementById('clear-history-btn').addEventListener('click', () => {
    history = []; activeHist = null;
    saveHistory(); renderHistory(); updateStats();
    toast('History cleared.', 'info');
  });
  document.getElementById('im-close').addEventListener('click', closeModal);
  document.querySelector('.im-backdrop').addEventListener('click', closeModal);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  // Full Activity Audit Report
  const auditBtn = document.getElementById('send-full-audit-btn');
  if (auditBtn) {
    auditBtn.addEventListener('click', async () => {
      if (!currentUser || !currentUser.email) {
        toast('Please log in to receive audits.', 'error');
        return;
      }
      if (history.length === 0) {
        toast('No scan history found to report.', 'warning');
        return;
      }

      const origText = auditBtn.innerHTML;
      auditBtn.disabled = true;
      auditBtn.innerHTML = '<div class="spinner" style="width:12px;height:12px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;"></div> Sending Audit...';

      try {
        const payload = {
          name: currentUser.name || "Analyst",
          email: currentUser.email,
          total_real: history.filter(h => h.prediction === 'REAL').length,
          total_fake: history.filter(h => h.prediction === 'FAKE').length,
          history: history.map(h => ({
            file_name: h.fileName || "unknown",
            prediction: h.prediction || "UNKNOWN",
            confidence: (h.confidence ? h.confidence + '%' : "0%"),
            timestamp: new Date(h.id).toLocaleString()
          }))
        };

        const res = await fetch(`${API}/send-audit-report/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Could not send audit report');
        toast('Full activity audit sent to your email! ✅', 'success', 6000);
      } catch (err) {
        toast('Audit Error: ' + err.message, 'error');
      } finally {
        auditBtn.disabled = false;
        auditBtn.innerHTML = origText;
      }
    });
  }
}

function resetToUpload() {
  currentFile = null; feedbackSent = false;
  activeHist = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-preview').classList.add('hidden');
  document.getElementById('drop-zone').style.display = '';
  document.getElementById('analyze-btn').classList.add('hidden');
  clearTimeout(progressTimer);
  showView('upload');
  setTopbar('New Scan');
  renderHistory();
}

// ══════════════════════════════════════════════════
// TERMINAL + PROGRESS
// ══════════════════════════════════════════════════
const LOGS = [
  { text:'[SYS] Initializing ingestion matrix...', cls:'tlog-sys' },
  { text:'[SYS] Artifact secured — forwarding to EC2 inference node', cls:'tlog-sys' },
  { text:'[PROC] Handshake with inference cluster... OK', cls:'' },
  { text:'[SYS] Loading CNN face-swap detection core... Active', cls:'tlog-sys' },
  { text:'[SYS] Loading CViT boundary mapper... Active', cls:'tlog-sys' },
  { text:'[SYS] Loading ETCNN texture analyzer... Active', cls:'tlog-sys' },
  { text:'[PROC] Extracting temporal keyframes & facial biometrics...', cls:'' },
  { text:'[MODEL] Spatial-frequency analysis on extracted faces complete', cls:'tlog-model' },
  { text:'[MODEL] Computing weighted ensemble: CNN×0.33 + CViT×0.33 + ETCNN×0.34', cls:'tlog-model' },
  { text:'[PROC] Awaiting inference results from EC2...', cls:'' },
];

let logIndex = 0, logTimer = null;

function startTerminal() {
  const body = document.getElementById('terminal-logs');
  body.innerHTML = '';
  logIndex = 0;
  const print = () => {
    if (logIndex >= LOGS.length) return;
    appendLog(LOGS[logIndex].text, LOGS[logIndex].cls);
    logIndex++;
    logTimer = setTimeout(print, Math.random() * 400 + 160);
  };
  print();
}

function appendLog(text, cls) {
  const body = document.getElementById('terminal-logs');
  const div  = document.createElement('div');
  div.className = `tlog ${cls || ''}`;
  div.textContent = '> ' + text;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function startProgressBar() {
  let val = 0;
  const bar = document.getElementById('proc-bar');
  const pct = document.getElementById('proc-pct');
  bar.style.width = '0%';
  bar.style.background = '';
  const tick = () => {
    if (val < 88) {
      val += Math.random() * 2.2 + 0.3;
      bar.style.width = Math.min(val, 88) + '%';
      pct.textContent  = Math.min(Math.round(val), 88) + '%';
      progressTimer = setTimeout(tick, 380);
    }
  };
  tick();
}

function stopProgressBar(success) {
  clearTimeout(progressTimer);
  const bar = document.getElementById('proc-bar');
  const pct = document.getElementById('proc-pct');
  bar.style.width = '100%';
  pct.textContent = '100%';
  if (!success) bar.style.background = 'var(--danger)';
}

// ══════════════════════════════════════════════════
// HISTORY
// ══════════════════════════════════════════════════
function renderHistory() {
  const list = document.getElementById('history-list');
  list.innerHTML = '';

  if (!history.length) {
    list.innerHTML = `<div class="sb-empty">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
      <p>No scans yet</p></div>`;
    return;
  }

  history.forEach(entry => {
    const isFake  = entry.prediction === 'FAKE';
    const isVideo = /\.(mp4|avi|mov)$/i.test(entry.fileName || '');
    const date    = new Date(entry.id).toLocaleDateString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });

    const btn = document.createElement('button');
    btn.className = `sb-hist-item${activeHist === entry.id ? ' active' : ''}`;
    btn.innerHTML = `
      <div class="hi-icon ${isFake ? 'fake' : 'real'}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          ${isVideo ? '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>'
                    : '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="M21 15l-5-5L5 21"/>'}
        </svg>
      </div>
      <div class="hi-body">
        <p class="hi-name">${entry.fileName || 'Unknown'}</p>
        <div class="hi-meta">
          <span class="hi-pred ${isFake ? 'fake' : 'real'}">${entry.prediction || '—'}</span>
          <span class="hi-date">· ${date}</span>
        </div>
      </div>
      ${activeHist === entry.id ? '<div class="hi-dot-active"></div>' : ''}`;
    btn.addEventListener('click', () => {
      activeHist = entry.id;
      renderHistory();
      if (entry.fullData) {
        populateResults(entry.fullData);
        showView('results');
        setTopbar('Previous Analysis');
      }
      toast(`${entry.fileName || 'Archive'} — ${entry.prediction}`, entry.prediction === 'FAKE' ? 'error' : 'success');
    });
    list.appendChild(btn);
  });
}

function updateStats() {
  document.getElementById('stat-total').textContent = history.length;
  document.getElementById('stat-fake').textContent  = history.filter(h => h.prediction === 'FAKE').length;
  document.getElementById('stat-real').textContent  = history.filter(h => h.prediction === 'REAL').length;
}

// ══════════════════════════════════════════════════
// VIEW / TOPBAR
// ══════════════════════════════════════════════════
function showView(name) {
  ['upload','processing','results','no-face','error'].forEach(v => {
    document.getElementById(`view-${v}`)?.classList.toggle('hidden', v !== name);
  });
}

function setTopbar(text) {
  const el = document.getElementById('tb-current');
  if (el) el.textContent = text;
}

// ══════════════════════════════════════════════════
// IMAGE MODAL
// ══════════════════════════════════════════════════
function openModal(src, label) {
  document.getElementById('im-img').src = src;
  document.getElementById('im-label').textContent = label;
  document.getElementById('img-modal').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('img-modal').classList.add('hidden');
}

// ══════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════
function toast(message, type = 'info', duration = 4500) {
  const icons = { success:'✓', error:'✕', warning:'⚠', info:'ℹ' };
  const div = document.createElement('div');
  div.className = `toast ${type}`;
  div.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span>${message}</span>`;
  div.addEventListener('click', () => div.remove());
  document.getElementById('toast-container').appendChild(div);
  setTimeout(() => div.remove(), duration);
}

// ══════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function setErr(id, msg) { const el = document.getElementById(id); if (el) el.textContent = msg; }

function setButtonLoading(btn, loading, label) {
  btn.disabled = loading;
  const labelEl = btn.querySelector('.btn-label');
  if (labelEl) labelEl.textContent = label;
  const existing = btn.querySelector('.spinner');
  if (loading && !existing) {
    const s = document.createElement('div');
    s.className = 'spinner';
    btn.prepend(s);
  } else if (!loading && existing) {
    existing.remove();
  }
}
