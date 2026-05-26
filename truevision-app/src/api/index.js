import axios from 'axios';
import { API_BASE } from './config';

const api = axios.create({ baseURL: API_BASE, timeout: 600000 });

// ── Health check ────────────────────────────────────────────────
export const checkHealth = () => api.get('/');

// ── Capabilities ────────────────────────────────────────────────
export const fetchCapabilities = () => api.get('/detection/capabilities/');

// ── Process a file ─────────────────────────────────────────────
export const processFile = (file, onUploadProgress) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/process/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  });
};

// ── Submit feedback ─────────────────────────────────────────────
export const submitFeedback = (label) =>
  api.post('/feedback/', { label });
