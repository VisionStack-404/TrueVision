import { useState, useRef, useCallback } from 'react';
import { Upload, X, FileVideo, ImageIcon, CheckCircle2 } from 'lucide-react';

const ACCEPTED = {
  'image/jpeg': '.jpg/.jpeg',
  'image/png': '.png',
  'image/webp': '.webp',
  'video/mp4': '.mp4',
  'video/avi': '.avi',
  'video/quicktime': '.mov',
};

export default function UploadZone({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState(null); // { name, size, type, url }
  const inputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!ACCEPTED[file.type] && !file.name.match(/\.(jpg|jpeg|png|webp|mp4|avi|mov)$/i)) {
      alert('Unsupported file type.');
      return;
    }
    const url = file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    setPreview({ name: file.name, size: file.size, type: file.type, url });
    onFile(file);
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile, disabled]);

  const clearPreview = (e) => {
    e.stopPropagation();
    setPreview(null);
    if (inputRef.current) inputRef.current.value = '';
    onFile(null);
  };

  const isVideo = preview?.type?.startsWith('video/');
  const sizeLabel = preview ? (preview.size > 1048576
    ? `${(preview.size / 1048576).toFixed(1)} MB`
    : `${(preview.size / 1024).toFixed(0)} KB`) : '';

  return (
    <div
      onDragOver={e => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && !preview && inputRef.current?.click()}
      className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer select-none
        ${dragging
          ? 'border-violet-500 bg-violet-500/8 scale-[1.01]'
          : preview
            ? 'border-violet-500/30 bg-violet-500/4 cursor-default'
            : 'border-white/10 hover:border-violet-500/50 hover:bg-violet-500/4'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.mp4,.avi,.mov"
        className="hidden"
        onChange={e => handleFile(e.target.files[0])}
        disabled={disabled}
      />

      {!preview ? (
        <div className="flex flex-col items-center justify-center gap-4 p-12 text-center">
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
              <Upload size={28} className="text-violet-400" />
            </div>
            {/* Decorative rings */}
            <div className="absolute -inset-3 rounded-3xl border border-violet-500/10 animate-ping" style={{animationDuration:'3s'}} />
          </div>
          <div>
            <p className="font-semibold text-white/80 text-sm mb-1">
              Drag & drop your media file
            </p>
            <p className="text-xs text-white/30">
              Images: .jpg .png .webp — Videos: .mp4 .avi .mov
            </p>
          </div>
          <button
            type="button"
            className="px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-violet-500/20"
            onClick={e => { e.stopPropagation(); inputRef.current?.click(); }}
          >
            Browse Files
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-5 p-5">
          {/* Thumbnail or icon */}
          <div className="w-20 h-20 rounded-xl overflow-hidden bg-white/5 border border-white/8 flex items-center justify-center shrink-0">
            {preview.url ? (
              <img src={preview.url} alt="preview" className="w-full h-full object-cover" />
            ) : isVideo ? (
              <FileVideo size={32} className="text-violet-400" />
            ) : (
              <ImageIcon size={32} className="text-violet-400" />
            )}
          </div>
          {/* Info */}
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-white/90 text-sm truncate">{preview.name}</p>
            <p className="text-xs text-white/30 mt-1">{sizeLabel} · {isVideo ? 'Video' : 'Image'}</p>
            <div className="flex items-center gap-1.5 mt-2">
              <CheckCircle2 size={13} className="text-emerald-400" />
              <span className="text-xs text-emerald-400 font-medium">Ready for analysis</span>
            </div>
          </div>
          <button
            onClick={clearPreview}
            className="p-2 rounded-lg hover:bg-white/8 text-white/30 hover:text-white/70 transition-all"
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
