import { useState } from 'react';
import { ZoomIn } from 'lucide-react';

export default function ImageModal({ children }) {
  const [modal, setModal] = useState(null); // { src, label }

  const open = (src, label) => setModal({ src, label });
  const close = () => setModal(null);

  return (
    <>
      {children(open)}

      {modal && (
        <div
          className="fixed inset-0 z-[9998] flex items-center justify-center p-6"
          onClick={close}
        >
          <div className="absolute inset-0 bg-black/85 backdrop-blur-sm" />
          <div
            className="relative max-w-3xl max-h-[85vh] rounded-2xl overflow-hidden border border-violet-500/30 shadow-2xl shadow-violet-500/20"
            onClick={e => e.stopPropagation()}
          >
            <div className="scan-line" />
            <img
              src={modal.src}
              alt={modal.label}
              className="block max-w-full max-h-[80vh] object-contain"
            />
            <div className="absolute bottom-3 left-3 bg-black/70 text-violet-300 text-xs font-mono px-3 py-1.5 rounded-lg border-l-2 border-violet-500">
              {modal.label}
            </div>
            <button
              onClick={close}
              className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/50 border border-white/10 flex items-center justify-center text-white/60 hover:text-white hover:bg-black/70 transition-all"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export function ImageGrid({ images, label }) {
  const placeholder = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(
    '<svg width="160" height="100" xmlns="http://www.w3.org/2000/svg"><rect width="160" height="100" fill="#111"/><text x="80" y="55" text-anchor="middle" fill="#444" font-family="sans-serif" font-size="11">Unavailable</text></svg>'
  );

  return (
    <ImageModal>
      {(open) => (
        <div className="flex gap-3 overflow-x-auto pb-2">
          {images?.length ? images.map((url, i) => (
            <div
              key={i}
              className="group relative shrink-0 w-40 h-24 rounded-xl overflow-hidden border border-white/8 cursor-zoom-in hover:border-violet-500/50 transition-all hover:shadow-lg hover:shadow-violet-500/10 hover:-translate-y-1"
              onClick={() => open(url, `${label} #${i}`)}
            >
              <img
                src={url}
                alt={`${label} ${i}`}
                className="w-full h-full object-cover"
                onError={e => { e.target.src = placeholder; }}
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center">
                <ZoomIn size={20} className="text-white opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
                <span className="text-[10px] font-mono text-white/60">{label}_{i}</span>
              </div>
            </div>
          )) : (
            <p className="text-xs text-white/20 py-4">No {label.toLowerCase()}s detected.</p>
          )}
        </div>
      )}
    </ImageModal>
  );
}
