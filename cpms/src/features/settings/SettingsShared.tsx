import React, { useState, useCallback } from 'react';
import { cn } from '../../utils/cn';

// ─── Shared UI Primitives ────────────────────────────────────────────────────

export function Field({ label, hint, error, required, children }: {
  label: string; hint?: string; error?: string; required?: boolean; children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-semibold text-text-secondary">
        {label} {required && <span className="text-danger">*</span>}
      </label>
      {children}
      {hint && !error && <p className="text-xs text-text-tertiary">{hint}</p>}
      {error && <p className="text-xs text-danger font-medium">{error}</p>}
    </div>
  );
}

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'w-full h-11 px-3 bg-white border border-border-subtle rounded-lg text-sm text-text-primary placeholder:text-text-tertiary',
        'focus:outline-none focus:ring-2 focus:ring-brand-primary-light focus:border-brand-primary transition-all',
        className
      )}
      {...props}
    />
  );
}

export function Textarea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'w-full px-3 py-2.5 bg-white border border-border-subtle rounded-lg text-sm text-text-primary placeholder:text-text-tertiary resize-none',
        'focus:outline-none focus:ring-2 focus:ring-brand-primary-light focus:border-brand-primary transition-all',
        className
      )}
      {...props}
    />
  );
}

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'w-full h-11 px-3 bg-white border border-border-subtle rounded-lg text-sm text-text-primary appearance-none',
        'focus:outline-none focus:ring-2 focus:ring-brand-primary-light focus:border-brand-primary transition-all',
        className
      )}
      {...props}
    >
      {children}
    </select>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <label className="flex items-center gap-3 cursor-pointer group select-none">
      <div
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          'relative w-11 h-6 rounded-full transition-colors cursor-pointer focus:outline-none',
          checked ? 'bg-brand-primary' : 'bg-border-strong'
        )}
      >
        <div className={cn(
          'absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200',
          checked ? 'translate-x-5' : 'translate-x-0'
        )} />
      </div>
      {label && <span className="text-sm font-medium text-text-primary group-hover:text-brand-primary transition-colors">{label}</span>}
    </label>
  );
}

export function SegmentedControl({ options, value, onChange }: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="inline-flex p-1 gap-1 bg-surface-ground border border-border-subtle rounded-xl" role="group">
      {options.map(opt => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={value === opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'px-4 py-1.5 rounded-lg text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-brand-primary-light',
            value === opt.value
              ? 'bg-brand-primary-xlt text-brand-primary border border-brand-primary/30 shadow-sm'
              : 'text-text-secondary hover:text-text-primary'
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function SectionCard({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface-card rounded-xl border border-border-subtle shadow-sm overflow-hidden">
      {title && (
        <div className="px-6 py-3 border-b border-border-subtle bg-surface-ground">
          <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-wider">{title}</h3>
        </div>
      )}
      <div className="p-6 space-y-5">{children}</div>
    </div>
  );
}

export function AvatarUpload({ initials, label }: { initials: string; label: string }) {
  return (
    <div className="flex items-center gap-4">
      <div className="w-16 h-16 rounded-full bg-brand-primary text-white flex items-center justify-center font-bold text-xl shadow-sm shrink-0">
        {initials}
      </div>
      <button
        type="button"
        className="text-sm font-semibold text-brand-primary hover:text-brand-primary-mid px-3 py-1.5 bg-brand-primary-xlt rounded-lg transition-colors"
      >
        {label}
      </button>
    </div>
  );
}

// Branded Discard Dialog (reused from PatientFormDrawer pattern)
export function DiscardDialog({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in px-4">
      <div className="bg-surface-card w-full max-w-sm rounded-xl shadow-xl p-6 border border-border-subtle">
        <h3 className="text-lg font-bold text-text-primary mb-2">Discard unsaved changes?</h3>
        <p className="text-sm text-text-secondary mb-6">Your edits to this section haven't been saved yet.</p>
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 text-sm font-semibold text-text-secondary hover:bg-surface-ground rounded-lg transition-colors">
            Keep editing
          </button>
          <button onClick={onConfirm} className="px-4 py-2 text-sm font-semibold text-white bg-danger hover:bg-danger/90 rounded-lg shadow-sm transition-colors">
            Discard changes
          </button>
        </div>
      </div>
    </div>
  );
}

// Success Toast
export function useToast() {
  const [visible, setVisible] = useState(false);
  const [message, setMessage] = useState('');

  const showToast = useCallback((msg: string) => {
    setMessage(msg);
    setVisible(true);
    setTimeout(() => setVisible(false), 2800);
  }, []);

  const Toast = visible ? (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[70] animate-slide-up">
      <div className="flex items-center gap-3 bg-text-primary text-white px-5 py-3 rounded-full shadow-lg text-sm font-semibold">
        <svg className="w-4 h-4 text-accent-green shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        {message}
      </div>
    </div>
  ) : null;

  return { showToast, Toast };
}
