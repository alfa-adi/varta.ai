import React from 'react';
import { useToastStore } from '../../stores/toastStore';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { cn } from '../../utils/cn';

export function ToastProvider() {
  const { toasts, removeToast } = useToastStore();

  return (
    /* On mobile: bottom-[4.5rem] keeps toasts above the 64px bottom nav.
       On desktop: bottom-4 is standard. max-w prevents overflow on 360px. */
    <div
      aria-label="Notifications"
      className="fixed bottom-[4.5rem] md:bottom-4 right-4 left-4 md:left-auto z-[60] flex flex-col gap-2 pointer-events-none"
    >
      {toasts.map((toast) => (
        <div 
          key={toast.id} 
          className={cn(
            "pointer-events-auto flex items-center gap-3 p-4 rounded-xl shadow-lg border border-border-subtle bg-surface-card animate-slide-up",
            "w-full md:min-w-[300px] md:max-w-sm",
            toast.type === 'success' && "border-l-4 border-l-success",
            toast.type === 'warning' && "border-l-4 border-l-warning",
            toast.type === 'error'   && "border-l-4 border-l-danger",
            toast.type === 'info'    && "border-l-4 border-l-brand-primary"
          )}
          role={toast.type === 'error' || toast.type === 'warning' ? 'alert' : 'status'}
        >
          {toast.type === 'success' && <CheckCircle2 className="text-success shrink-0" size={20} aria-hidden="true" />}
          {toast.type === 'error'   && <AlertCircle  className="text-danger shrink-0"  size={20} aria-hidden="true" />}
          {toast.type === 'warning' && <AlertCircle  className="text-warning shrink-0" size={20} aria-hidden="true" />}
          {toast.type === 'info'    && <Info          className="text-brand-primary shrink-0" size={20} aria-hidden="true" />}
          
          <p className="flex-1 text-sm font-semibold text-text-primary">{toast.message}</p>
          
          {toast.action && (
            <button 
              onClick={() => {
                toast.action!.onClick();
                removeToast(toast.id);
              }}
              className="text-sm font-bold text-brand-primary hover:underline px-2 focus-ring"
            >
              {toast.action.label}
            </button>
          )}

          <button
            onClick={() => removeToast(toast.id)}
            aria-label="Dismiss notification"
            className="text-text-tertiary hover:text-text-secondary focus-ring rounded"
          >
            <X size={16} />
          </button>
        </div>
      ))}
    </div>
  );
}
