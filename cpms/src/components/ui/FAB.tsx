import React from 'react';
import { cn } from '../../utils/cn';
import { Plus } from 'lucide-react';

export interface FABProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  expanded?: boolean;
  icon?: React.ReactNode;
  label?: string;
}

export function FAB({ expanded = false, icon, label, className, ...props }: FABProps) {
  return (
    <button
      className={cn(
        'fixed bottom-6 right-6 flex items-center justify-center bg-brand-primary text-text-on-brand shadow-fab transition-all duration-300 hover:bg-brand-primary-mid focus:outline-none focus:ring-4 focus:ring-brand-primary-light z-50',
        expanded ? 'rounded-xl px-4 py-3 h-14' : 'rounded-full h-14 w-14',
        className
      )}
      {...props}
    >
      <span className={cn("transition-transform duration-300", expanded ? "rotate-0" : "rotate-90")}>
        {icon || <Plus size={24} />}
      </span>
      {expanded && label && (
        <span className="ml-2 text-md font-semibold whitespace-nowrap opacity-100 animate-fade-in">
          {label}
        </span>
      )}
    </button>
  );
}
