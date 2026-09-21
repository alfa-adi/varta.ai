import React from 'react';
import { FlaskConical } from 'lucide-react';
import { cn } from '../../utils/cn';

interface DemoBadgeProps {
  label?: string;
  className?: string;
}

/**
 * Small chip to clearly label placeholder/demo-only features.
 * Use on any UI element that looks functional but isn't in demo mode.
 */
export function DemoBadge({ label = 'Demo only', className }: DemoBadgeProps) {
  return (
    <span
      title="This feature is placeholder/demo only and not functional."
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider',
        'bg-amber-100 text-amber-700 border border-amber-200',
        className
      )}
    >
      <FlaskConical size={9} aria-hidden="true" />
      {label}
    </span>
  );
}
