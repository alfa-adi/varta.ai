import React from 'react';
import { cn } from '../../utils/cn';

export type TagVariant = 'symptom' | 'medicine' | 'test' | 'allergy' | 'chronic' | 'neutral';

export interface TagPillProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant: TagVariant;
  children: React.ReactNode;
}

const variantStyles: Record<TagVariant, string> = {
  symptom: 'bg-tag-symptom-bg text-tag-symptom-text',
  medicine: 'bg-tag-medicine-bg text-tag-medicine-text',
  test: 'bg-tag-test-bg text-tag-test-text',
  allergy: 'bg-tag-allergy-bg text-tag-allergy-text',
  chronic: 'bg-tag-chronic-bg text-tag-chronic-text',
  neutral: 'bg-border-subtle text-text-secondary',
};

export function TagPill({ variant, children, className, ...props }: TagPillProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-sm text-xs font-medium',
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
