import React from 'react';
import { X, Check } from 'lucide-react';
import { cn } from '../../../utils/cn';

export interface FilterState {
  urgency: string[];
  conditions: string[];
  timeRange: string; // 'all', '30days', '90days', '1year'
  language: string[];
  archiveState: 'active' | 'archived' | 'all';
}

interface FilterSheetProps {
  isOpen: boolean;
  onClose: () => void;
  filters: FilterState;
  onChange: (filters: FilterState) => void;
  onReset: () => void;
}

const URGENCY_OPTIONS = [
  { id: 'urgent', label: 'Urgent', color: 'bg-danger text-white' },
  { id: 'attention', label: 'Attention', color: 'bg-warning text-white' },
  { id: 'routine', label: 'Routine', color: 'bg-brand-primary text-white' }
];

const TIME_RANGES = [
  { id: 'all', label: 'Any time' },
  { id: '30days', label: 'Last 30 days' },
  { id: '90days', label: 'Last 90 days' },
  { id: '1year', label: 'Last 1 year' }
];

const ARCHIVE_STATES = [
  { id: 'active', label: 'Active Patients' },
  { id: 'archived', label: 'Archived Only' },
  { id: 'all', label: 'All Patients' }
];

const COMMON_CONDITIONS = ['Hypertension', 'Type 2 Diabetes', 'Asthma', 'Arthritis', 'Thyroid'];
const LANGUAGES = ['English', 'Hindi', 'Marathi', 'Gujarati', 'Tamil'];

export function FilterSheet({ isOpen, onClose, filters, onChange, onReset }: FilterSheetProps) {
  if (!isOpen) return null;

  const toggleArrayItem = (array: string[], item: string) => {
    if (array.includes(item)) {
      return array.filter(i => i !== item);
    }
    return [...array, item];
  };

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
      />
      
      {/* Sheet */}
      <div className="fixed inset-x-0 bottom-0 max-h-[90vh] md:inset-y-0 md:inset-x-auto md:right-0 w-full md:max-w-sm bg-surface-card border-t md:border-l md:border-t-0 border-border-subtle shadow-2xl z-50 flex flex-col rounded-t-2xl md:rounded-none animate-slide-up md:animate-slide-in-right pb-safe">
        
        <div className="flex items-center justify-between p-4 border-b border-border-subtle bg-surface-ground">
          <h2 className="text-lg font-bold text-text-primary">Filters</h2>
          <div className="flex items-center gap-2">
            <button onClick={onReset} className="text-sm font-semibold text-text-tertiary hover:text-text-secondary px-2 py-1">
              Reset
            </button>
            <button onClick={onClose} className="p-2 text-text-secondary hover:bg-surface-card rounded-md transition-colors">
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {/* Archive State (CRITICAL) */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Patient Status</h3>
            <div className="flex flex-col gap-2">
              {ARCHIVE_STATES.map(state => (
                <label 
                  key={state.id} 
                  className="flex items-center gap-3 cursor-pointer group"
                  onClick={() => onChange({ ...filters, archiveState: state.id as 'active' | 'archived' | 'all' })}
                >
                  <div className={cn(
                    "w-4 h-4 rounded-full border flex items-center justify-center transition-colors",
                    filters.archiveState === state.id ? "border-brand-primary bg-brand-primary" : "border-border-subtle group-hover:border-brand-primary/50"
                  )}>
                    {filters.archiveState === state.id && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                  </div>
                  <span className="text-sm text-text-primary font-medium">{state.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Urgency */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Urgency</h3>
            <div className="flex flex-wrap gap-2">
              {URGENCY_OPTIONS.map(opt => {
                const isActive = filters.urgency.includes(opt.id);
                return (
                  <button
                    key={opt.id}
                    onClick={() => onChange({ ...filters, urgency: toggleArrayItem(filters.urgency, opt.id) })}
                    className={cn(
                      "px-3 py-1.5 rounded-full text-sm font-semibold border transition-all flex items-center gap-1.5",
                      isActive ? "border-brand-primary bg-brand-primary-light text-brand-primary" : "border-border-subtle text-text-secondary hover:border-brand-primary/30"
                    )}
                  >
                    {isActive && <Check size={14} />}
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Time Range */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Last Visit</h3>
            <select 
              value={filters.timeRange}
              onChange={(e) => onChange({ ...filters, timeRange: e.target.value })}
              className="w-full h-10 px-3 bg-surface-ground border border-border-subtle rounded-lg text-sm focus:outline-none focus:border-brand-primary transition-colors"
            >
              {TIME_RANGES.map(range => (
                <option key={range.id} value={range.id}>{range.label}</option>
              ))}
            </select>
          </div>

          {/* Conditions */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Common Conditions</h3>
            <div className="flex flex-wrap gap-2">
              {COMMON_CONDITIONS.map(cond => {
                const isActive = filters.conditions.includes(cond);
                return (
                  <button
                    key={cond}
                    onClick={() => onChange({ ...filters, conditions: toggleArrayItem(filters.conditions, cond) })}
                    className={cn(
                      "px-3 py-1.5 rounded-lg text-sm font-semibold border transition-all flex items-center gap-1.5",
                      isActive ? "border-brand-primary bg-brand-primary text-text-on-brand shadow-sm" : "border-border-subtle bg-surface-ground text-text-secondary hover:border-brand-primary/30"
                    )}
                  >
                    {isActive && <Check size={14} />}
                    {cond}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Languages */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Preferred Language</h3>
            <div className="flex flex-wrap gap-2">
              {LANGUAGES.map(lang => {
                const isActive = filters.language.includes(lang);
                return (
                  <button
                    key={lang}
                    onClick={() => onChange({ ...filters, language: toggleArrayItem(filters.language, lang) })}
                    className={cn(
                      "px-3 py-1.5 rounded-lg text-sm font-semibold border transition-all flex items-center gap-1.5",
                      isActive ? "border-brand-primary bg-brand-primary-light text-brand-primary" : "border-border-subtle text-text-secondary hover:border-brand-primary/30"
                    )}
                  >
                    {isActive && <Check size={14} />}
                    {lang}
                  </button>
                );
              })}
            </div>
          </div>

        </div>
        
        <div className="p-4 border-t border-border-subtle bg-surface-ground">
          <button 
            onClick={onClose}
            className="w-full h-12 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-xl font-bold text-sm shadow-sm transition-colors"
          >
            Show Results
          </button>
        </div>
      </div>
    </>
  );
}
