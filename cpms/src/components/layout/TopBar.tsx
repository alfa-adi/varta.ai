import React from 'react';
import { Bell, Search } from 'lucide-react';
import { useSearchStore } from '../../stores/searchStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useToastStore } from '../../stores/toastStore';
import { useNavigate } from 'react-router-dom';

interface TopBarProps {
  title: string;
}

export function TopBar({ title }: TopBarProps) {
  const openSearch = useSearchStore((state) => state.openSearch);
  const { doctor } = useSettingsStore();
  const addToast = useToastStore(state => state.addToast);
  const navigate = useNavigate();
  
  const initials = doctor.name
    .replace(/^Dr\.?\s+/i, '')
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map(n => n[0].toUpperCase())
    .join('');

  return (
    <header className="h-16 bg-surface-card border-b border-border-subtle flex items-center justify-between px-4 md:px-6 sticky top-0 z-30">
      <h1 className="text-xl font-bold text-text-primary tracking-tight">
        {title}
      </h1>
      
      <div className="flex items-center gap-2 md:gap-4">
        {/* Mobile search icon */}
        <button 
          onClick={openSearch}
          aria-label="Open search"
          className="md:hidden p-2 text-text-secondary hover:bg-brand-primary-light hover:text-brand-primary rounded-full transition-colors focus-ring"
        >
          <Search size={20} />
        </button>
        
        {/* Desktop search pill */}
        <button 
          onClick={openSearch}
          aria-label="Open search (Ctrl+K)"
          className="relative hidden md:flex items-center w-64 bg-brand-primary-xlt text-sm text-text-tertiary hover:text-text-secondary rounded-full py-2 pl-3 pr-4 border border-transparent hover:border-brand-primary/30 transition-all group focus-ring"
        >
          <Search size={18} className="text-text-tertiary group-hover:text-brand-primary transition-colors mr-2" />
          <span className="flex-1 text-left">Search...</span>
          <div className="hidden lg:flex items-center gap-1 opacity-70">
            <kbd className="bg-surface-card border border-border-subtle rounded px-1.5 text-[10px] font-semibold text-text-secondary shadow-sm">Ctrl</kbd>
            <span className="text-[10px] text-text-tertiary font-bold">+</span>
            <kbd className="bg-surface-card border border-border-subtle rounded px-1.5 text-[10px] font-semibold text-text-secondary shadow-sm">K</kbd>
          </div>
        </button>
        
        {/* Notifications */}
        <button
          aria-label="Notifications (coming soon)"
          onClick={() => addToast({ type: 'info', message: 'Notifications coming soon.' })}
          className="relative p-2 text-text-secondary hover:bg-brand-primary-light hover:text-brand-primary rounded-full transition-colors focus-ring"
        >
          <Bell size={20} />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-danger border-2 border-white" aria-hidden="true" />
        </button>
        
        {/* Doctor avatar → Settings */}
        <button
          aria-label={`Settings – ${doctor.name}`}
          onClick={() => navigate('/settings/doctor')}
          className="h-8 w-8 rounded-full bg-brand-primary flex items-center justify-center text-text-on-brand font-bold text-xs shadow-sm ml-1 hover:ring-2 hover:ring-brand-primary/30 transition-all focus-ring"
        >
          {initials}
        </button>
      </div>
    </header>
  );
}
