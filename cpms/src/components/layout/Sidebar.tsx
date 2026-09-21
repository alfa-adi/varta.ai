import React from 'react';
import { cn } from '../../utils/cn';
import { Home, Users, Calendar, Settings, Activity, Menu } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useSettingsStore } from '../../stores/settingsStore';

interface SidebarProps {
  isExpanded: boolean;
  onToggle: () => void;
}

export function Sidebar({ isExpanded, onToggle }: SidebarProps) {
  const { doctor, clinic } = useSettingsStore();

  const navItems = [
    { icon: <Home size={24} />, label: 'Dashboard', path: '/' },
    { icon: <Users size={24} />, label: 'Patients', path: '/patients' },
    { icon: <Calendar size={24} />, label: 'Appointments', path: '/appointments' },
    { icon: <Activity size={24} />, label: 'Reports', path: '/reports' },
    { icon: <Settings size={24} />, label: 'Settings', path: '/settings' },
  ];

  return (
    <aside
      className={cn(
        'hidden md:flex fixed top-0 left-0 h-screen bg-surface-card border-r border-border-subtle flex-col transition-all duration-300 z-40',
        isExpanded ? 'w-[240px]' : 'w-[72px]'
      )}
    >
      <div className="h-16 flex items-center justify-center border-b border-border-subtle">
        <button
          onClick={onToggle}
          className="p-2 rounded-md hover:bg-brand-primary-light text-brand-primary transition-colors focus:outline-none focus:ring-2 focus:ring-brand-primary"
        >
          <Menu size={24} />
        </button>
      </div>

      <nav className="flex-1 py-4 flex flex-col gap-2 px-3 overflow-y-auto">
        {navItems.map((item, idx) => (
          <NavLink
            key={idx}
            to={item.path}
            className={({ isActive }) => cn(
              'flex items-center rounded-lg py-3 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-primary relative',
              isExpanded ? 'px-4 justify-start' : 'justify-center',
              isActive 
                ? 'text-brand-primary bg-brand-primary-light font-bold' 
                : 'text-text-secondary hover:bg-surface-ground hover:text-text-primary'
            )}
            title={!isExpanded ? item.label : undefined}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1.5 h-8 bg-brand-primary rounded-r-full" />
                )}
                <span className="shrink-0">{item.icon}</span>
                {isExpanded && (
                  <span className="ml-4 text-md whitespace-nowrap">
                    {item.label}
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      
      {isExpanded && (
        <div className="p-4 border-t border-border-subtle mt-auto">
          <div className="bg-brand-primary-light p-3 rounded-md">
            <p className="text-sm text-brand-primary font-bold truncate">{doctor.name}</p>
            <p className="text-xs text-text-secondary mt-0.5 truncate">{clinic.name}</p>
          </div>
        </div>
      )}
    </aside>
  );
}
