import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Home, Users, Calendar, Settings, Activity, Plus, Mic, UserPlus, FileText, UserCircle, LogOut, MoreHorizontal } from 'lucide-react';
import { cn } from '../../utils/cn';
import { useRecordingStore } from '../../stores/recordingStore';
import { usePatientFormStore } from '../../stores/patientFormStore';
import { BottomSheet } from '../ui/BottomSheet';
import { useSettingsStore } from '../../stores/settingsStore';
import { useAuthStore } from '../../stores/authStore';

export function BottomNav() {
  const navigate = useNavigate();
  const [isFabOpen, setIsFabOpen] = useState(false);
  const [isMoreOpen, setIsMoreOpen] = useState(false);
  
  const openLauncher = useRecordingStore(state => state.openLauncher);
  const openPatientForm = usePatientFormStore(state => state.openForm);
  const { doctor } = useSettingsStore();
  const { logout } = useAuthStore();

  const navItems = [
    { icon: <Home size={24} />, label: 'Home', path: '/' },
    { icon: <Users size={24} />, label: 'Patients', path: '/patients' },
    { isFab: true },
    { icon: <Calendar size={24} />, label: 'Calendar', path: '/appointments' },
    { icon: <MoreHorizontal size={24} />, label: 'More', isMore: true },
  ];

  const handleFabClick = () => setIsFabOpen(true);
  const handleMoreClick = () => setIsMoreOpen(true);

  const handleAction = (action: () => void) => {
    action();
    setIsFabOpen(false);
    setIsMoreOpen(false);
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    setIsMoreOpen(false);
  };

  return (
    <>
      <nav
        role="navigation"
        aria-label="Main navigation"
        className="fixed bottom-0 left-0 right-0 bg-surface-card border-t border-border-subtle pb-safe z-40 flex items-center justify-around h-16 md:hidden px-2 shadow-[0_-2px_10px_rgba(0,0,0,0.05)]"
      >
        {navItems.map((item, idx) => {
          if (item.isFab) {
            return (
              <div key="fab" className="relative -top-5 flex flex-col items-center justify-center">
                <button
                  onClick={handleFabClick}
                  className="flex items-center justify-center bg-brand-primary text-text-on-brand shadow-fab hover:bg-brand-primary-mid focus-ring rounded-full h-14 w-14 transition-transform active:scale-95"
                  aria-label="Quick actions"
                  aria-haspopup="dialog"
                >
                  <Plus size={28} />
                </button>
              </div>
            );
          }

          if (item.isMore) {
            return (
              <button
                key="more"
                onClick={handleMoreClick}
                aria-label="More options"
                aria-haspopup="dialog"
                aria-expanded={isMoreOpen}
                className={cn(
                  'flex flex-col items-center justify-center w-16 h-full transition-colors focus-ring',
                  isMoreOpen ? 'text-brand-primary' : 'text-text-secondary hover:text-text-primary'
                )}
              >
                {item.icon}
                <span className="text-[10px] mt-1 font-medium">{item.label}</span>
              </button>
            );
          }

          return (
            <NavLink
              key={item.path}
              to={item.path!}
              aria-current={undefined}  // NavLink handles this via className
              className={({ isActive }) => cn(
                'flex flex-col items-center justify-center w-16 h-full transition-colors focus-ring',
                isActive 
                  ? 'text-brand-primary' 
                  : 'text-text-secondary hover:text-text-primary'
              )}
            >
              {item.icon}
              <span className="text-[10px] mt-1 font-medium">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* FAB Action Sheet */}
      <BottomSheet isOpen={isFabOpen} onClose={() => setIsFabOpen(false)} title="Quick Actions">
        <div className="flex flex-col gap-2 pb-4">
          <button 
            onClick={() => handleAction(openLauncher)}
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-surface-ground transition-colors focus:outline-none focus:bg-surface-ground text-left"
          >
            <div className="bg-brand-primary-light p-3 rounded-full text-brand-primary">
              <Mic size={24} />
            </div>
            <div>
              <div className="font-semibold text-text-primary">New Session</div>
              <div className="text-sm text-text-secondary">Start a voice consultation</div>
            </div>
          </button>
          
          <button 
            onClick={() => handleAction(openPatientForm)}
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-surface-ground transition-colors focus:outline-none focus:bg-surface-ground text-left"
          >
            <div className="bg-accent-green-light p-3 rounded-full text-accent-green">
              <UserPlus size={24} />
            </div>
            <div>
              <div className="font-semibold text-text-primary">Add Patient</div>
              <div className="text-sm text-text-secondary">Register a new patient</div>
            </div>
          </button>
          
          <button 
            onClick={() => handleAction(() => {})} // Placeholder for now
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-surface-ground transition-colors focus:outline-none focus:bg-surface-ground text-left opacity-70"
          >
            <div className="bg-surface-ground border border-border-default p-3 rounded-full text-text-secondary">
              <FileText size={24} />
            </div>
            <div>
              <div className="font-semibold text-text-primary flex items-center gap-2">
                Quick Note <span className="text-[10px] bg-border-subtle px-1.5 py-0.5 rounded text-text-secondary">Soon</span>
              </div>
              <div className="text-sm text-text-secondary">Write a manual text note</div>
            </div>
          </button>
        </div>
      </BottomSheet>

      {/* More Sheet */}
      <BottomSheet isOpen={isMoreOpen} onClose={() => setIsMoreOpen(false)} title="More">
        <div className="flex flex-col gap-2 pb-4">
          <div className="flex items-center gap-4 p-4 mb-2 bg-brand-primary-xlt rounded-xl">
            <div className="bg-brand-primary-light p-3 rounded-full text-brand-primary">
              <UserCircle size={24} />
            </div>
            <div>
              <div className="font-bold text-brand-primary">{doctor.name}</div>
              <div className="text-sm text-brand-primary/70">View Profile</div>
            </div>
          </div>

          <button 
            onClick={() => handleNavigate('/reports')}
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-surface-ground transition-colors focus:outline-none focus:bg-surface-ground text-left"
          >
            <Activity size={24} className="text-text-secondary" />
            <span className="font-medium text-text-primary">Reports</span>
          </button>

          <button 
            onClick={() => handleNavigate('/settings')}
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-surface-ground transition-colors focus:outline-none focus:bg-surface-ground text-left"
          >
            <Settings size={24} className="text-text-secondary" />
            <span className="font-medium text-text-primary">Settings</span>
          </button>
          
          <div className="h-px bg-border-subtle my-2" />

          <button 
            onClick={() => handleAction(() => { logout(); navigate('/login'); })} 
            className="flex items-center gap-4 p-4 rounded-xl hover:bg-danger-light transition-colors focus:outline-none focus:bg-danger-light text-left text-danger"
            aria-label="Sign out"
          >
            <LogOut size={24} />
            <span className="font-medium">Sign Out</span>
          </button>
        </div>
      </BottomSheet>
    </>
  );
}
