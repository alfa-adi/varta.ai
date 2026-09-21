import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AppShell } from '../../components/layout/AppShell';
import { Building2, User, Mic, Globe, Bell, ShieldCheck, ArrowLeft } from 'lucide-react';
import { cn } from '../../utils/cn';
import { ClinicProfileSection } from './sections/ClinicProfileSection';
import { DoctorProfileSection } from './sections/DoctorProfileSection';
import { RecordingSection } from './sections/RecordingSection';
import { DisplaySection } from './sections/DisplaySection';
import { NotificationsSection } from './sections/NotificationsSection';
import { AccountSection } from './sections/AccountSection';

type SectionId = 'clinic' | 'doctor' | 'recording' | 'display' | 'notifications' | 'account';

const navItems: { id: SectionId; label: string; icon: React.ReactNode; description: string }[] = [
  { id: 'clinic', label: 'Clinic Profile', icon: <Building2 size={18} />, description: 'Name, address, prescription header' },
  { id: 'doctor', label: 'Doctor Profile', icon: <User size={18} />, description: 'Your identity and signature' },
  { id: 'recording', label: 'Recording & AI', icon: <Mic size={18} />, description: 'Mode, language, AI summary' },
  { id: 'display', label: 'Language & Display', icon: <Globe size={18} />, description: 'Date, time, UI language' },
  { id: 'notifications', label: 'Notifications', icon: <Bell size={18} />, description: 'Reminders and quiet hours' },
  { id: 'account', label: 'Account & Data', icon: <ShieldCheck size={18} />, description: 'Sign-out, export, data' },
];

const sectionTitles: Record<SectionId, string> = {
  clinic: 'Clinic Profile',
  doctor: 'Doctor Profile',
  recording: 'Recording & AI',
  display: 'Language & Display',
  notifications: 'Notifications',
  account: 'Account & Data',
};

const sectionSubtitles: Record<SectionId, string> = {
  clinic: 'Set up your clinic identity and prescription header',
  doctor: 'Manage your personal details and prescription signature',
  recording: 'Configure default recording mode, languages, and AI behaviour',
  display: 'Adjust date format, time format, and interface language',
  notifications: 'Control reminders and quiet hours',
  account: 'Account details, data export, and sign-out',
};

export function SettingsPage() {
  const { sectionId } = useParams<{ sectionId: string }>();
  const navigate = useNavigate();
  
  // Default to clinic on desktop if no section selected
  const activeSection = (sectionId as SectionId) || 'clinic';

  const renderSection = () => {
    switch (activeSection) {
      case 'clinic': return <ClinicProfileSection />;
      case 'doctor': return <DoctorProfileSection />;
      case 'recording': return <RecordingSection />;
      case 'display': return <DisplaySection />;
      case 'notifications': return <NotificationsSection />;
      case 'account': return <AccountSection />;
    }
  };

  return (
    <AppShell title="Settings">
      <div className="animate-fade-in">
        {/* Page Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-text-primary">Settings</h1>
          <p className="text-sm text-text-secondary mt-0.5">Manage your clinic and preferences.</p>
        </div>

        {/* Two-column layout */}
        <div className="flex gap-6 items-start">
          
          {/* Left Nav — ~220px. Hidden on mobile if section selected */}
          <nav className={cn(
            "w-full md:w-[220px] shrink-0 bg-surface-card rounded-xl border border-border-subtle shadow-sm overflow-hidden md:sticky md:top-[88px]",
            sectionId ? "hidden md:block" : "block"
          )}>
            {navItems.map((item, idx) => (
              <button
                key={item.id}
                onClick={() => navigate(`/settings/${item.id}`)}
                className={cn(
                  'w-full flex items-center gap-3 px-4 py-3.5 text-left transition-all relative focus:outline-none focus:ring-2 focus:ring-inset focus:ring-brand-primary-light',
                  idx > 0 && 'border-t border-border-subtle',
                  activeSection === item.id
                    ? 'bg-brand-primary-xlt text-brand-primary'
                    : 'text-text-secondary hover:bg-surface-ground hover:text-text-primary'
                )}
              >
                {activeSection === item.id && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-7 bg-brand-primary rounded-r-full" />
                )}
                <span className={cn('shrink-0', activeSection === item.id ? 'text-brand-primary' : 'text-text-tertiary')}>
                  {item.icon}
                </span>
                <div className="min-w-0">
                  <p className={cn('text-sm font-semibold truncate', activeSection === item.id ? 'text-brand-primary' : '')}>{item.label}</p>
                  <p className="text-xs text-text-tertiary truncate mt-0.5 hidden lg:block">{item.description}</p>
                </div>
              </button>
            ))}
          </nav>

          {/* Right Content Panel. Hidden on mobile if no section selected */}
          <div className={cn(
            "flex-1 min-w-0 max-w-[760px]",
            sectionId ? "block" : "hidden md:block"
          )}>
            {/* Section header */}
            <div className="mb-5 flex items-center gap-3">
              {sectionId && (
                <button
                  onClick={() => navigate('/settings')}
                  className="md:hidden p-2 -ml-2 text-text-secondary hover:bg-surface-ground rounded-full transition-colors flex items-center justify-center min-h-[44px] min-w-[44px]"
                  aria-label="Back to settings list"
                >
                  <ArrowLeft size={20} />
                </button>
              )}
              <div>
                <h2 className="text-lg font-bold text-text-primary">{sectionTitles[activeSection]}</h2>
                <p className="text-sm text-text-secondary mt-0.5">{sectionSubtitles[activeSection]}</p>
              </div>
            </div>
            {renderSection()}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
