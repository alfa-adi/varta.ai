import React, { useState } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import { useAuthStore } from '../../../stores/authStore';
import { SectionCard } from '../SettingsShared';
import { LogOut, Download, Phone, ShieldAlert } from 'lucide-react';

function DestructiveRow({ icon, title, description, actionLabel, destructive = false, onClick, disabled = false }: {
  icon: React.ReactNode; title: string; description: string; actionLabel: string; destructive?: boolean; onClick?: () => void; disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 shrink-0 ${destructive ? 'text-danger' : 'text-text-tertiary'}`}>{icon}</div>
        <div>
          <p className={`text-sm font-semibold ${destructive ? 'text-danger' : 'text-text-primary'}`}>{title}</p>
          <p className="text-xs text-text-tertiary mt-0.5">{description}</p>
        </div>
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={onClick}
        className={`shrink-0 px-4 py-2 text-sm font-semibold rounded-lg border transition-colors ${disabled ? 'opacity-60 cursor-not-allowed' : ''} ${
          destructive
            ? 'border-danger text-danger bg-danger-light'
            : 'border-border-default text-text-secondary bg-surface-ground'
        }`}
        title={disabled ? "Coming soon" : undefined}
      >
        {actionLabel}
      </button>
    </div>
  );
}

export function AccountSection() {
  const { doctor, clinic } = useSettingsStore();
  const { logout, resetDemoSession } = useAuthStore();
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [clearInput, setClearInput] = useState('');

  const initials = doctor.name
    ?.replace(/^Dr\.?\s+/i, '')
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map(n => n[0].toUpperCase())
    .join('');

  const handleClearData = () => {
    if (clearInput === 'CLEAR') {
      alert('Data cleared successfully (simulation)');
      setShowClearConfirm(false);
      setClearInput('');
    }
  };

  return (
    <div className="space-y-5">
      <SectionCard title="Account">
        <div className="flex items-center gap-4 p-4 bg-brand-primary-xlt rounded-xl border border-brand-primary/20">
          <div className="w-12 h-12 rounded-full bg-brand-primary text-white flex items-center justify-center font-bold text-lg shrink-0">
            {initials}
          </div>
          <div>
            <p className="font-bold text-text-primary">{doctor.name}</p>
            <p className="text-sm text-text-secondary">{doctor.qualification} · {doctor.specialisation}</p>
            <p className="text-xs text-text-tertiary mt-0.5">Reg: {doctor.regNumber} · {clinic.name}</p>
          </div>
        </div>
        <div className="space-y-4 mt-2">
          <DestructiveRow
            icon={<Phone size={16} />}
            title="Change Phone Number"
            description="Update your registered mobile number for account access."
            actionLabel="Change"
            disabled
          />
          <div className="h-px bg-border-subtle" />
          <DestructiveRow
            icon={<LogOut size={16} />}
            title="Sign Out"
            description="Sign out of your account on this device."
            actionLabel="Sign Out"
            onClick={logout}
          />
        </div>
      </SectionCard>

      <SectionCard title="Data">
        <div className="space-y-4">
          <DestructiveRow
            icon={<Download size={16} />}
            title="Export Patient Data"
            description="Download a full archive of all patient records and sessions as a JSON export."
            actionLabel="Export"
            disabled
          />
        </div>
      </SectionCard>

      <SectionCard title="Danger Zone">
        <div className="space-y-4">
          <DestructiveRow
            icon={<ShieldAlert size={16} />}
            title="Clear All Data"
            description="Permanently delete all patient data, sessions, and settings. This cannot be undone."
            actionLabel="Clear Data"
            destructive
            onClick={() => setShowClearConfirm(true)}
          />
          <DestructiveRow
            icon={<LogOut size={16} />}
            title="Reset Demo Session"
            description="[Developer] Reset the auth store to simulate a brand new unconfigured user."
            actionLabel="Reset Demo"
            destructive
            onClick={resetDemoSession}
          />
        </div>
      </SectionCard>

      {showClearConfirm && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in px-4">
          <div className="bg-surface-card w-full max-w-sm rounded-xl shadow-xl p-6 border border-border-subtle">
            <h3 className="text-lg font-bold text-danger mb-2">Clear All Data?</h3>
            <p className="text-sm text-text-secondary mb-4">
              This will permanently delete all patient records, sessions, and settings. This action <strong>cannot</strong> be undone.
            </p>
            <p className="text-xs font-semibold text-text-primary mb-2">
              Type <code className="bg-surface-ground px-1 py-0.5 rounded text-danger">CLEAR</code> to confirm.
            </p>
            <input 
              type="text" 
              value={clearInput}
              onChange={e => setClearInput(e.target.value)}
              className="w-full h-10 px-3 border border-border-subtle rounded-lg mb-6 text-sm focus:outline-none focus:border-danger focus:ring-1 focus:ring-danger"
              placeholder="CLEAR"
            />
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => { setShowClearConfirm(false); setClearInput(''); }} 
                className="px-4 py-2 text-sm font-semibold text-text-secondary hover:bg-surface-ground rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleClearData} 
                disabled={clearInput !== 'CLEAR'}
                className="px-4 py-2 text-sm font-semibold text-white bg-danger hover:bg-danger/90 rounded-lg shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Permanently Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
