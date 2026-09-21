import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import { Field, Input, Textarea, SectionCard, AvatarUpload, DiscardDialog, useToast } from '../SettingsShared';

export function ClinicProfileSection() {
  const { clinic, updateClinic } = useSettingsStore();
  const [draft, setDraft] = useState({ ...clinic });
  const [isDirty, setIsDirty] = useState(false);
  const [showDiscard, setShowDiscard] = useState(false);
  const { showToast, Toast } = useToast();

  useEffect(() => {
    setDraft({ ...clinic });
    setIsDirty(false);
  }, []);

  const set = (key: keyof typeof draft) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setDraft(d => ({ ...d, [key]: e.target.value }));
    setIsDirty(true);
  };

  const handleSave = () => {
    updateClinic(draft);
    setIsDirty(false);
    showToast('Clinic profile saved');
  };

  const handleDiscard = () => {
    if (isDirty) { setShowDiscard(true); } 
  };

  return (
    <div className="space-y-5">
      {Toast}
      {showDiscard && (
        <DiscardDialog
          onConfirm={() => { setDraft({ ...clinic }); setIsDirty(false); setShowDiscard(false); }}
          onCancel={() => setShowDiscard(false)}
        />
      )}

      <SectionCard title="Clinic Identity">
        <AvatarUpload initials={draft.name.slice(0, 2).toUpperCase()} label="Upload Logo" />
        <Field label="Clinic Name" required>
          <Input value={draft.name} onChange={set('name')} placeholder="e.g. Calm Clinic" />
        </Field>
        <Field label="Address">
          <Textarea value={draft.address} onChange={set('address')} rows={2} placeholder="Full clinic address" />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Phone">
            <Input value={draft.phone} onChange={set('phone')} placeholder="+91 20 2567 8900" />
          </Field>
          <Field label="GSTIN" hint="Optional, appears on prescriptions">
            <Input value={draft.gstin} onChange={set('gstin')} placeholder="27AAAAA0000A1Z5" />
          </Field>
        </div>
      </SectionCard>

      {/* Prescription Preview */}
      <SectionCard title="Prescription Header Preview">
        <div className="border border-border-default rounded-lg p-4 bg-surface-ground font-sans">
          <div className="flex justify-between items-start">
            <div>
              <p className="font-bold text-text-primary text-base">{draft.name || 'Clinic Name'}</p>
              <p className="text-xs text-text-secondary mt-0.5 max-w-xs">{draft.address || 'Clinic Address'}</p>
              <p className="text-xs text-text-secondary">{draft.phone}</p>
            </div>
            <div className="text-right">
              <div className="w-10 h-10 bg-brand-primary-xlt border border-brand-primary/20 rounded flex items-center justify-center text-brand-primary font-bold text-sm">
                {draft.name.slice(0, 2).toUpperCase()}
              </div>
              {draft.gstin && <p className="text-[10px] text-text-tertiary mt-1">GSTIN: {draft.gstin}</p>}
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-border-subtle flex gap-4 text-[10px] text-text-tertiary">
            <span>Date: {new Date().toLocaleDateString('en-IN')}</span>
            <span>Rx #: PREVIEW-001</span>
          </div>
        </div>
        <p className="text-xs text-text-tertiary">This is how your clinic header appears on generated prescriptions.</p>
      </SectionCard>

      <div className="flex justify-end gap-3 pt-2">
        {isDirty && (
          <button onClick={handleDiscard} className="px-5 py-2.5 text-sm font-semibold text-text-secondary hover:bg-surface-card rounded-full transition-colors">
            Discard
          </button>
        )}
        <button
          onClick={handleSave}
          disabled={!isDirty}
          className="px-6 py-2.5 text-sm font-bold text-white bg-brand-primary hover:bg-brand-primary-mid rounded-full shadow-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Save Changes
        </button>
      </div>
    </div>
  );
}
