import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import { Field, Input, SectionCard, AvatarUpload, DiscardDialog, useToast } from '../SettingsShared';

export function DoctorProfileSection() {
  const { doctor, updateDoctor } = useSettingsStore();
  const [draft, setDraft] = useState({ ...doctor });
  const [isDirty, setIsDirty] = useState(false);
  const [showDiscard, setShowDiscard] = useState(false);
  const { showToast, Toast } = useToast();

  useEffect(() => { setDraft({ ...doctor }); setIsDirty(false); }, []);

  const set = (key: keyof typeof draft) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setDraft(d => ({ ...d, [key]: e.target.value }));
    setIsDirty(true);
  };

  const handleSave = () => { updateDoctor(draft); setIsDirty(false); showToast('Doctor profile saved'); };

  return (
    <div className="space-y-5">
      {Toast}
      {showDiscard && (
        <DiscardDialog
          onConfirm={() => { setDraft({ ...doctor }); setIsDirty(false); setShowDiscard(false); }}
          onCancel={() => setShowDiscard(false)}
        />
      )}

      <SectionCard title="Doctor Identity">
        <AvatarUpload initials={draft.firstName?.slice(0, 1).toUpperCase() + (draft.name?.split(' ').pop()?.[0] || '').toUpperCase()} label="Upload Photo" />
        <div className="grid grid-cols-2 gap-4">
          <Field label="Full Name" required>
            <Input value={draft.name} onChange={set('name')} placeholder="Dr. Arjun Mehta" />
          </Field>
          <Field label="First Name" hint="Used in greetings">
            <Input value={draft.firstName} onChange={set('firstName')} placeholder="Arjun" />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Qualification" required>
            <Input value={draft.qualification} onChange={set('qualification')} placeholder="MBBS, MD" />
          </Field>
          <Field label="Reg. Number" hint="Appears on prescriptions">
            <Input value={draft.regNumber} onChange={set('regNumber')} placeholder="MH-12345" />
          </Field>
        </div>
        <Field label="Specialisation">
          <Input value={draft.specialisation} onChange={set('specialisation')} placeholder="General Physician" />
        </Field>
      </SectionCard>

      <SectionCard title="Signature">
        <div className="border-2 border-dashed border-border-default rounded-xl p-6 text-center">
          <p className="text-sm text-text-tertiary mb-3">Signature appears on printed prescriptions</p>
          <button type="button" className="text-sm font-semibold text-brand-primary hover:text-brand-primary-mid px-4 py-2 bg-brand-primary-xlt rounded-lg transition-colors">
            Upload Signature Image
          </button>
          <p className="text-xs text-text-tertiary mt-2">PNG or SVG, transparent background, max 500KB</p>
        </div>
      </SectionCard>

      <div className="flex justify-end gap-3 pt-2">
        {isDirty && (
          <button onClick={() => setShowDiscard(true)} className="px-5 py-2.5 text-sm font-semibold text-text-secondary hover:bg-surface-card rounded-full transition-colors">
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
