import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import type { NotificationSettings } from '../../../stores/settingsStore';
import { SectionCard, Toggle, Field, Input, DiscardDialog, useToast } from '../SettingsShared';

export function NotificationsSection() {
  const { notifications, updateNotifications } = useSettingsStore();
  const [draft, setDraft] = useState<NotificationSettings>({ ...notifications });
  const [isDirty, setIsDirty] = useState(false);
  const [showDiscard, setShowDiscard] = useState(false);
  const { showToast, Toast } = useToast();

  useEffect(() => { setDraft({ ...notifications }); setIsDirty(false); }, []);

  const set = <K extends keyof NotificationSettings>(key: K, value: NotificationSettings[K]) => {
    setDraft(d => ({ ...d, [key]: value }));
    setIsDirty(true);
  };

  const handleSave = () => { updateNotifications(draft); setIsDirty(false); showToast('Notification preferences saved'); };

  return (
    <div className="space-y-5">
      {Toast}
      {showDiscard && (
        <DiscardDialog
          onConfirm={() => { setDraft({ ...notifications }); setIsDirty(false); setShowDiscard(false); }}
          onCancel={() => setShowDiscard(false)}
        />
      )}

      <SectionCard title="Reminders">
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-text-primary">Follow-up Reminders</p>
              <p className="text-xs text-text-tertiary mt-0.5">Get notified when a patient's follow-up date is approaching.</p>
            </div>
            <Toggle checked={draft.followUpReminders} onChange={v => set('followUpReminders', v)} />
          </div>
          <div className="h-px bg-border-subtle" />
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-text-primary">Pending Prescription Reminders</p>
              <p className="text-xs text-text-tertiary mt-0.5">Alert when a session has no prescription attached.</p>
            </div>
            <Toggle checked={draft.pendingRxReminders} onChange={v => set('pendingRxReminders', v)} />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Quiet Hours">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-semibold text-text-primary">Enable Quiet Hours</p>
            <p className="text-xs text-text-tertiary mt-0.5">Suppress all notifications during the set time window.</p>
          </div>
          <Toggle checked={draft.quietHoursEnabled} onChange={v => set('quietHoursEnabled', v)} />
        </div>
        {draft.quietHoursEnabled && (
          <div className="grid grid-cols-2 gap-4 mt-2 animate-fade-in">
            <Field label="Quiet from">
              <Input type="time" value={draft.quietHoursStart} onChange={e => set('quietHoursStart', e.target.value)} />
            </Field>
            <Field label="Quiet until">
              <Input type="time" value={draft.quietHoursEnd} onChange={e => set('quietHoursEnd', e.target.value)} />
            </Field>
          </div>
        )}
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
