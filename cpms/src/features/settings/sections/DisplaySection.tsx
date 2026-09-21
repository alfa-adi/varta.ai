import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import type { DisplaySettings } from '../../../stores/settingsStore';
import { SectionCard, SegmentedControl, Toggle, Select, Field, DiscardDialog, useToast } from '../SettingsShared';

const UI_LANGUAGES = ['English','Hindi','Marathi','Gujarati','Tamil','Telugu','Kannada','Bengali','Malayalam','Punjabi'];

export function DisplaySection() {
  const { display, updateDisplay } = useSettingsStore();
  const [draft, setDraft] = useState<DisplaySettings>({ ...display });
  const [isDirty, setIsDirty] = useState(false);
  const [showDiscard, setShowDiscard] = useState(false);
  const { showToast, Toast } = useToast();

  useEffect(() => { setDraft({ ...display }); setIsDirty(false); }, []);

  const set = <K extends keyof DisplaySettings>(key: K, value: DisplaySettings[K]) => {
    setDraft(d => ({ ...d, [key]: value }));
    setIsDirty(true);
  };

  const handleSave = () => { updateDisplay(draft); setIsDirty(false); showToast('Display settings saved'); };

  const sampleDate = new Date(2026, 5, 12); // 12 June 2026
  const formatPreview = () => {
    if (draft.dateFormat === 'DD/MM/YYYY') return '12/06/2026';
    if (draft.dateFormat === 'MM/DD/YYYY') return '06/12/2026';
    return '2026-06-12';
  };

  return (
    <div className="space-y-5">
      {Toast}
      {showDiscard && (
        <DiscardDialog
          onConfirm={() => { setDraft({ ...display }); setIsDirty(false); setShowDiscard(false); }}
          onCancel={() => setShowDiscard(false)}
        />
      )}

      <SectionCard title="Language">
        <Field label="Interface Language" hint="Changes the UI language across the app (Available after restart)">
          <Select value={draft.uiLanguage} onChange={e => set('uiLanguage', e.target.value)}>
            {UI_LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
          </Select>
        </Field>
      </SectionCard>

      <SectionCard title="Date & Time">
        <Field label="Date Format">
          <SegmentedControl
            options={[
              { label: 'DD/MM/YYYY', value: 'DD/MM/YYYY' },
              { label: 'MM/DD/YYYY', value: 'MM/DD/YYYY' },
              { label: 'ISO (YYYY-MM-DD)', value: 'YYYY-MM-DD' },
            ]}
            value={draft.dateFormat}
            onChange={v => set('dateFormat', v as any)}
          />
          <p className="text-xs text-text-tertiary mt-1.5">Preview: <span className="font-semibold text-text-secondary">{formatPreview()}</span></p>
        </Field>
        <Field label="Time Format">
          <SegmentedControl
            options={[{ label: '12-hour (2:30 PM)', value: '12h' }, { label: '24-hour (14:30)', value: '24h' }]}
            value={draft.timeFormat}
            onChange={v => set('timeFormat', v as any)}
          />
        </Field>
      </SectionCard>

      <SectionCard title="Accessibility">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-text-primary">High-Contrast Mode</p>
            <p className="text-xs text-text-tertiary mt-0.5">Increases contrast for better visibility. Coming soon.</p>
          </div>
          <Toggle checked={draft.highContrast} onChange={v => set('highContrast', v)} />
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
