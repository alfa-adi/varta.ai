import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import type { RecordingSettings } from '../../../stores/settingsStore';
import { SectionCard, SegmentedControl, Toggle, Select, Field, DiscardDialog, useToast } from '../SettingsShared';

const LANGUAGES = ['English','Hindi','Marathi','Gujarati','Tamil','Telugu','Kannada','Bengali','Malayalam','Punjabi','Odia','Assamese','Urdu','Kashmiri'];

export function RecordingSection() {
  const { recording, updateRecording } = useSettingsStore();
  const [draft, setDraft] = useState<RecordingSettings>({ ...recording });
  const [isDirty, setIsDirty] = useState(false);
  const [showDiscard, setShowDiscard] = useState(false);
  const { showToast, Toast } = useToast();

  useEffect(() => { setDraft({ ...recording }); setIsDirty(false); }, []);

  const set = <K extends keyof RecordingSettings>(key: K, value: RecordingSettings[K]) => {
    setDraft(d => ({ ...d, [key]: value }));
    setIsDirty(true);
  };

  const handleSave = () => { updateRecording(draft); setIsDirty(false); showToast('Recording preferences saved'); };

  return (
    <div className="space-y-5">
      {Toast}
      {showDiscard && (
        <DiscardDialog
          onConfirm={() => { setDraft({ ...recording }); setIsDirty(false); setShowDiscard(false); }}
          onCancel={() => setShowDiscard(false)}
        />
      )}

      <SectionCard title="Default Recording Mode">
        <div className="space-y-1.5">
          <p className="text-sm font-semibold text-text-secondary">Mode</p>
          <SegmentedControl
            options={[{ label: 'Transcribe', value: 'Transcribe' }, { label: 'Translate', value: 'Translate' }]}
            value={draft.defaultMode}
            onChange={(v) => set('defaultMode', v as any)}
          />
          <p className="text-xs text-text-tertiary mt-1">
            {draft.defaultMode === 'Transcribe' ? 'Converts speech to text in the same language.' : 'Converts speech and translates it to your language.'}
          </p>
        </div>
        <Field label="Your (Doctor) Language">
          <Select value={draft.defaultDoctorLanguage} onChange={e => set('defaultDoctorLanguage', e.target.value)}>
            {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
          </Select>
        </Field>
        {draft.defaultMode === 'Translate' && (
          <Field label="Default Patient Language" hint="Session transcripts will be translated into your language.">
            <Select value={draft.defaultPatientLanguage} onChange={e => set('defaultPatientLanguage', e.target.value)}>
              {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
            </Select>
          </Field>
        )}
      </SectionCard>

      <SectionCard title="AI & Session Behaviour">
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-text-primary">AI-Generated Summary</p>
              <p className="text-xs text-text-tertiary mt-0.5">Auto-generate SOAP notes after each session ends.</p>
            </div>
            <Toggle checked={draft.aiSummaryEnabled} onChange={v => set('aiSummaryEnabled', v)} />
          </div>
          <div className="h-px bg-border-subtle" />
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-text-primary">Confirm Before Ending</p>
              <p className="text-xs text-text-tertiary mt-0.5">Show a confirmation prompt before stopping the recording.</p>
            </div>
            <Toggle checked={draft.confirmBeforeEnding} onChange={v => set('confirmBeforeEnding', v)} />
          </div>
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
