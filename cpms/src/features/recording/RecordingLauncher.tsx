import React, { useState, useEffect } from 'react';
import { useRecordingStore } from '../../stores/recordingStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useQuery } from '@tanstack/react-query';
import { fetchPatients } from '../../data/mock';
import { X, Mic, Languages, ArrowRight, User } from 'lucide-react';
import { cn } from '../../utils/cn';

export function RecordingLauncher() {
  const { launcherOpen, closeLauncher, mode, setMode, currentPatientId, setPatient, startRecording } = useRecordingStore();
  const { recording } = useSettingsStore();

  const [docLang, setDocLang] = useState(recording.defaultDoctorLanguage);
  const [patLang, setPatLang] = useState(recording.defaultPatientLanguage);

  useEffect(() => {
    if (launcherOpen) {
      setMode(recording.defaultMode);
      setDocLang(recording.defaultDoctorLanguage);
      setPatLang(recording.defaultPatientLanguage);
    }
  }, [launcherOpen, recording.defaultMode, recording.defaultDoctorLanguage, recording.defaultPatientLanguage, setMode]);

  const { data: patients } = useQuery({
    queryKey: ['patients', 'today'],
    queryFn: fetchPatients,
  });

  if (!launcherOpen) return null;

  const handleStart = () => {
    if (currentPatientId) {
      startRecording();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col justify-end"
      role="dialog"
      aria-modal="true"
      aria-label="Start New Session"
    >
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={closeLauncher}
        aria-hidden="true"
      />
      
      {/* Bottom Sheet */}
      <div className="relative bg-surface-ground w-full h-[70vh] rounded-t-3xl shadow-2xl flex flex-col animate-slide-up">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border-subtle bg-surface-card rounded-t-3xl shrink-0">
          <div>
            <h2 className="text-xl font-bold text-text-primary">Start New Session</h2>
            <p className="text-sm text-text-secondary mt-1">Configure consultation mode</p>
          </div>
          <button 
            onClick={closeLauncher}
            aria-label="Close session launcher"
            className="p-2 text-text-tertiary hover:text-text-primary hover:bg-surface-ground rounded-full transition-colors focus-ring"
          >
            <X size={24} />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {/* Patient Selection (if none selected) */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-text-tertiary uppercase tracking-wider">Patient</h3>
            {currentPatientId ? (
              <div className="flex items-center gap-4 p-4 border border-border-subtle rounded-xl bg-surface-card shadow-sm">
                <div className="h-12 w-12 rounded-full bg-brand-primary text-text-on-brand flex items-center justify-center text-xl font-bold">
                  {patients?.find(p => p.id === currentPatientId)?.name.charAt(0) || 'P'}
                </div>
                <div>
                  <p className="font-bold text-text-primary text-lg">
                    {patients?.find(p => p.id === currentPatientId)?.name || 'Unknown Patient'}
                  </p>
                  <button onClick={() => setPatient('')} className="text-sm text-brand-primary font-medium hover:underline">
                    Change Patient
                  </button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {patients?.map(p => (
                  <button 
                    key={p.id}
                    onClick={() => setPatient(p.id)}
                    className="flex items-center gap-3 p-3 border border-border-subtle rounded-xl bg-surface-card hover:border-brand-primary hover:bg-brand-primary-xlt transition-colors text-left"
                  >
                    <div className="h-10 w-10 rounded-full bg-surface-ground flex items-center justify-center text-text-secondary font-bold">
                      {p.name.charAt(0)}
                    </div>
                    <div>
                      <p className="font-semibold text-text-primary text-sm">{p.name}</p>
                      <p className="text-xs text-text-tertiary">{p.patientId}</p>
                    </div>
                  </button>
                ))}
                <button className="flex items-center gap-3 p-3 border border-dashed border-border-subtle rounded-xl text-text-secondary hover:text-brand-primary hover:border-brand-primary transition-colors justify-center">
                  <User size={20} />
                  <span className="font-medium text-sm">Search other patients</span>
                </button>
              </div>
            )}
          </div>

          {/* Mode Selection */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-text-tertiary uppercase tracking-wider">Consultation Mode</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Transcribe Mode */}
              <button 
                onClick={() => setMode('Transcribe')}
                className={cn(
                  "p-5 rounded-2xl border-2 text-left transition-all relative overflow-hidden",
                  mode === 'Transcribe' 
                    ? "border-brand-primary bg-brand-primary-xlt shadow-md" 
                    : "border-border-subtle bg-surface-card hover:border-brand-primary-light"
                )}
              >
                <div className={cn(
                  "h-10 w-10 rounded-full flex items-center justify-center mb-4 transition-colors",
                  mode === 'Transcribe' ? "bg-brand-primary text-text-on-brand" : "bg-surface-ground text-text-secondary"
                )}>
                  <Mic size={20} />
                </div>
                <h4 className={cn("text-lg font-bold mb-1", mode === 'Transcribe' ? "text-brand-primary" : "text-text-primary")}>Transcribe</h4>
                <p className="text-sm text-text-secondary mb-4">Captures conversation in the same language. Differentiates doctor and patient voices.</p>
                
                {mode === 'Transcribe' && (
                  <select 
                    value={docLang} 
                    onChange={e => setDocLang(e.target.value)}
                    className="w-full p-2.5 rounded-lg border border-brand-primary-light bg-surface-card text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand-primary" 
                    onClick={e => e.stopPropagation()}
                  >
                    <option value="English">English</option>
                    <option value="Hindi">Hindi</option>
                    <option value="Gujarati">Gujarati</option>
                    <option value="Marathi">Marathi</option>
                  </select>
                )}
              </button>

              {/* Translate Mode */}
              <button 
                onClick={() => setMode('Translate')}
                className={cn(
                  "p-5 rounded-2xl border-2 text-left transition-all relative overflow-hidden",
                  mode === 'Translate' 
                    ? "border-brand-primary bg-brand-primary-xlt shadow-md" 
                    : "border-border-subtle bg-surface-card hover:border-brand-primary-light"
                )}
              >
                <div className={cn(
                  "h-10 w-10 rounded-full flex items-center justify-center mb-4 transition-colors",
                  mode === 'Translate' ? "bg-brand-primary text-text-on-brand" : "bg-surface-ground text-text-secondary"
                )}>
                  <Languages size={20} />
                </div>
                <h4 className={cn("text-lg font-bold mb-1", mode === 'Translate' ? "text-brand-primary" : "text-text-primary")}>Translate</h4>
                <p className="text-sm text-text-secondary mb-4">Real-time bilingual transcription. Speak your language, prints in theirs.</p>
                
                {mode === 'Translate' && (
                  <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                    <select 
                      value={docLang}
                      onChange={e => setDocLang(e.target.value)}
                      className="flex-1 p-2.5 rounded-lg border border-brand-primary-light bg-surface-card text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand-primary"
                    >
                      <option value="English">English</option>
                      <option value="Hindi">Hindi</option>
                      <option value="Marathi">Marathi</option>
                    </select>
                    <ArrowRight size={16} className="text-brand-primary shrink-0" />
                    <select 
                      value={patLang}
                      onChange={e => setPatLang(e.target.value)}
                      className="flex-1 p-2.5 rounded-lg border border-brand-primary-light bg-surface-card text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-brand-primary"
                    >
                      <option value="Hindi">Hindi</option>
                      <option value="Gujarati">Gujarati</option>
                      <option value="Marathi">Marathi</option>
                    </select>
                  </div>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Footer Action */}
        <div className="p-6 bg-surface-card border-t border-border-subtle shrink-0">
          <button 
            disabled={!currentPatientId}
            onClick={handleStart}
            className="w-full py-4 rounded-xl bg-brand-primary text-text-on-brand font-bold text-lg shadow-sm hover:bg-brand-primary-mid transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Mic size={20} />
            Start Recording
          </button>
        </div>

      </div>
    </div>
  );
}
