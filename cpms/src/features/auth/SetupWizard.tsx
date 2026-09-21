import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { Check, Camera, Building2, User, Mic, ArrowRight, ArrowLeft } from 'lucide-react';
import { cn } from '../../utils/cn';

export function SetupWizard() {
  const navigate = useNavigate();
  const { completeSetup } = useAuthStore();
  const { updateDoctor, updateClinic, updateRecording, doctor, clinic, recording } = useSettingsStore();

  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  // Draft state initialized with whatever is in settingsStore
  const [draftDoctor, setDraftDoctor] = useState(doctor);
  const [draftClinic, setDraftClinic] = useState(clinic);
  const [draftRecording, setDraftRecording] = useState(recording);

  const handleNext = () => setStep(s => Math.min(3, s + 1));
  const handleBack = () => setStep(s => Math.max(1, s - 1));

  const handleFinish = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setSuccess(true);
      
      // Wait for success animation
      setTimeout(() => {
        // Commit drafts to global settings
        updateDoctor(draftDoctor);
        updateClinic(draftClinic);
        updateRecording(draftRecording);
        
        // Complete auth setup
        completeSetup();
        
        // Let AuthGuard route them to dashboard naturally, but explicitly navigating is safe
        navigate('/', { replace: true });
      }, 1500);
    }, 1000);
  };

  if (success) {
    return (
      <div className="min-h-screen bg-surface-ground flex flex-col items-center justify-center py-12 px-4">
        <div className="w-full max-w-sm bg-surface-card rounded-2xl shadow-xl border border-border-subtle p-8 flex flex-col items-center text-center animate-scale-in">
          <div className="w-16 h-16 bg-success/10 text-success rounded-full flex items-center justify-center mb-6">
            <Check size={32} />
          </div>
          <h2 className="text-2xl font-bold text-text-primary mb-2">Your clinic is ready</h2>
          <p className="text-text-secondary">Taking you to your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-ground flex flex-col items-center py-12 px-4 sm:px-6">
      
      {/* Top Progress Indicator */}
      <div className="w-full max-w-3xl mb-8 flex justify-center">
        <div className="flex items-center gap-2">
          <StepBadge num={1} label="Your Details" active={step >= 1} current={step === 1} />
          <div className={cn("w-12 h-0.5 rounded-full", step >= 2 ? "bg-brand-primary" : "bg-border-subtle")} />
          <StepBadge num={2} label="Clinic Details" active={step >= 2} current={step === 2} />
          <div className={cn("w-12 h-0.5 rounded-full", step >= 3 ? "bg-brand-primary" : "bg-border-subtle")} />
          <StepBadge num={3} label="Preferences" active={step >= 3} current={step === 3} />
        </div>
      </div>

      <div className="w-full max-w-3xl bg-surface-card rounded-2xl shadow-sm border border-border-subtle overflow-hidden flex flex-col min-h-[600px]">
        
        {/* Wizard Content */}
        <div className="flex-1 p-8 md:p-12">
          
          {step === 1 && (
            <div className="animate-fade-in max-w-xl mx-auto space-y-8">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-text-primary">Tell us about yourself</h2>
                <p className="text-text-secondary mt-2">This helps personalize your workspace and prescriptions.</p>
              </div>

              <div className="flex justify-center mb-8">
                <div className="relative group cursor-pointer">
                  <div className="w-24 h-24 rounded-full bg-surface-ground border-2 border-dashed border-border-subtle flex flex-col items-center justify-center text-text-tertiary group-hover:border-brand-primary group-hover:text-brand-primary transition-colors">
                    <User size={32} />
                  </div>
                  <div className="absolute bottom-0 right-0 p-1.5 bg-surface-card rounded-full border border-border-subtle shadow-sm">
                    <Camera size={16} className="text-text-secondary" />
                  </div>
                </div>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Full Name</label>
                  <input type="text" value={draftDoctor.name} onChange={e => setDraftDoctor({...draftDoctor, name: e.target.value})} placeholder="Dr. Jane Doe" className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Medical Reg. Number</label>
                  <input type="text" value={draftDoctor.regNumber} onChange={e => setDraftDoctor({...draftDoctor, regNumber: e.target.value})} placeholder="e.g. MH-12345" className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Specialisation</label>
                  <select value={draftDoctor.specialisation} onChange={e => setDraftDoctor({...draftDoctor, specialisation: e.target.value})} className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors">
                    <option value="">Select specialisation</option>
                    <option value="General Physician">General Physician</option>
                    <option value="Cardiologist">Cardiologist</option>
                    <option value="Pediatrician">Pediatrician</option>
                    <option value="Dermatologist">Dermatologist</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="animate-fade-in max-w-xl mx-auto space-y-8">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-text-primary">Clinic Details</h2>
                <p className="text-text-secondary mt-2">These details will appear on your generated prescriptions.</p>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Clinic Name</label>
                  <input type="text" value={draftClinic.name} onChange={e => setDraftClinic({...draftClinic, name: e.target.value})} placeholder="e.g. HealthFirst Clinic" className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Address</label>
                  <textarea value={draftClinic.address} onChange={e => setDraftClinic({...draftClinic, address: e.target.value})} placeholder="Complete clinic address" className="w-full h-24 p-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors resize-none" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Phone</label>
                    <input type="text" value={draftClinic.phone} onChange={e => setDraftClinic({...draftClinic, phone: e.target.value})} placeholder="Clinic contact" className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors" />
                  </div>
                  <div>
                    <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">GSTIN <span className="text-text-tertiary normal-case font-normal">(Optional)</span></label>
                    <input type="text" value={draftClinic.gstin} onChange={e => setDraftClinic({...draftClinic, gstin: e.target.value})} placeholder="Optional" className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors" />
                  </div>
                </div>
              </div>

              {/* Prescription Preview Snippet */}
              <div className="mt-8 p-6 rounded-xl border border-border-subtle bg-surface-ground/50">
                <p className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-4">Prescription Header Preview</p>
                <div className="bg-white p-6 border-t-8 border-t-brand-primary rounded shadow-sm">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-bold text-lg text-slate-900">{draftDoctor.name || 'Dr. Name'}</h4>
                      <p className="text-xs text-slate-500">{draftDoctor.specialisation || 'Specialisation'} • {draftDoctor.regNumber || 'Reg No'}</p>
                    </div>
                    <div className="text-right">
                      <h4 className="font-bold text-slate-900">{draftClinic.name || 'Clinic Name'}</h4>
                      <p className="text-xs text-slate-500 max-w-[200px] truncate">{draftClinic.address || 'Clinic Address'}</p>
                      <p className="text-xs text-slate-500">{draftClinic.phone || 'Phone'}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="animate-fade-in max-w-xl mx-auto space-y-8">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-text-primary">Recording Preferences</h2>
                <p className="text-text-secondary mt-2">Set up how CPMS listens and documents your sessions.</p>
              </div>

              <div className="space-y-8">
                <div className="space-y-3">
                  <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider">Default Recording Mode</label>
                  <div className="flex bg-surface-ground p-1 rounded-xl border border-border-subtle">
                    <button 
                      onClick={() => setDraftRecording({...draftRecording, defaultMode: 'Transcribe'})}
                      className={cn("flex-1 py-2.5 rounded-lg text-sm font-bold transition-all", draftRecording.defaultMode === 'Transcribe' ? 'bg-surface-card shadow-sm text-brand-primary' : 'text-text-secondary hover:text-text-primary')}
                    >
                      Transcribe (Single Language)
                    </button>
                    <button 
                      onClick={() => setDraftRecording({...draftRecording, defaultMode: 'Translate'})}
                      className={cn("flex-1 py-2.5 rounded-lg text-sm font-bold transition-all", draftRecording.defaultMode === 'Translate' ? 'bg-surface-card shadow-sm text-brand-primary' : 'text-text-secondary hover:text-text-primary')}
                    >
                      Translate (Bilingual)
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Doctor Language</label>
                    <select value={draftRecording.defaultDoctorLanguage} onChange={e => setDraftRecording({...draftRecording, defaultDoctorLanguage: e.target.value})} className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors">
                      <option>English</option>
                      <option>Hindi</option>
                      <option>Marathi</option>
                    </select>
                  </div>
                  {draftRecording.defaultMode === 'Translate' && (
                    <div className="animate-fade-in">
                      <label className="block text-xs font-bold text-text-secondary uppercase tracking-wider mb-1.5">Patient Language</label>
                      <select value={draftRecording.defaultPatientLanguage} onChange={e => setDraftRecording({...draftRecording, defaultPatientLanguage: e.target.value})} className="w-full h-12 px-4 rounded-xl border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary transition-colors">
                        <option>Hindi</option>
                        <option>Marathi</option>
                        <option>Gujarati</option>
                        <option>Tamil</option>
                      </select>
                    </div>
                  )}
                </div>

                <div className="space-y-4 pt-4 border-t border-border-subtle">
                  <label className="flex items-center justify-between p-4 bg-surface-ground border border-border-subtle rounded-xl cursor-pointer hover:border-brand-primary/50 transition-colors">
                    <div>
                      <h4 className="font-bold text-text-primary text-sm">AI Summary Generation</h4>
                      <p className="text-xs text-text-secondary mt-0.5">Automatically extract symptoms, medicines, and advice.</p>
                    </div>
                    <div className={cn("w-12 h-6 rounded-full p-1 transition-colors", draftRecording.aiSummaryEnabled ? 'bg-brand-primary' : 'bg-border-subtle')}>
                      <div className={cn("w-4 h-4 bg-white rounded-full transition-transform", draftRecording.aiSummaryEnabled ? 'translate-x-6' : 'translate-x-0')} />
                    </div>
                  </label>
                  <label className="flex items-center justify-between p-4 bg-surface-ground border border-border-subtle rounded-xl cursor-pointer hover:border-brand-primary/50 transition-colors">
                    <div>
                      <h4 className="font-bold text-text-primary text-sm">Confirm Before Ending</h4>
                      <p className="text-xs text-text-secondary mt-0.5">Show a confirmation dialog before stopping a recording.</p>
                    </div>
                    <div className={cn("w-12 h-6 rounded-full p-1 transition-colors", draftRecording.confirmBeforeEnding ? 'bg-brand-primary' : 'bg-border-subtle')} onClick={() => setDraftRecording({...draftRecording, confirmBeforeEnding: !draftRecording.confirmBeforeEnding})}>
                      <div className={cn("w-4 h-4 bg-white rounded-full transition-transform", draftRecording.confirmBeforeEnding ? 'translate-x-6' : 'translate-x-0')} />
                    </div>
                  </label>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Footer Actions */}
        <div className="p-6 border-t border-border-subtle bg-surface-card flex items-center justify-between shrink-0">
          <button 
            onClick={handleBack}
            className={cn("px-6 h-12 rounded-xl font-bold text-sm transition-colors flex items-center gap-2", step === 1 ? 'invisible' : 'text-text-secondary hover:bg-surface-ground')}
          >
            <ArrowLeft size={18} />
            Back
          </button>
          
          {step < 3 ? (
            <button 
              onClick={handleNext}
              className="px-8 h-12 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-xl font-bold text-sm shadow-sm transition-colors flex items-center gap-2"
            >
              Continue
              <ArrowRight size={18} />
            </button>
          ) : (
            <button 
              onClick={handleFinish}
              disabled={loading}
              className="px-8 h-12 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-xl font-bold text-sm shadow-sm transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Saving...
                </div>
              ) : (
                <>
                  Finish Setup
                  <Check size={18} />
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function StepBadge({ num, label, active, current }: { num: number, label: string, active: boolean, current: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={cn(
        "w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors",
        active ? "bg-brand-primary text-text-on-brand" : "bg-surface-card border border-border-subtle text-text-tertiary"
      )}>
        {active && !current ? <Check size={16} /> : num}
      </div>
      <span className={cn(
        "text-sm font-bold transition-colors hidden md:block",
        current ? "text-text-primary" : active ? "text-text-secondary" : "text-text-tertiary"
      )}>
        {label}
      </span>
    </div>
  );
}
