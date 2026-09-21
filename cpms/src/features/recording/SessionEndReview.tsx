import React from 'react';
import { useRecordingStore } from '../../stores/recordingStore';
import { useQuery } from '@tanstack/react-query';
import { fetchPatient, fetchSessions } from '../../data/mock';
import { TagPill } from '../../components/ui/TagPill';
import { CheckCircle2, ChevronDown, Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export function SessionEndReview() {
  const { isReviewing, elapsedSeconds, mode, currentPatientId, liveTranscript, discardSession, saveSession } = useRecordingStore();
  const navigate = useNavigate();

  const { data: patient } = useQuery({
    queryKey: ['patient', currentPatientId],
    queryFn: () => fetchPatient(currentPatientId!),
    enabled: !!currentPatientId,
  });

  const { data: sessions } = useQuery({
    queryKey: ['sessions', currentPatientId],
    queryFn: () => fetchSessions(currentPatientId!),
    enabled: !!currentPatientId,
  });

  if (!isReviewing) return null;

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    return `${m} min`;
  };

  const handleSave = () => {
    saveSession();
    if (currentPatientId) {
      navigate(`/patients/${currentPatientId}`);
    } else {
      navigate('/');
    }
  };

  const handleGeneratePrescription = () => {
    saveSession();
    if (currentPatientId) {
      // Navigate to the most recent real session's prescription
      const latestSessionId = sessions?.[0]?.id || 's-1-1';
      navigate(`/patients/${currentPatientId}/prescription/${latestSessionId}`);
    }
  };

  const handleDiscard = () => {
    if (window.confirm('Are you sure you want to discard this session?')) {
      discardSession();
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-surface-ground flex flex-col animate-fade-in overflow-hidden">
      
      {/* Header */}
      <div className="bg-surface-card border-b border-border-subtle p-6 shrink-0 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-accent-green-light text-accent-green flex items-center justify-center">
            <CheckCircle2 size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-text-primary">Session Complete</h1>
            <p className="text-sm text-text-secondary">
              {patient?.name} • {formatTime(elapsedSeconds)} • {mode}
            </p>
          </div>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6 max-w-4xl mx-auto w-full pb-32">
        <div className="space-y-6">
          
          {/* AI Summary Card */}
          <div className="bg-surface-card border border-border-subtle rounded-xl shadow-sm overflow-hidden">
            <div className="bg-brand-primary-xlt border-b border-border-subtle p-4">
              <h2 className="font-bold text-brand-primary">AI Generated Summary</h2>
              <p className="text-sm text-text-secondary">Review and edit before saving</p>
            </div>
            
            <div className="p-5 space-y-6">
              
              <div>
                <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Chief Complaint</label>
                <input 
                  type="text" 
                  className="w-full p-3 rounded-lg border border-border-subtle bg-surface-ground text-text-primary focus:outline-none focus:border-brand-primary" 
                  defaultValue="Mild fever and cough for 2 days."
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="flex items-center justify-between text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">
                    Symptoms <button className="text-brand-primary hover:underline"><Plus size={14}/></button>
                  </label>
                  <div className="flex flex-wrap gap-2 p-3 min-h-[48px] rounded-lg border border-border-subtle bg-surface-ground">
                    <TagPill variant="symptom">Fever ✕</TagPill>
                    <TagPill variant="symptom">Cough ✕</TagPill>
                  </div>
                </div>

                <div>
                  <label className="flex items-center justify-between text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">
                    Medicines <button className="text-brand-primary hover:underline"><Plus size={14}/></button>
                  </label>
                  <div className="flex flex-wrap gap-2 p-3 min-h-[48px] rounded-lg border border-border-subtle bg-surface-ground">
                    <TagPill variant="medicine">Paracetamol 500mg ✕</TagPill>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Vitals</label>
                <div className="flex gap-4">
                  <div className="flex-1 flex items-center bg-surface-ground border border-border-subtle rounded-lg px-3 py-2">
                    <span className="text-xs text-text-secondary w-12">BP</span>
                    <input type="text" className="bg-transparent border-none focus:outline-none w-full text-sm font-bold text-text-primary" placeholder="e.g. 120/80" />
                  </div>
                  <div className="flex-1 flex items-center bg-surface-ground border border-border-subtle rounded-lg px-3 py-2">
                    <span className="text-xs text-text-secondary w-12">Temp</span>
                    <input type="text" className="bg-transparent border-none focus:outline-none w-full text-sm font-bold text-text-primary" placeholder="e.g. 98.6" defaultValue="99.2" />
                  </div>
                  <div className="flex-1 flex items-center bg-surface-ground border border-border-subtle rounded-lg px-3 py-2">
                    <span className="text-xs text-text-secondary w-12">SpO2</span>
                    <input type="text" className="bg-transparent border-none focus:outline-none w-full text-sm font-bold text-text-primary" placeholder="e.g. 98%" />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Doctor's Notes</label>
                <textarea 
                  className="w-full p-3 rounded-lg border border-border-subtle bg-surface-ground text-text-primary focus:outline-none focus:border-brand-primary min-h-[100px] resize-y" 
                  defaultValue="BP is stable. Mild viral infection suspected. Advised rest and hydration."
                />
              </div>

            </div>
          </div>

          {/* Follow up */}
          <div className="bg-surface-card border border-border-subtle rounded-xl shadow-sm p-5">
             <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-3">Schedule Follow-up</label>
             <div className="flex flex-wrap gap-3">
               <button className="px-4 py-2 rounded-lg border border-border-subtle bg-surface-ground text-text-secondary text-sm hover:border-brand-primary hover:text-brand-primary">None</button>
               <button className="px-4 py-2 rounded-lg border border-brand-primary bg-brand-primary-light text-brand-primary font-medium text-sm">+3 Days</button>
               <button className="px-4 py-2 rounded-lg border border-border-subtle bg-surface-ground text-text-secondary text-sm hover:border-brand-primary hover:text-brand-primary">+7 Days</button>
               <button className="px-4 py-2 rounded-lg border border-border-subtle bg-surface-ground text-text-secondary text-sm hover:border-brand-primary hover:text-brand-primary">+14 Days</button>
               <button className="px-4 py-2 rounded-lg border border-border-subtle bg-surface-ground text-text-secondary text-sm hover:border-brand-primary hover:text-brand-primary">+1 Month</button>
             </div>
          </div>

          {/* Transcript Collapsible */}
          <div className="bg-surface-card border border-border-subtle rounded-xl shadow-sm overflow-hidden">
            <button className="w-full flex items-center justify-between p-4 hover:bg-surface-ground transition-colors">
              <span className="font-bold text-text-primary">View Full Transcript</span>
              <ChevronDown size={20} className="text-text-tertiary" />
            </button>
          </div>

        </div>
      </div>

      {/* Sticky Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-surface-card border-t border-border-subtle flex justify-between items-center shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-20">
        <button 
          onClick={handleDiscard}
          className="px-6 py-3 text-danger font-semibold hover:bg-danger-light rounded-lg transition-colors"
        >
          Discard Session
        </button>
        <div className="flex gap-3">
          <button 
            onClick={handleGeneratePrescription}
            className="px-6 py-3 border border-brand-primary text-brand-primary font-bold rounded-lg hover:bg-brand-primary-light transition-colors"
          >
            Generate Prescription
          </button>
          <button 
            onClick={handleSave}
            className="px-8 py-3 bg-brand-primary text-text-on-brand font-bold rounded-lg hover:bg-brand-primary-mid transition-colors shadow-sm"
          >
            Save Session
          </button>
        </div>
      </div>

    </div>
  );
}
