import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchPatient, fetchSessions } from '../../data/mock';
import type { PrescribedMedicine, Session } from '../../types';
import { PrescriptionDocument } from './PrescriptionDocument';
import { ArrowLeft, Printer, Share2, Download, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '../../utils/cn';

export function PrescriptionPage() {
  const { patientId, sessionId } = useParams<{ patientId: string, sessionId: string }>();
  const navigate = useNavigate();

  const { data: patient, isLoading: patientLoading } = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => fetchPatient(patientId!),
    enabled: !!patientId,
  });

  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['sessions', patientId],
    queryFn: () => fetchSessions(patientId!),
    enabled: !!patientId,
  });

  // Local state for the prescription edits
  const [isExtracted, setIsExtracted] = useState(true);
  
  const [medicines, setMedicines] = useState<PrescribedMedicine[]>([]);
  const [tests, setTests] = useState<string[]>([]);
  const [advice, setAdvice] = useState('');
  const [chiefComplaint, setChiefComplaint] = useState('');
  const [followUp, setFollowUp] = useState('');

  // Extract from session on load
  useEffect(() => {
    if (sessions && sessionId) {
      const session = sessions.find((s: Session) => s.id === sessionId);
      if (session && isExtracted) {
        setChiefComplaint(session.chiefComplaint || '');
        setAdvice(session.prescription?.advice || '');
        setFollowUp(session.prescription?.followUpDate || session.followUpDate || '');
        
        if (session.prescription?.medicines) {
          setMedicines(session.prescription.medicines);
        }
        
        if (session.prescription?.tests) {
          setTests(session.prescription.tests.map(t => t.label));
        }
      }
    }
  }, [sessions, sessionId, isExtracted]);

  // Handle mode toggle (Extracted vs Manual)
  const handleToggleMode = (extracted: boolean) => {
    setIsExtracted(extracted);
    if (!extracted) {
      // Clear for manual entry
      setMedicines([]);
      setTests([]);
      setAdvice('');
      setChiefComplaint('');
      setFollowUp('');
    }
  };

  const isLoading = patientLoading || sessionsLoading;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-surface-ground flex items-center justify-center">
        <div className="text-text-secondary">Loading prescription...</div>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="min-h-screen bg-surface-ground flex items-center justify-center">
        <div className="text-center">
          <AlertCircle size={48} className="text-text-tertiary mx-auto mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">Patient not found</h2>
          <button onClick={() => navigate('/')} className="text-brand-primary font-medium hover:underline">Go to Dashboard</button>
        </div>
      </div>
    );
  }

  const session = sessions?.find((s: Session) => s.id === sessionId);

  if (!session) {
    return (
      <div className="min-h-screen bg-surface-ground flex items-center justify-center">
        <div className="text-center">
          <AlertCircle size={48} className="text-text-tertiary mx-auto mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">Session not found</h2>
          <p className="text-text-secondary mb-6">The session ID <code className="bg-surface-ground px-2 py-1 rounded text-sm">{sessionId}</code> does not exist for this patient.</p>
          <button onClick={() => navigate(`/patients/${patientId}`)} className="text-brand-primary font-medium hover:underline">← Back to Patient Profile</button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-ground flex flex-col pb-24">
      
      {/* Top Header */}
      <header className="bg-surface-card border-b border-border-subtle sticky top-0 z-20 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between px-4 md:px-6 py-4 gap-4">
          <div className="flex items-center gap-3 md:gap-4">
            <button 
              onClick={() => navigate(-1)}
              className="h-10 w-10 rounded-full flex items-center justify-center hover:bg-surface-ground text-text-secondary transition-colors shrink-0"
            >
              <ArrowLeft size={20} />
            </button>
            <div className="min-w-0">
              <h1 className="text-lg md:text-xl font-bold text-text-primary truncate">Digital Prescription</h1>
              <p className="text-xs md:text-sm text-text-secondary truncate">For {patient.name} • {session.shortDate}</p>
            </div>
          </div>
          
          <div className="flex bg-surface-ground p-1 rounded-lg border border-border-subtle overflow-x-auto shrink-0">
            <button 
              onClick={() => handleToggleMode(true)}
              className={cn(
                "flex-1 md:flex-none px-3 md:px-4 py-1.5 rounded-md text-xs md:text-sm font-semibold transition-colors shadow-sm whitespace-nowrap",
                isExtracted ? "bg-surface-card text-text-primary" : "text-text-secondary hover:text-text-primary"
              )}
            >
              Extracted
            </button>
            <button 
              onClick={() => handleToggleMode(false)}
              className={cn(
                "flex-1 md:flex-none px-3 md:px-4 py-1.5 rounded-md text-xs md:text-sm font-semibold transition-colors shadow-sm whitespace-nowrap",
                !isExtracted ? "bg-surface-card text-text-primary" : "text-text-secondary hover:text-text-primary"
              )}
            >
              Manual entry
            </button>
          </div>
        </div>
      </header>

      {/* Main Document Area */}
      <main className="flex-1 p-4 md:p-8 overflow-x-hidden">
        <PrescriptionDocument 
          patient={patient}
          session={session}
          isExtracted={isExtracted}
          medicines={medicines}
          onAddMedicine={(m) => setMedicines([...medicines, m])}
          onRemoveMedicine={(id) => setMedicines(medicines.filter(m => m.id !== id))}
          tests={tests}
          onAddTest={(t) => setTests([...tests, t])}
          onRemoveTest={(t) => setTests(tests.filter(test => test !== t))}
          advice={advice}
          onChangeAdvice={setAdvice}
          followUp={followUp}
          onChangeFollowUp={setFollowUp}
          chiefComplaint={chiefComplaint}
          onChangeChiefComplaint={setChiefComplaint}
        />
      </main>

      {/* Sticky Bottom Action Bar */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-surface-card border-t border-border-subtle flex flex-wrap justify-between items-center gap-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-20 pb-safe">
        <div className="flex gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0 hide-scrollbar">
          <button className="flex items-center gap-2 px-4 py-2 md:px-5 md:py-3 border border-border-subtle text-text-primary font-semibold rounded-lg hover:bg-surface-ground transition-colors whitespace-nowrap">
            <Printer size={18} />
            <span className="hidden sm:inline">Print</span>
          </button>
          <button className="flex items-center gap-2 px-4 py-2 md:px-5 md:py-3 border border-brand-primary text-brand-primary font-semibold rounded-lg hover:bg-brand-primary-light transition-colors whitespace-nowrap">
            <Share2 size={18} />
            <span className="hidden sm:inline">WhatsApp</span>
          </button>
          <button className="flex items-center gap-2 px-4 py-2 md:px-5 md:py-3 border border-border-subtle text-text-primary font-semibold rounded-lg hover:bg-surface-ground transition-colors whitespace-nowrap">
            <Download size={18} />
            <span className="hidden sm:inline">Download</span>
          </button>
          
          <button 
            onClick={() => navigate(`/patients/${patient.id}`)}
            className="md:hidden flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-brand-primary text-text-on-brand font-bold rounded-lg hover:bg-brand-primary-mid transition-colors shadow-sm whitespace-nowrap ml-auto"
          >
            <CheckCircle2 size={18} />
            Save
          </button>
        </div>
        
        <button 
          onClick={() => navigate(`/patients/${patient.id}`)}
          className="hidden md:flex items-center gap-2 px-8 py-3 bg-brand-primary text-text-on-brand font-bold rounded-lg hover:bg-brand-primary-mid transition-colors shadow-sm"
        >
          <CheckCircle2 size={20} />
          Attach & Save
        </button>
      </div>

    </div>
  );
}
