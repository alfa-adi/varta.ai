import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchSessions } from '../data/mock';
import { usePatientDataStore } from '../stores/patientDataStore';
import { TopBar } from '../components/layout/TopBar';
import { ProfileHeader } from '../components/patients/ProfileHeader';
import { TimelineStrip } from '../components/patients/TimelineStrip';
import { SessionDetail } from '../components/patients/SessionDetail';
import { FAB } from '../components/ui/FAB';
import { ArrowLeft, Plus } from 'lucide-react';
import { useRecordingStore } from '../stores/recordingStore';

export function PatientProfile() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [fabExpanded, setFabExpanded] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const openLauncher = useRecordingStore(state => state.openLauncher);

  // Smart back: use browser history if available, fall back to /patients
  const handleBack = () => {
    if (window.history.length > 2) {
      navigate(-1);
    } else {
      navigate('/patients');
    }
  };
  
  const getPatientById = usePatientDataStore(state => state.getPatientById);
  const patient = id ? getPatientById(id) : undefined;
  const patientLoading = false;
  const patientError = !!id && !patient;

  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['sessions', id],
    queryFn: () => fetchSessions(id!),
    enabled: !!id,
  });

  // Default to the latest session (index 0 if newest first) when sessions load
  React.useEffect(() => {
    if (sessions && sessions.length > 0 && !selectedSessionId) {
      setSelectedSessionId(sessions[0].id);
    }
  }, [sessions, selectedSessionId]);

  if (patientLoading) {
    return (
      <div className="min-h-screen bg-surface-ground flex items-center justify-center">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-12 rounded-full bg-brand-primary/20 mb-4" />
          <p className="text-text-secondary font-medium">Loading patient profile...</p>
        </div>
      </div>
    );
  }

  if (patientError || !patient) {
    return (
      <div className="min-h-screen bg-surface-ground flex flex-col items-center justify-center">
        <p className="text-danger font-medium mb-4">Patient not found.</p>
        <button onClick={() => navigate('/patients')} className="text-brand-primary underline hover:text-brand-primary-mid">
          Return to Patients Directory
        </button>
      </div>
    );
  }

  const selectedSession = sessions?.find(s => s.id === selectedSessionId);
  const isLatestSession = sessions && sessions.length > 0 ? sessions[0].id === selectedSessionId : false;

  return (
    <div className="min-h-screen flex flex-col bg-surface-ground animate-fade-in relative">
      
      {/* Top Bar with Back Button */}
      <header className="h-16 bg-surface-card border-b border-border-subtle flex items-center px-4 md:px-6 sticky top-0 z-30 shrink-0">
        <button 
          onClick={handleBack}
          aria-label="Go back"
          className="flex items-center gap-2 text-text-secondary hover:text-brand-primary transition-colors font-medium mr-6 focus-ring rounded"
        >
          <ArrowLeft size={20} />
          <span className="hidden sm:inline">Back</span>
        </button>
        <div className="h-8 w-px bg-border-subtle mx-4 hidden md:block"></div>
        <h1 className="text-lg font-bold text-text-primary hidden md:block">
          Patient Profile
        </h1>
      </header>

      {/* Main Profile Layout */}
      <div className="flex-1 flex flex-col">
        <ProfileHeader patient={patient} />
        
        {sessionsLoading ? (
          <div className="h-24 flex items-center justify-center border-b border-border-subtle bg-surface-ground/50">
            <div className="animate-pulse h-8 w-64 bg-border-subtle rounded-full" />
          </div>
        ) : sessions && sessions.length > 0 ? (
          <>
            <TimelineStrip 
              sessions={sessions} 
              selectedSessionId={selectedSessionId || sessions[0].id}
              onSelectSession={setSelectedSessionId}
              onNewSession={() => openLauncher(id)}
            />
            
            <main className="flex-1 p-6 overflow-y-auto">
              {selectedSession ? (
                <SessionDetail 
                  session={selectedSession} 
                  isLatest={isLatestSession}
                />
              ) : (
                <div className="text-center py-12 text-text-tertiary">Select a session to view details</div>
              )}
            </main>
          </>
        ) : (
          <main className="flex-1 p-6 flex flex-col items-center justify-center">
            <div className="bg-surface-card p-8 rounded-xl shadow-sm border border-border-subtle text-center max-w-md w-full">
              <div className="h-16 w-16 bg-brand-primary-xlt text-brand-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <Plus size={32} />
              </div>
              <h3 className="text-lg font-bold text-text-primary mb-2">No past sessions</h3>
              <p className="text-sm text-text-secondary mb-6">This patient doesn't have any recorded sessions yet.</p>
              <button 
                onClick={() => openLauncher(id)}
                className="w-full py-3 bg-brand-primary text-text-on-brand font-semibold rounded-lg shadow-sm hover:bg-brand-primary-mid transition-colors"
              >
                Start First Consultation
              </button>
            </div>
          </main>
        )}
      </div>

      <FAB 
        icon={<Plus size={24} />} 
        label="New Session" 
        expanded={fabExpanded}
        onMouseEnter={() => setFabExpanded(true)}
        onMouseLeave={() => setFabExpanded(false)}
        onClick={() => openLauncher(id)}
      />
    </div>
  );
}
