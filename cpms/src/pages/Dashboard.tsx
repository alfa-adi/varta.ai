import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../components/layout/AppShell';
import { PatientCard } from '../components/patients/PatientCard';
import { FAB } from '../components/ui/FAB';
import { fetchStats } from '../data/mock';
import { useRecordingStore } from '../stores/recordingStore';
import { usePatientFormStore } from '../stores/patientFormStore';
import { usePatientDataStore } from '../stores/patientDataStore';
import { Users, CheckCircle2, Clock, CalendarDays, Mic, Plus, UserPlus } from 'lucide-react';
import { cn } from '../utils/cn';

export function Dashboard() {
  const navigate = useNavigate();
  const [fabExpanded, setFabExpanded] = useState(false);
  const openLauncher = useRecordingStore(state => state.openLauncher);
  const openPatientForm = usePatientFormStore(state => state.openForm);

  const patients = usePatientDataStore(state => state.patients);
  const patientsLoading = false; // Synchronous from store

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
  });

  return (
    <AppShell title="Dashboard">
      <div className="space-y-6 animate-fade-in">
        {/* Dashboard Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard title="Seen Today" value={stats?.seenToday} icon={<Users size={20} />} loading={statsLoading} />
          <StatCard title="Next 2 Hours" value={stats?.nextTwoHours} icon={<Clock size={20} />} loading={statsLoading} />
          <StatCard title="Pending Rx" value={stats?.pendingRx} icon={<CheckCircle2 size={20} />} loading={statsLoading} highlight />
          <StatCard title="Due This Week" value={stats?.dueThisWeek} icon={<CalendarDays size={20} />} loading={statsLoading} />
        </div>

        {/* Patients List */}
        <div>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-text-primary">Today's Queue</h2>
            <button
              onClick={() => navigate('/patients')}
              className="text-sm font-semibold text-brand-primary hover:text-brand-primary-mid transition-colors focus-ring rounded"
            >
              View All
            </button>
          </div>

          {patientsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="bg-surface-card rounded-xl p-5 h-48 animate-pulse border border-border-subtle" />
              ))}
            </div>
          ) : patients.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="h-16 w-16 rounded-full bg-brand-primary-xlt text-brand-primary flex items-center justify-center mb-4">
                <Users size={28} />
              </div>
              <h3 className="text-base font-bold text-text-primary mb-1">No patients yet</h3>
              <p className="text-sm text-text-secondary mb-5">Add your first patient to get started.</p>
              <button
                onClick={openPatientForm}
                className="flex items-center gap-2 px-5 py-2.5 bg-brand-primary text-text-on-brand font-bold rounded-xl shadow-sm hover:bg-brand-primary-mid transition-colors"
              >
                <Plus size={16} /> Add Patient
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {patients?.map(patient => (
                <PatientCard 
                  key={patient.id} 
                  patient={patient} 
                  onClick={() => navigate(`/patients/${patient.id}`)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <div 
        className="hidden md:flex fixed bottom-6 right-6 flex-col items-end gap-3 z-50"
        onMouseEnter={() => setFabExpanded(true)}
        onMouseLeave={() => setFabExpanded(false)}
      >
        {fabExpanded && (
          <div className="flex flex-col items-end gap-2 mb-2 animate-slide-up">
            <button 
              onClick={() => { setFabExpanded(false); openPatientForm(); }}
              className="flex items-center gap-3 bg-surface-card hover:bg-surface-ground text-text-primary px-4 py-2.5 rounded-xl shadow-lg border border-border-subtle font-semibold transition-colors focus:outline-none"
            >
              <UserPlus size={18} className="text-brand-primary" />
              Add Patient
            </button>
            <button 
              onClick={() => { setFabExpanded(false); openLauncher(); }}
              className="flex items-center gap-3 bg-surface-card hover:bg-surface-ground text-text-primary px-4 py-2.5 rounded-xl shadow-lg border border-border-subtle font-semibold transition-colors focus:outline-none"
            >
              <Mic size={18} className="text-brand-primary" />
              Start Consultation
            </button>
          </div>
        )}
        <button
          aria-label={fabExpanded ? 'Close actions menu' : 'Open actions menu'}
          aria-expanded={fabExpanded}
          onClick={() => setFabExpanded(prev => !prev)}
          className={cn(
            'flex items-center justify-center bg-brand-primary text-text-on-brand shadow-fab transition-all duration-300 hover:bg-brand-primary-mid focus-ring',
            'rounded-full h-14 w-14'
          )}
        >
          <span className={cn("transition-transform duration-300", fabExpanded ? "rotate-45" : "rotate-0")}>
            <Plus size={24} />
          </span>
        </button>
      </div>
    </AppShell>
  );
}

// Helper component for Stats
function StatCard({ title, value, icon, loading, highlight = false }: { title: string; value?: number; icon: React.ReactNode; loading: boolean; highlight?: boolean }) {
  return (
    <div className={`rounded-xl p-5 border shadow-sm flex items-center justify-between transition-colors ${
      highlight ? 'bg-brand-primary-light border-brand-primary/20 text-brand-primary' : 'bg-surface-card border-border-subtle text-text-secondary'
    }`}>
      <div>
        <p className="text-sm font-medium mb-1">{title}</p>
        {loading ? (
          <div className="h-8 w-16 bg-border-subtle rounded animate-pulse" />
        ) : (
          <p className={`text-2xl font-bold ${highlight ? 'text-brand-primary' : 'text-text-primary'}`}>
            {value}
          </p>
        )}
      </div>
      <div className={`p-3 rounded-full ${highlight ? 'bg-brand-primary-xlt text-brand-primary' : 'bg-surface-ground text-text-tertiary'}`}>
        {icon}
      </div>
    </div>
  );
}
