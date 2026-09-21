import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MoreVertical, Mic, Clock, Archive, User as UserIcon, Calendar, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '../../../utils/cn';
import type { Patient, Urgency } from '../../../types';
import { useRecordingStore } from '../../../stores/recordingStore';
import { usePatientDataStore } from '../../../stores/patientDataStore';
import { useToastStore } from '../../../stores/toastStore';

interface DirectoryCardProps {
  patient: Patient;
}

const urgencyConfig = {
  routine: { color: 'bg-blue-100 text-blue-800', border: 'border-blue-200', label: 'Routine' },
  attention: { color: 'bg-warning-light text-warning-dark', border: 'border-warning/30', label: 'Attention' },
  urgent: { color: 'bg-danger-light text-danger', border: 'border-danger/30', label: 'Urgent' }
};

export function DirectoryCard({ patient }: DirectoryCardProps) {
  const navigate = useNavigate();
  const openLauncher = useRecordingStore(state => state.openLauncher);
  const { updatePatient, archivePatient } = usePatientDataStore();
  const addToast = useToastStore(state => state.addToast);
  const [menuOpen, setMenuOpen] = useState(false);

  const initials = patient.name.split(' ').slice(0, 2).map(n => n[0].toUpperCase()).join('');
  const tagsToShow = patient.chronicConditions.slice(0, 2);
  const extraTags = Math.max(0, patient.chronicConditions.length - 2);

  const uConfig = urgencyConfig[patient.urgency];

  const handleStartSession = (e: React.MouseEvent) => {
    e.stopPropagation();
    openLauncher(patient.id);
  };

  const handleMenuClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setMenuOpen(!menuOpen);
  };

  const handleToggleUrgency = (e: React.MouseEvent) => {
    e.stopPropagation();
    setMenuOpen(false);
    updatePatient(patient.id, {
      urgency: patient.urgency === 'urgent' ? 'routine' : 'urgent'
    });
  };

  const handleArchiveToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setMenuOpen(false);
    const newArchivedState = !patient.isArchived;
    
    // Check if we are archiving or unarchiving
    if (newArchivedState) {
      if (window.confirm(`Are you sure you want to archive ${patient.name}?`)) {
        archivePatient(patient.id, true);
        addToast({
          message: `${patient.name} has been archived.`,
          type: 'success',
          action: {
            label: 'Undo',
            onClick: () => archivePatient(patient.id, false)
          }
        });
      }
    } else {
      archivePatient(patient.id, false);
      addToast({
        message: `${patient.name} restored from archive.`,
        type: 'info'
      });
    }
  };

  const navigateToProfile = () => {
    navigate(`/patients/${patient.id}`);
  };

  return (
    <div 
      onClick={navigateToProfile}
      className={cn(
        "group relative flex flex-col bg-surface-card rounded-2xl border transition-all cursor-pointer hover:shadow-md h-full overflow-hidden",
        patient.isArchived ? "opacity-75 border-border-default bg-slate-50" : "border-border-subtle hover:border-brand-primary/30"
      )}
    >
      {/* Top Urgency Strip */}
      <div className={cn("h-1.5 w-full", patient.isArchived ? "bg-slate-300" : (patient.urgency === 'urgent' ? "bg-danger" : patient.urgency === 'attention' ? "bg-warning" : "bg-brand-primary"))} />

      <div className="p-5 flex flex-col h-full relative">
        <div className="flex justify-between items-start mb-3">
          <div className="flex items-center gap-3">
            {patient.photo ? (
              <img src={patient.photo} alt={patient.name} className="w-12 h-12 rounded-full object-cover border border-border-subtle" />
            ) : (
              <div className="w-12 h-12 rounded-full bg-brand-primary-xlt text-brand-primary flex items-center justify-center font-bold text-lg border border-brand-primary/20 shrink-0">
                {initials}
              </div>
            )}
            <div>
              <h3 className="font-bold text-text-primary text-base line-clamp-1 group-hover:text-brand-primary transition-colors">
                {patient.name}
              </h3>
              <p className="text-xs text-text-secondary mt-0.5 font-medium flex items-center gap-1.5">
                <span>{patient.patientId}</span>
                <span className="w-1 h-1 rounded-full bg-border-subtle" />
                <span>{patient.age}y {patient.gender[0]}</span>
              </p>
            </div>
          </div>

          <div className="relative">
            <button 
              onClick={handleMenuClick}
              className="p-1.5 -mr-1.5 text-text-tertiary hover:text-brand-primary hover:bg-brand-primary-light rounded-md transition-colors"
            >
              <MoreVertical size={18} />
            </button>

            {menuOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={(e) => { e.stopPropagation(); setMenuOpen(false); }} />
                <div className="absolute right-0 top-8 w-44 bg-surface-card rounded-lg shadow-xl border border-border-subtle py-1.5 z-20 animate-scale-in">
                  <button onClick={handleStartSession} className="w-full px-4 py-2 text-left text-sm font-semibold text-text-primary hover:bg-surface-ground flex items-center gap-2">
                    <Mic size={16} className="text-brand-primary" /> Start Session
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); navigateToProfile(); }} className="w-full px-4 py-2 text-left text-sm text-text-primary hover:bg-surface-ground flex items-center gap-2">
                    <UserIcon size={16} className="text-text-secondary" /> View Profile
                  </button>
                  <button onClick={handleToggleUrgency} className="w-full px-4 py-2 text-left text-sm text-text-primary hover:bg-surface-ground flex items-center gap-2">
                    {patient.urgency === 'urgent' ? (
                      <><CheckCircle2 size={16} className="text-success" /> Mark Routine</>
                    ) : (
                      <><AlertCircle size={16} className="text-danger" /> Mark Urgent</>
                    )}
                  </button>
                  <div className="h-px bg-border-subtle my-1.5" />
                  <button onClick={handleArchiveToggle} className="w-full px-4 py-2 text-left text-sm font-semibold text-danger hover:bg-danger-light flex items-center gap-2">
                    <Archive size={16} /> {patient.isArchived ? 'Restore Patient' : 'Archive Patient'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Labels/Badges */}
        <div className="flex flex-wrap gap-2 mb-4">
          {!patient.isArchived && (
            <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase border", uConfig.color, uConfig.border)}>
              {uConfig.label}
            </span>
          )}
          {patient.isArchived && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase border bg-slate-100 text-slate-600 border-slate-200">
              Archived
            </span>
          )}
          
          {tagsToShow.map(tag => (
            <span key={tag.id} className="px-2 py-0.5 rounded bg-brand-primary-light text-brand-primary text-xs font-semibold border border-brand-primary/20">
              {tag.label}
            </span>
          ))}
          {extraTags > 0 && (
            <span className="px-2 py-0.5 rounded bg-surface-ground text-text-tertiary text-xs font-semibold border border-border-subtle">
              +{extraTags} more
            </span>
          )}
        </div>

        {/* Latest note snippet */}
        <p className="text-sm text-text-secondary line-clamp-2 mb-5 flex-1 relative group-hover:text-text-primary transition-colors">
          {patient.livingSummary || "No clinical summary available yet. Start a session to document history."}
        </p>

        {/* Footer Area */}
        <div className="flex items-center justify-between pt-4 border-t border-border-subtle mt-auto">
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase font-bold text-text-tertiary tracking-wider flex items-center gap-1">
              <Calendar size={12} /> Last Visit
            </span>
            <span className="text-xs font-semibold text-text-secondary flex items-center gap-1.5">
              {patient.lastVisitDate}
              {patient.sessionCount && patient.sessionCount > 0 && (
                <>
                  <span className="w-1 h-1 rounded-full bg-border-subtle" />
                  <span className="text-brand-primary">{patient.sessionCount} sessions</span>
                </>
              )}
            </span>
          </div>
          
          {!patient.isArchived && (
            <button 
              onClick={handleStartSession}
              className="md:hidden group-hover:flex items-center gap-1.5 h-8 px-3 rounded-lg bg-brand-primary text-text-on-brand font-semibold text-xs hover:bg-brand-primary-mid transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-primary-light"
            >
              <Mic size={14} /> Record
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
