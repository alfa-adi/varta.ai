import React from 'react';
import type { Patient } from '../../types';
import { TagPill } from '../ui/TagPill';
import { Calendar, Phone, Activity, Clock, Mic } from 'lucide-react';
import { cn } from '../../utils/cn';
import { useRecordingStore } from '../../stores/recordingStore';

interface PatientCardProps {
  patient: Patient;
  onClick?: () => void;
}

export function PatientCard({ patient, onClick }: PatientCardProps) {
  const openLauncher = useRecordingStore(state => state.openLauncher);

  const urgencyColors = {
    routine: 'bg-brand-primary-light text-brand-primary',
    attention: 'bg-warning-light text-warning',
    urgent: 'bg-danger-light text-danger'
  };

  return (
    <div 
      onClick={onClick}
      className="bg-surface-card rounded-xl p-5 shadow-sm border border-border-subtle hover:shadow-md transition-all cursor-pointer group flex flex-col gap-4"
    >
      <div className="flex justify-between items-start">
        <div className="flex gap-4 items-center">
          <div className="h-12 w-12 rounded-full bg-brand-primary-xlt border border-brand-primary-light flex items-center justify-center text-brand-primary font-bold text-lg">
            {patient.name.charAt(0)}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary group-hover:text-brand-primary transition-colors">
              {patient.name}
            </h3>
            <div className="flex items-center gap-2 text-sm text-text-secondary mt-0.5">
              <span>{patient.patientId}</span>
              <span className="h-1 w-1 rounded-full bg-border-strong"></span>
              <span>{patient.age}y {patient.gender.charAt(0)}</span>
            </div>
          </div>
        </div>
        
        <span className={cn(
          "px-2.5 py-1 rounded-full text-xs font-semibold capitalize",
          urgencyColors[patient.urgency]
        )}>
          {patient.urgency}
        </span>
      </div>

      <p className="text-sm text-text-secondary line-clamp-2 leading-relaxed">
        {patient.livingSummary}
      </p>

      {(patient.allergies.length > 0 || patient.chronicConditions.length > 0) && (
        <div className="flex flex-wrap gap-2 pt-2 border-t border-border-subtle/50">
          {patient.allergies.map(allergy => (
            <TagPill key={allergy.id} variant={allergy.variant}>
              {allergy.label}
            </TagPill>
          ))}
          {patient.chronicConditions.map(condition => (
            <TagPill key={condition.id} variant={condition.variant}>
              {condition.label}
            </TagPill>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-text-tertiary font-medium pt-2">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <Phone size={14} className="text-text-tertiary" />
            {patient.phone}
          </span>
          <span className="flex items-center gap-1">
            <Activity size={14} className="text-text-tertiary" />
            {patient.sessionCount || 0} visits
          </span>
        </div>
        <span className="flex items-center gap-1">
          <Clock size={14} />
          Last: {patient.lastVisitDate}
        </span>
        <button 
          onClick={(e) => { e.stopPropagation(); openLauncher(patient.id); }}
          className="h-8 w-8 ml-2 rounded-full bg-brand-primary-xlt text-brand-primary flex items-center justify-center hover:bg-brand-primary-light transition-colors group"
        >
          <Mic size={16} className="group-hover:scale-110 transition-transform" />
        </button>
      </div>
    </div>
  );
}
