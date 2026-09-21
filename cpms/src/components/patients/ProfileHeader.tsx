import React from 'react';
import type { Patient } from '../../types';
import { TagPill } from '../ui/TagPill';
import { Sparkles, FileText, Share2, Edit3 } from 'lucide-react';

interface ProfileHeaderProps {
  patient: Patient;
}

export function ProfileHeader({ patient }: ProfileHeaderProps) {
  return (
    <div className="sticky top-0 z-20 bg-surface-card border-b border-border-subtle flex flex-col md:flex-row shadow-sm min-h-[168px]">
      
      {/* Left side (60%) */}
      <div className="flex-1 py-4 px-6 flex flex-col justify-center border-b md:border-b-0 md:border-r border-border-subtle">
        <div className="flex items-start gap-5">
          <div className="h-[72px] w-[72px] rounded-full bg-brand-primary text-text-on-brand flex items-center justify-center text-3xl font-bold shadow-md shrink-0">
            {patient.name.charAt(0)}
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-text-primary leading-tight">
              {patient.name}
            </h1>
            <div className="flex flex-wrap items-center gap-2 mt-1 text-text-secondary text-sm font-medium">
              <span>{patient.age} yrs</span>
              <span>&middot;</span>
              <span>{patient.gender}</span>
              <span>&middot;</span>
              <span>{patient.bloodGroup || 'O+'}</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-1.5 text-xs text-text-tertiary">
              <span className="font-mono bg-surface-ground px-1.5 py-0.5 rounded border border-border-subtle">
                {patient.patientId}
              </span>
              {patient.abhaId && (
                <span className="font-mono bg-surface-ground px-1.5 py-0.5 rounded border border-border-subtle">
                  {patient.abhaId}
                </span>
              )}
            </div>

            {/* Always-visible alert tags */}
            {(patient.allergies.length > 0 || patient.chronicConditions.length > 0) && (
              <div className="flex flex-wrap gap-2 mt-3">
                {patient.allergies.map(allergy => (
                  <TagPill key={allergy.id} variant={allergy.variant}>{allergy.label}</TagPill>
                ))}
                {patient.chronicConditions.map(chronic => (
                  <TagPill key={chronic.id} variant={chronic.variant}>{chronic.label}</TagPill>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right side (40%) */}
      <div className="w-full md:w-[40%] py-4 px-6 flex flex-col justify-between bg-surface-ground">
        <div className="bg-brand-primary-xlt border border-brand-primary/20 rounded-lg p-3 shadow-sm h-full flex flex-col">
          <div className="flex items-center gap-1.5 mb-1 text-brand-primary font-semibold text-xs uppercase tracking-wider">
            <Sparkles size={14} className="fill-brand-primary/20" />
            AI Health Snapshot
          </div>
          <p className="text-sm text-text-secondary leading-relaxed line-clamp-3">
            {patient.livingSummary}
          </p>
          <button className="text-xs text-brand-primary font-semibold hover:underline mt-auto self-start pt-1">
            Expand Summary
          </button>
        </div>

        <div className="flex gap-2 mt-4 justify-end">
          <ActionButton icon={<FileText size={18} />} label="Prescriptions" />
          <ActionButton icon={<Share2 size={18} />} label="Share" />
          <ActionButton icon={<Edit3 size={18} />} label="Edit" />
        </div>
      </div>
      
    </div>
  );
}

function ActionButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="flex items-center justify-center p-2.5 rounded-lg bg-surface-card border border-border-subtle text-text-secondary hover:text-brand-primary hover:border-brand-primary hover:bg-brand-primary-xlt transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-primary-light">
      {icon}
      <span className="sr-only">{label}</span>
    </button>
  );
}
