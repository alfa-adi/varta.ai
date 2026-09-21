import React, { useState, useEffect } from 'react';
import type { Session } from '../../types';
import { TagPill } from '../ui/TagPill';
import { TranscriptBubbles } from './TranscriptBubbles';
import { ChevronDown, ChevronUp, FileText, Printer, Send, MessageCircle, MoreVertical, Plus, ArrowRight } from 'lucide-react';
import { cn } from '../../utils/cn';
import { useNavigate, useParams } from 'react-router-dom';

interface SessionDetailProps {
  session: Session;
  isLatest: boolean;
}

export function SessionDetail({ session, isLatest }: SessionDetailProps) {
  const navigate = useNavigate();
  const { id: patientId } = useParams<{ id: string }>();

  // Reset state when session changes
  const [summaryExpanded, setSummaryExpanded] = useState(isLatest);
  const [transcriptExpanded, setTranscriptExpanded] = useState(!isLatest);

  useEffect(() => {
    setSummaryExpanded(isLatest);
    setTranscriptExpanded(!isLatest);
  }, [session.id, isLatest]);

  return (
    <div className="flex flex-col gap-4 max-w-4xl mx-auto pb-24">
      
      {/* Session Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-surface-ground p-2 rounded-lg sticky top-16 z-10">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold text-text-primary">
            Session {session.sessionNumber} — {session.date}
          </h2>
          <span className="px-2 py-0.5 rounded text-xs font-semibold bg-border-subtle text-text-secondary border border-border-default">
            {session.duration}
          </span>
          <span className="px-2 py-0.5 rounded text-xs font-semibold bg-brand-primary-light text-brand-primary border border-brand-primary-light">
            {session.mode} {session.languagePair ? `(${session.languagePair})` : ''}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => navigate(`/patients/${patientId}/prescription/${session.id}`)}
            className="px-4 py-2 bg-brand-primary text-text-on-brand text-sm font-semibold rounded-lg shadow-sm hover:bg-brand-primary-mid transition-colors flex items-center gap-2"
          >
            <FileText size={16} />
            Generate Prescription
          </button>
          <button className="p-2 text-text-secondary hover:text-brand-primary hover:bg-brand-primary-light rounded-lg transition-colors">
            <MoreVertical size={20} />
          </button>
        </div>
      </div>

      {/* Section A - Summary */}
      <CollapsibleCard 
        title="Session Summary" 
        expanded={summaryExpanded} 
        onToggle={() => setSummaryExpanded(!summaryExpanded)}
      >
        <div className="space-y-4 p-4">
          <div>
            <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-1">Chief Complaint</h4>
            <p className="text-text-primary font-medium">{session.chiefComplaint}</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Symptoms Detected</h4>
              <div className="flex flex-wrap gap-2">
                {session.symptoms.length ? session.symptoms.map(s => (
                  <TagPill key={s.id} variant={s.variant}>{s.label}</TagPill>
                )) : <span className="text-sm text-text-tertiary">None recorded</span>}
              </div>
            </div>
            <div>
              <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Tests Ordered</h4>
              <div className="flex flex-wrap gap-2">
                {session.tests.length ? session.tests.map(t => (
                  <TagPill key={t.id} variant={t.variant}>{t.label}</TagPill>
                )) : <span className="text-sm text-text-tertiary">None recorded</span>}
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Vitals Recorded</h4>
            <div className="flex flex-wrap items-center gap-3">
              {session.vitals.length ? session.vitals.map((v, i) => (
                <div key={i} className="flex items-center bg-surface-ground px-3 py-1.5 rounded-md border border-border-subtle shadow-sm">
                  <span className="text-xs text-text-secondary mr-2">{v.label}</span>
                  <span className="text-sm font-bold text-brand-primary">{v.value}</span>
                </div>
              )) : <span className="text-sm text-text-tertiary">No vitals recorded</span>}
            </div>
          </div>

          <div className="bg-brand-primary-xlt/50 p-4 rounded-lg border border-border-subtle">
            <h4 className="text-xs font-bold text-brand-primary uppercase tracking-wider mb-2">Doctor's Notes</h4>
            <p className="text-sm text-text-secondary leading-relaxed whitespace-pre-wrap">{session.doctorNotes}</p>
          </div>

          {session.followUpDate && (
            <div className="flex items-center justify-between border-t border-border-subtle pt-4 mt-2">
              <div className="text-sm text-text-secondary">
                Follow-up suggested on <span className="font-bold text-text-primary">{session.followUpDate}</span>
              </div>
              <button className="flex items-center gap-1.5 text-xs font-semibold text-accent-green hover:underline">
                <MessageCircle size={14} />
                Send WhatsApp reminder
              </button>
            </div>
          )}
        </div>
      </CollapsibleCard>

      {/* Section B - Transcript */}
      <CollapsibleCard 
        title="Conversation Transcript" 
        expanded={transcriptExpanded} 
        onToggle={() => setTranscriptExpanded(!transcriptExpanded)}
        noPadding
      >
        <div className="h-[400px]">
          <TranscriptBubbles transcript={session.transcript} mode={session.mode} />
        </div>
      </CollapsibleCard>

      {/* Section C - Reports */}
      <CollapsibleCard title="Attached Reports" expanded={true} onToggle={() => {}}>
        <div className="p-4">
          {session.reports.length ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {session.reports.map(r => (
                <div key={r.id} className="flex items-center justify-between p-3 border border-border-subtle rounded-lg bg-surface-ground">
                  <div>
                    <p className="text-sm font-semibold text-text-primary">{r.title}</p>
                    <p className="text-xs text-text-tertiary">{r.date} • {r.type.toUpperCase()}</p>
                  </div>
                  <span className={cn(
                    "text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded",
                    r.status === 'reviewed' ? "bg-accent-green-light text-accent-green" : "bg-warning-light text-warning"
                  )}>
                    {r.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 border-2 border-dashed border-border-subtle rounded-lg">
              <p className="text-sm text-text-tertiary mb-3">No reports attached to this session.</p>
              <button className="text-brand-primary text-sm font-semibold flex items-center justify-center gap-1.5 mx-auto hover:underline">
                <Plus size={16} /> Add Report
              </button>
            </div>
          )}
        </div>
      </CollapsibleCard>

      {/* Section D - Prescription */}
      {session.prescription && (
        <CollapsibleCard title="Prescription Issued" expanded={true} onToggle={() => {}}>
          <div className="p-4 flex flex-col gap-4">
            <div>
              <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Medicines</h4>
              <div className="flex flex-wrap gap-2">
                {session.prescription.medicines.map(m => (
                  <TagPill key={m.id} variant="medicine">{m.name} ({m.frequency})</TagPill>
                ))}
              </div>
            </div>
            
            {session.prescription.advice && (
              <div>
                <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-1">General Advice</h4>
                <p className="text-sm text-text-secondary">{session.prescription.advice}</p>
              </div>
            )}

            <div className="flex gap-3 pt-3 border-t border-border-subtle">
              <button className="flex items-center gap-1.5 text-sm font-semibold text-brand-primary hover:underline">
                <FileText size={16} /> View Full PDF
              </button>
              <button className="flex items-center gap-1.5 text-sm font-semibold text-brand-primary hover:underline">
                <Printer size={16} /> Print
              </button>
              <button className="flex items-center gap-1.5 text-sm font-semibold text-brand-primary hover:underline">
                <Send size={16} /> Resend
              </button>
            </div>
          </div>
        </CollapsibleCard>
      )}

    </div>
  );
}

function CollapsibleCard({ 
  title, 
  expanded, 
  onToggle, 
  children,
  noPadding = false
}: { 
  title: string; 
  expanded: boolean; 
  onToggle: () => void; 
  children: React.ReactNode;
  noPadding?: boolean;
}) {
  return (
    <div className="bg-surface-card rounded-xl shadow-sm border border-border-subtle overflow-hidden transition-all duration-300">
      <div 
        className={cn(
          "flex items-center justify-between p-4 cursor-pointer hover:bg-brand-primary-xlt/50 transition-colors",
          expanded && "border-b border-border-subtle"
        )}
        onClick={onToggle}
      >
        <h3 className="text-base font-bold text-text-primary">{title}</h3>
        <button className="p-1 text-text-tertiary hover:text-brand-primary transition-colors">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>
      <div 
        className={cn(
          "transition-all duration-300 origin-top",
          expanded ? "block opacity-100" : "hidden opacity-0",
          !noPadding && expanded ? "" : ""
        )}
      >
        {children}
      </div>
    </div>
  );
}
