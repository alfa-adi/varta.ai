import React, { useState } from 'react';
import { X, Calendar as CalendarIcon, Clock, AlertTriangle } from 'lucide-react';
import { useAppointmentsStore } from '../../../stores/appointmentsStore';
import type { LabReport } from '../../../types';

interface FollowUpModalProps {
  report: LabReport;
  onClose: () => void;
}

export function FollowUpModal({ report, onClose }: FollowUpModalProps) {
  const addAppointment = useAppointmentsStore(state => state.addAppointment);
  
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [time, setTime] = useState('14:00');
  const [urgency, setUrgency] = useState<'routine' | 'attention' | 'urgent'>(report.isAbnormal ? 'attention' : 'routine');
  const [notes, setNotes] = useState(`Review ${report.reportType} findings.`);

  const handleSave = () => {
    if (!report.patientId) return;

    addAppointment({
      id: `a-fu-${Date.now()}`,
      patientId: report.patientId,
      patientName: report.patientName,
      date,
      time,
      duration: 15,
      type: 'Follow-up',
      urgency,
      sourceReportId: report.id,
      sourceReportContext: notes
    });
    
    // In a real app we might also show a toast here
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-surface-card w-full max-w-md rounded-2xl shadow-2xl flex flex-col animate-scale-in border border-border-subtle">
        <div className="flex items-center justify-between p-5 border-b border-border-subtle">
          <div>
            <h3 className="text-lg font-bold text-text-primary">Schedule Follow-up</h3>
            <p className="text-sm text-text-secondary">For {report.patientName}</p>
          </div>
          <button onClick={onClose} className="p-2 text-text-tertiary hover:bg-surface-ground rounded-full transition-colors">
            <X size={20} />
          </button>
        </div>

        <div className="p-5 space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-text-secondary uppercase tracking-wider">Date</label>
              <div className="relative">
                <CalendarIcon size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
                <input 
                  type="date" 
                  value={date}
                  onChange={e => setDate(e.target.value)}
                  className="w-full h-10 pl-9 pr-3 rounded-lg border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary"
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-text-secondary uppercase tracking-wider">Time</label>
              <div className="relative">
                <Clock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
                <input 
                  type="time" 
                  value={time}
                  onChange={e => setTime(e.target.value)}
                  className="w-full h-10 pl-9 pr-3 rounded-lg border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary"
                />
              </div>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-text-secondary uppercase tracking-wider">Urgency</label>
            <div className="flex gap-2">
              {(['routine', 'attention', 'urgent'] as const).map(u => (
                <button
                  key={u}
                  onClick={() => setUrgency(u)}
                  className={`flex-1 py-2 text-sm font-semibold rounded-lg border transition-all ${
                    urgency === u 
                      ? u === 'routine' ? 'bg-brand-primary-light border-brand-primary text-brand-primary'
                        : u === 'attention' ? 'bg-warning/20 border-warning text-warning'
                        : 'bg-danger-light border-danger text-danger'
                      : 'bg-surface-ground border-border-subtle text-text-secondary hover:bg-surface-card'
                  }`}
                >
                  {u.charAt(0).toUpperCase() + u.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-text-secondary uppercase tracking-wider">Reason / Notes</label>
            <textarea 
              value={notes}
              onChange={e => setNotes(e.target.value)}
              className="w-full h-24 p-3 rounded-lg border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary resize-none"
            />
          </div>
          
          {report.isAbnormal && (
            <div className="flex items-start gap-2 p-3 bg-warning/10 border border-warning/20 rounded-lg text-sm text-warning-dark">
              <AlertTriangle size={16} className="shrink-0 mt-0.5" />
              <p>This report contains abnormal findings. Ensure the patient is advised to bring hard copies if needed.</p>
            </div>
          )}
        </div>

        <div className="p-5 border-t border-border-subtle bg-surface-ground/50 flex justify-end gap-3 rounded-b-2xl">
          <button onClick={onClose} className="px-5 py-2.5 text-sm font-bold text-text-secondary hover:bg-surface-card rounded-xl transition-colors">
            Cancel
          </button>
          <button onClick={handleSave} className="px-6 py-2.5 text-sm font-bold text-white bg-brand-primary hover:bg-brand-primary-mid rounded-xl shadow-sm transition-colors">
            Schedule Appointment
          </button>
        </div>
      </div>
    </div>
  );
}
