import React, { useState } from 'react';
import type { LabReport } from '../../../types';
import { useReportsStore } from '../../../stores/reportsStore';
import { useSettingsStore } from '../../../stores/settingsStore';
import { PatientMatchControls } from './PatientMatchControls';
import { FollowUpModal } from './FollowUpModal';
import { CheckCircle2, Clock, Calendar, Download, Printer, UserCircle2, BrainCircuit, ShieldAlert, ArrowDown, ArrowUp, ArrowLeft } from 'lucide-react';
import { cn } from '../../../utils/cn';

interface DetailPaneProps {
  report: LabReport | null;
  onBack?: () => void;
}

export function DetailPane({ report, onBack }: DetailPaneProps) {
  const { markAsReviewed, addToPatientRecord } = useReportsStore();
  const [showFollowUp, setShowFollowUp] = useState(false);

  if (!report) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-surface-ground text-text-tertiary">
        <p className="font-medium text-sm">Select a report to view details</p>
      </div>
    );
  }

  const isMatched = report.patientId !== null;
  const isReviewed = report.workflowStatus === 'reviewed';

  return (
    <div className="flex-1 flex flex-col h-full bg-surface-ground overflow-y-auto">
      {/* Action Bar */}
      <div className="bg-surface-card border-b border-border-subtle p-4 flex items-center justify-between sticky top-0 z-10 shadow-sm overflow-x-auto">
        <div className="flex items-center gap-2 md:gap-3 shrink-0">
          {onBack && (
            <button 
              onClick={onBack}
              className="md:hidden p-2 mr-1 text-text-secondary hover:bg-surface-ground hover:text-text-primary rounded-lg transition-colors flex items-center justify-center min-h-[44px] min-w-[44px]"
              aria-label="Back to reports list"
            >
              <ArrowLeft size={20} />
            </button>
          )}
          <button className="p-2 text-text-secondary hover:bg-surface-ground hover:text-text-primary rounded-lg transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center" title="Download">
            <Download size={18} />
          </button>
          <button className="p-2 text-text-secondary hover:bg-surface-ground hover:text-text-primary rounded-lg transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center" title="Print">
            <Printer size={18} />
          </button>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            disabled={!isMatched}
            onClick={() => setShowFollowUp(true)}
            className="flex items-center gap-2 px-4 py-2 bg-surface-ground border border-border-subtle hover:bg-surface-card text-text-primary rounded-lg text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Calendar size={16} />
            Create Follow-up
          </button>
          
          <button 
            disabled={!isMatched || report.addedToRecord}
            onClick={() => addToPatientRecord(report.id)}
            className="flex items-center gap-2 px-4 py-2 bg-surface-ground border border-border-subtle hover:bg-surface-card text-text-primary rounded-lg text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <UserCircle2 size={16} />
            {report.addedToRecord ? 'Added to Patient Record' : 'Add to Patient Record'}
          </button>
          
          <button
            disabled={!isMatched || isReviewed}
            onClick={() => markAsReviewed(report.id)}
            className="flex items-center gap-2 px-5 py-2 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-lg text-sm font-bold shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isReviewed ? <CheckCircle2 size={18} /> : <Clock size={18} />}
            {isReviewed ? 'Reviewed' : 'Mark as Reviewed'}
          </button>
        </div>
      </div>

      <div className="p-6 max-w-4xl mx-auto w-full space-y-6">
        
        {/* Unmatched State */}
        {!isMatched && <PatientMatchControls reportId={report.id} />}

        {/* Header Details */}
        <div className="bg-surface-card border border-border-subtle rounded-xl p-6 shadow-sm flex flex-wrap justify-between gap-6">
          <div>
            <h2 className={cn("text-2xl font-bold mb-1", !isMatched ? 'text-text-secondary italic' : 'text-text-primary')}>
              {report.patientName}
            </h2>
            <div className="flex items-center gap-4 text-sm font-medium text-text-secondary">
              <span>{isMatched ? 'PID: PT-XXXX • 45y M' : 'Unmatched'}</span>
            </div>
          </div>
          
          <div className="text-right">
            <h3 className="text-lg font-bold text-text-primary">{report.reportType}</h3>
            <p className="text-sm font-semibold text-text-secondary mt-0.5">{report.labName}</p>
            <p className="text-xs text-text-tertiary mt-2">
              Received: {new Date(report.receivedAt).toLocaleString()}
            </p>
          </div>
        </div>

        {/* AI Extraction Banner */}
        {report.findings.length > 0 && (
          <div className="bg-brand-primary-xlt border border-brand-primary-light rounded-xl p-4 flex gap-4 shadow-sm items-start">
            <div className="bg-brand-primary text-text-on-brand p-2 rounded-lg shrink-0 mt-0.5">
              <BrainCircuit size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-bold text-brand-primary">Extracted values — clinician verification required.</h4>
              </div>
              <p className="text-sm text-text-secondary">
                Our AI has extracted structured findings from the original document. Please verify these values against the original report before making clinical decisions.
              </p>
            </div>
          </div>
        )}

        {/* Key Findings Card */}
        {report.findings.length > 0 && (
          <div className="bg-surface-card border border-border-subtle rounded-xl shadow-sm overflow-hidden">
            <div className="p-4 border-b border-border-subtle bg-surface-ground/50 flex justify-between items-center">
              <h3 className="font-bold text-text-primary">Key Findings</h3>
              {report.isAbnormal && (
                <span className="flex items-center gap-1 text-xs uppercase font-bold text-danger bg-danger-light px-2.5 py-1 rounded-md">
                  <ShieldAlert size={14} /> Abnormalities Detected
                </span>
              )}
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-text-tertiary uppercase bg-surface-ground">
                  <tr>
                    <th className="px-6 py-3 font-bold tracking-wider">Test Name</th>
                    <th className="px-6 py-3 font-bold tracking-wider">Value</th>
                    <th className="px-6 py-3 font-bold tracking-wider">Unit</th>
                    <th className="px-6 py-3 font-bold tracking-wider">Reference Range</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {report.findings.map(finding => {
                    const isHigh = finding.status === 'high';
                    const isLow = finding.status === 'low';
                    const isAbnormal = isHigh || isLow;
                    
                    return (
                      <tr key={finding.id} className={cn(isAbnormal ? "bg-danger/5" : "")}>
                        <td className="px-6 py-4 font-semibold text-text-primary">{finding.testName}</td>
                        <td className="px-6 py-4">
                          <span className={cn(
                            "inline-flex items-center gap-1.5 font-bold text-base",
                            isAbnormal ? "text-danger" : "text-text-primary"
                          )}>
                            {finding.value}
                            {isHigh && <ArrowUp size={16} className="text-danger" />}
                            {isLow && <ArrowDown size={16} className="text-danger" />}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-text-secondary">{finding.unit}</td>
                        <td className="px-6 py-4 text-text-tertiary">{finding.referenceRange}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Original Document Placeholder */}
        <div className="bg-surface-card border border-border-subtle rounded-xl shadow-sm p-6">
          <h3 className="font-bold text-text-primary mb-4">Original Document</h3>
          <div className="w-full h-[600px] bg-surface-ground border-2 border-dashed border-border-subtle rounded-xl flex flex-col items-center justify-center text-text-tertiary">
            <Printer size={48} className="mb-4 opacity-50" />
            <p className="font-medium text-lg">Report Preview</p>
            <p className="text-sm mt-2 max-w-sm text-center">In a real environment, the scanned lab report or PDF would render here.</p>
          </div>
        </div>

      </div>

      {showFollowUp && (
        <FollowUpModal report={report} onClose={() => setShowFollowUp(false)} />
      )}
    </div>
  );
}
