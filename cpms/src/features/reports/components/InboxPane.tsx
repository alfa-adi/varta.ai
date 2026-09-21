import React, { useMemo, useState } from 'react';
import { useReportsStore } from '../../../stores/reportsStore';
import type { LabReport, ReportWorkflowStatus } from '../../../types';
import { Search, Filter, FlaskConical, AlertCircle, CheckCircle2, UserX, Clock } from 'lucide-react';
import { cn } from '../../../utils/cn';

interface InboxPaneProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

type FilterType = 'All' | 'Needs Review' | 'Abnormal' | 'Reviewed' | 'Unmatched';

export function InboxPane({ selectedId, onSelect }: InboxPaneProps) {
  const reports = useReportsStore(state => state.reports);
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterType>('All');

  const filteredReports = useMemo(() => {
    let result = reports;

    // Filter
    if (activeFilter === 'Needs Review') result = result.filter(r => r.workflowStatus === 'needs_review');
    else if (activeFilter === 'Abnormal') result = result.filter(r => r.isAbnormal);
    else if (activeFilter === 'Reviewed') result = result.filter(r => r.workflowStatus === 'reviewed');
    else if (activeFilter === 'Unmatched') result = result.filter(r => r.workflowStatus === 'unmatched');

    // Search
    if (query.trim()) {
      const q = query.toLowerCase();
      result = result.filter(r => 
        r.patientName.toLowerCase().includes(q) || 
        r.reportType.toLowerCase().includes(q) ||
        r.labName.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)
      );
    }

    // Sort: newest first
    return result.sort((a, b) => new Date(b.receivedAt).getTime() - new Date(a.receivedAt).getTime());
  }, [reports, query, activeFilter]);

  const StatusBadge = ({ report }: { report: LabReport }) => {
    if (report.workflowStatus === 'unmatched') {
      return <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-text-tertiary bg-surface-ground px-2 py-0.5 rounded-full border border-border-subtle"><UserX size={10} /> Unmatched</span>;
    }
    if (report.workflowStatus === 'reviewed') {
      return <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-success bg-success/10 px-2 py-0.5 rounded-full border border-success/20"><CheckCircle2 size={10} /> Reviewed</span>;
    }
    if (report.isAbnormal) {
      return <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-danger bg-danger/10 px-2 py-0.5 rounded-full border border-danger/20"><AlertCircle size={10} /> Abnormal</span>;
    }
    return <span className="flex items-center gap-1 text-[10px] uppercase font-bold text-warning-dark bg-warning/10 px-2 py-0.5 rounded-full border border-warning/20"><Clock size={10} /> Needs Review</span>;
  };

  return (
    <div className="flex flex-col h-full bg-surface-card border-r border-border-subtle">
      <div className="p-4 border-b border-border-subtle shrink-0">
        <div className="relative mb-3">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
          <input 
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search reports..."
            className="w-full h-10 pl-9 pr-3 rounded-lg border border-border-subtle bg-surface-ground text-sm focus:outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary transition-all"
          />
        </div>
        
        <div className="flex gap-2 overflow-x-auto pb-1 hide-scrollbar">
          {(['All', 'Needs Review', 'Abnormal', 'Reviewed', 'Unmatched'] as FilterType[]).map(f => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={cn(
                "px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition-colors border",
                activeFilter === f 
                  ? "bg-brand-primary text-text-on-brand border-brand-primary" 
                  : "bg-surface-ground text-text-secondary border-border-subtle hover:bg-surface-card"
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filteredReports.map(report => (
          <button
            key={report.id}
            onClick={() => onSelect(report.id)}
            className={cn(
              "w-full text-left p-4 border-b border-border-subtle hover:bg-surface-ground transition-colors relative flex items-start gap-3",
              selectedId === report.id ? "bg-brand-primary-xlt border-l-4 border-l-brand-primary" : "border-l-4 border-l-transparent"
            )}
          >
            <div className={cn("h-10 w-10 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-bold text-sm", 
              report.workflowStatus === 'unmatched' ? 'bg-surface-ground text-text-tertiary border border-border-subtle' : 'bg-brand-primary text-text-on-brand'
            )}>
              {report.workflowStatus === 'unmatched' ? '?' : report.patientName.charAt(0)}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2 mb-1">
                <p className={cn("font-bold text-sm truncate", report.workflowStatus === 'unmatched' ? 'text-text-secondary italic' : 'text-text-primary')}>
                  {report.patientName}
                </p>
                <StatusBadge report={report} />
              </div>
              
              <p className="text-xs font-semibold text-text-secondary truncate">{report.reportType}</p>
              
              <div className="flex items-center justify-between gap-2 mt-1.5">
                <p className="text-xs text-text-tertiary truncate flex items-center gap-1"><FlaskConical size={12}/> {report.labName}</p>
                <p className="text-[10px] text-text-tertiary font-medium">
                  {new Date(report.receivedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                </p>
              </div>
            </div>
          </button>
        ))}

        {filteredReports.length === 0 && (
          <div className="p-8 text-center">
            <FlaskConical size={32} className="mx-auto text-border-subtle mb-3" />
            <p className="text-sm font-semibold text-text-secondary">No reports found.</p>
            {(query || activeFilter !== 'All') && (
              <button 
                onClick={() => { setQuery(''); setActiveFilter('All'); }}
                className="text-xs font-bold text-brand-primary hover:underline mt-2"
              >
                Clear filters
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
