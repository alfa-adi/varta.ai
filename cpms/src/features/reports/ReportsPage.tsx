import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AppShell } from '../../components/layout/AppShell';
import { InboxPane } from './components/InboxPane';
import { DetailPane } from './components/DetailPane';
import { useReportsStore } from '../../stores/reportsStore';
import { Upload } from 'lucide-react';
import { cn } from '../../utils/cn';

export function ReportsPage() {
  const reports = useReportsStore(state => state.reports);
  const addMockReport = useReportsStore(state => state.addMockReport);
  
  const { reportId } = useParams<{ reportId: string }>();
  const navigate = useNavigate();
  
  // Default to first report on desktop if none selected, but leave empty on mobile
  // We handle this via CSS hidden classes below, but keep state logic simple here
  const selectedReport = reportId ? reports.find(r => r.id === reportId) : null;

  const needsReviewCount = reports.filter(r => r.workflowStatus === 'needs_review').length;
  const receivedThisWeek = reports.filter(r => {
    const d = new Date(r.receivedAt);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    return diff < 7 * 24 * 60 * 60 * 1000;
  }).length;

  return (
    <AppShell title="Lab Reports">
      <div className="flex flex-col h-full overflow-hidden animate-fade-in -mx-6 -my-6 bg-surface-card rounded-t-xl sm:rounded-none">
        
        {/* Page Header (Internal to Reports to match the custom top bar space) */}
        <div className="bg-surface-card border-b border-border-subtle p-4 px-6 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-xl font-bold text-text-primary">Lab Reports</h1>
            <p className="text-sm font-semibold text-text-secondary mt-0.5">
              {needsReviewCount} need review · {receivedThisWeek} received this week
            </p>
          </div>
          
          <button 
            onClick={addMockReport}
            className="flex items-center gap-2 px-4 py-2 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-lg text-sm font-bold shadow-sm transition-colors"
          >
            <Upload size={16} />
            Upload Report (Mock)
          </button>
        </div>

        {/* Split View */}
        <div className="flex-1 flex overflow-hidden">
          {/* Inbox Pane (Left) - Hidden on mobile if report is selected */}
          <div className={cn(
            "w-full md:w-[380px] lg:w-[420px] shrink-0 h-full",
            reportId ? "hidden md:block" : "block"
          )}>
            <InboxPane 
              selectedId={reportId || (reports[0]?.id || null)} 
              onSelect={(id) => navigate(`/reports/${id}`)} 
            />
          </div>
          
          {/* Detail Pane (Right) - Hidden on mobile if no report selected */}
          <div className={cn(
            "flex-1 border-l border-border-subtle bg-surface-ground h-full relative",
            reportId ? "flex" : "hidden md:flex"
          )}>
            <DetailPane report={selectedReport || reports[0] || null} onBack={() => navigate('/reports')} />
          </div>
        </div>
      </div>
    </AppShell>
  );
}
