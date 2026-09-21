import React, { useRef, useEffect } from 'react';
import type { Session } from '../../types';
import { Plus } from 'lucide-react';
import { cn } from '../../utils/cn';

interface TimelineStripProps {
  sessions: Session[];
  selectedSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
}

export function TimelineStrip({ sessions, selectedSessionId, onSelectSession, onNewSession }: TimelineStripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to selected session on mount
  useEffect(() => {
    if (scrollRef.current) {
      const selectedNode = scrollRef.current.querySelector('[data-selected="true"]');
      if (selectedNode) {
        selectedNode.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }
  }, [selectedSessionId]);

  // Sort sessions chronologically (oldest first) so timeline flows left-to-right
  // Reversing for display assuming data is usually newest-first
  const sortedSessions = [...sessions].reverse();

  return (
    <div className="w-full h-[108px] bg-brand-primary-xlt/50 border-b border-border-subtle overflow-x-auto custom-scrollbar flex items-center px-8 relative">
      <div className="flex items-center min-w-max relative py-4" ref={scrollRef}>
        
        {/* Continuous background line */}
        <div className="absolute top-1/2 left-4 right-12 h-[2px] bg-border-default -translate-y-[14px] z-0" />

        {sortedSessions.map((session, index) => {
          const isSelected = session.id === selectedSessionId;
          
          return (
            <div 
              key={session.id} 
              className="flex flex-col items-center relative z-10 mx-6 cursor-pointer group"
              onClick={() => onSelectSession(session.id)}
              data-selected={isSelected}
            >
              {/* Node */}
              <div 
                className={cn(
                  "h-[36px] w-[36px] rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300",
                  isSelected 
                    ? "bg-brand-primary text-text-on-brand ring-4 ring-brand-primary-light scale-110 shadow-md border border-brand-primary" 
                    : "bg-surface-card text-brand-primary border-2 border-brand-primary-light group-hover:border-brand-primary"
                )}
              >
                {session.sessionNumber}
              </div>
              
              {/* Labels */}
              <div className="flex flex-col items-center mt-3 text-center">
                <span className={cn(
                  "text-[11px] uppercase tracking-wider font-bold",
                  isSelected ? "text-brand-primary" : "text-text-secondary"
                )}>
                  {session.shortDate}
                </span>
                <span className="text-[10px] text-text-tertiary font-medium">
                  {session.type}
                </span>
              </div>
            </div>
          );
        })}

        {/* New Session Button */}
        <div 
          className="flex flex-col items-center relative z-10 ml-8 cursor-pointer group"
          onClick={onNewSession}
        >
          <div className="h-[36px] w-[36px] rounded-full bg-surface-card border-2 border-dashed border-text-tertiary text-text-tertiary flex items-center justify-center transition-all group-hover:border-brand-primary group-hover:text-brand-primary group-hover:bg-brand-primary-xlt">
            <Plus size={18} strokeWidth={2.5} />
          </div>
          <div className="flex flex-col items-center mt-3 text-center opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-[11px] uppercase tracking-wider font-bold text-brand-primary">
              New
            </span>
          </div>
        </div>

      </div>
    </div>
  );
}
