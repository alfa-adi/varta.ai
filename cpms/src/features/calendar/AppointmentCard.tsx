import React, { useState, useRef, useEffect } from 'react';
import type { Appointment } from '../../types';
import { cn } from '../../utils/cn';
import { Clock, Play, User as UserIcon, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AppointmentCardProps {
  appointment: Appointment;
}

export function AppointmentCard({ appointment }: AppointmentCardProps) {
  const [showPopover, setShowPopover] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const getUrgencyColor = (urgency: string) => {
    switch(urgency) {
      case 'urgent': return 'bg-danger text-danger border-danger/20';
      case 'attention': return 'bg-warning text-warning-dark border-warning/30';
      default: return 'bg-brand-primary text-brand-primary border-brand-primary/20';
    }
  };

  const colors = getUrgencyColor(appointment.urgency);
  const stripColor = colors.split(' ')[0];

  // Close popover when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setShowPopover(false);
      }
    }
    if (showPopover) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showPopover]);

  const handleStartSession = (e: React.MouseEvent) => {
    e.stopPropagation();
    // In a real flow, this might open the launcher or navigate to a recording pre-flight
    navigate(`/patients/${appointment.patientId}`);
  };

  const handleViewProfile = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/patients/${appointment.patientId}`);
  };

  const getEndTime = (startTime: string, durationMinutes: number) => {
    const [time, period] = startTime.split(' ');
    let [h, m] = time.split(':').map(Number);
    if (period === 'PM' && h !== 12) h += 12;
    if (period === 'AM' && h === 12) h = 0;
    
    const date = new Date();
    date.setHours(h, m + durationMinutes, 0, 0);
    
    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  };

  const timeRange = `${appointment.time} - ${getEndTime(appointment.time, appointment.duration)}`;

  return (
    <div className="relative h-full w-full">
      <button 
        onClick={() => setShowPopover(true)}
        className={cn(
          "w-full h-full text-left bg-surface-card rounded-md shadow-sm border border-border-strong flex overflow-hidden hover:shadow-md hover:border-brand-primary transition-all group",
          showPopover ? "ring-2 ring-brand-primary border-brand-primary bg-brand-primary-xlt shadow-md z-20 relative" : ""
        )}
      >
        {/* Urgency Strip */}
        <div className={cn("w-2 h-full shrink-0", stripColor)} />
        
        <div className="px-3 py-1.5 flex-1 flex flex-col justify-center gap-1">
          <div className="flex justify-between items-center">
            <h4 className="font-bold text-[14px] text-text-primary group-hover:text-brand-primary transition-colors truncate pr-2">
              {appointment.patientName}
            </h4>
            <span className="shrink-0 text-[11px] font-bold text-text-tertiary bg-surface-ground px-1.5 py-0.5 rounded uppercase border border-border-subtle">
              {appointment.type}
            </span>
          </div>
          
          <div className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
            <Clock size={14} className="text-text-tertiary" />
            {timeRange}
          </div>
        </div>
      </button>

      {/* Popover / Bottom Sheet */}
      {showPopover && (
        <>
          {/* Mobile Backdrop */}
          <div className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 md:hidden animate-fade-in" onClick={() => setShowPopover(false)} />
          
          <div 
            ref={popoverRef}
            className="fixed bottom-0 left-0 right-0 md:absolute md:top-1/2 md:bottom-auto md:left-auto md:right-auto md:-translate-y-1/2 w-full md:w-[280px] bg-surface-card rounded-t-2xl md:rounded-xl shadow-2xl md:shadow-xl border-t md:border border-border-subtle p-5 z-50 animate-slide-up md:animate-fade-in pb-safe md:pb-5"
            style={{ 
              ...(typeof window !== 'undefined' && window.innerWidth >= 768 ? { left: 'calc(100% + 8px)' } : {})
            }}
          >
          <div className="flex justify-between items-start mb-4 border-b border-border-subtle pb-4">
            <div>
              <h3 className="font-bold text-text-primary text-base">{appointment.patientName}</h3>
              <p className="text-sm font-medium text-text-secondary mt-1">{appointment.type} • {timeRange}</p>
            </div>
            <button onClick={() => setShowPopover(false)} className="text-text-tertiary hover:text-text-primary p-1">
              <X size={16} />
            </button>
          </div>

          <div className="flex flex-col gap-2">
            <button 
              onClick={handleStartSession}
              className="w-full flex items-center justify-center gap-2 bg-brand-primary text-text-on-brand font-bold py-2.5 rounded-lg hover:bg-brand-primary-mid transition-colors shadow-sm"
            >
              <Play size={16} className="fill-current" />
              Start Session
            </button>
            <button 
              onClick={handleViewProfile}
              className="w-full flex items-center justify-center gap-2 bg-surface-ground text-text-primary font-bold py-2.5 rounded-lg border border-border-subtle hover:bg-border-subtle/30 transition-colors"
            >
              <UserIcon size={16} />
              View Profile
            </button>
          </div>
          
          {/* Popover Arrow */}
          <div className="hidden md:block absolute top-1/2 -left-2 -translate-y-1/2 w-4 h-4 bg-surface-card border-l border-b border-border-subtle transform rotate-45" />
        </div>
        </>
      )}
    </div>
  );
}
