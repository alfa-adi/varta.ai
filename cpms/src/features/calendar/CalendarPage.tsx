import React, { useState } from 'react';
import type { Appointment } from '../../types';
import { useAppointmentsStore } from '../../stores/appointmentsStore';
import { DayView } from './DayView';
import { Sidebar } from '../../components/layout/Sidebar';
import { BottomNav } from '../../components/layout/BottomNav';
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon } from 'lucide-react';
import { cn } from '../../utils/cn';

export function CalendarPage() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  
  const appointments = useAppointmentsStore(state => state.appointments);
  const isLoading = false;

  const nextDay = () => {
    const next = new Date(currentDate);
    next.setDate(currentDate.getDate() + 1);
    setCurrentDate(next);
  };

  const prevDay = () => {
    const prev = new Date(currentDate);
    prev.setDate(currentDate.getDate() - 1);
    setCurrentDate(prev);
  };

  const getDayFormatted = (d: Date) => {
    return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  };

  // Filter for the current selected date
  const filteredAppointments = appointments?.filter(a => {
    // Format currentDate as YYYY-MM-DD local to match mock data strings safely
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, '0');
    const day = String(currentDate.getDate()).padStart(2, '0');
    const currentDateStr = `${year}-${month}-${day}`;
    
    return a.date === currentDateStr;
  }) || [];

  return (
    <div className="flex min-h-screen bg-surface-ground">
      <Sidebar isExpanded={isSidebarOpen} onToggle={() => setSidebarOpen(!isSidebarOpen)} />
      
      <div className={cn(
        "flex-1 flex flex-col transition-all duration-300 min-w-0 h-screen overflow-hidden",
        isSidebarOpen ? "md:ml-[240px]" : "md:ml-[72px]",
        "mb-16 md:mb-0" // Add bottom margin for BottomNav on mobile
      )}>
        
        {/* Header */}
        <header className="bg-surface-card border-b border-border-subtle p-4 md:p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 shadow-sm z-10 shrink-0">
          <div className="flex flex-col md:flex-row md:items-center gap-4 md:gap-6 w-full md:w-auto">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 md:h-12 md:w-12 rounded-xl bg-brand-primary-light flex items-center justify-center text-brand-primary">
                <CalendarIcon size={24} />
              </div>
              <div>
                <h1 className="text-xl md:text-2xl font-bold text-text-primary">Schedule</h1>
                <p className="text-sm text-text-secondary font-medium mt-0.5">{filteredAppointments.length} appointments</p>
              </div>
            </div>
            
            <div className="h-px w-full md:h-8 md:w-px bg-border-subtle my-2 md:mx-2 md:my-0" />

            <div className="flex items-center justify-between w-full md:w-auto gap-2 md:gap-4">
              <div className="flex items-center gap-2">
                <button onClick={prevDay} className="p-2 hover:bg-surface-ground rounded-lg transition-colors text-text-secondary hover:text-text-primary min-h-[44px] min-w-[44px] flex items-center justify-center">
                  <ChevronLeft size={20} />
                </button>
                <h2 className="text-base md:text-lg font-bold text-text-primary min-w-[140px] md:min-w-[240px] text-center">
                  {getDayFormatted(currentDate)}
                </h2>
                <button onClick={nextDay} className="p-2 hover:bg-surface-ground rounded-lg transition-colors text-text-secondary hover:text-text-primary min-h-[44px] min-w-[44px] flex items-center justify-center">
                  <ChevronRight size={20} />
                </button>
              </div>
              
              <button 
                onClick={() => setCurrentDate(new Date())}
                className="px-4 py-2 md:py-1.5 border border-border-subtle rounded-lg text-sm font-semibold hover:bg-surface-ground transition-colors"
              >
                Today
              </button>
            </div>
          </div>

          <div className="flex bg-surface-ground p-1 rounded-lg border border-border-subtle w-full md:w-auto mt-2 md:mt-0 justify-center">
            <button className="flex-1 md:flex-none px-5 py-2 md:py-1.5 rounded-md text-sm font-bold bg-surface-card text-brand-primary shadow-sm">
              Day
            </button>
            <button className="flex-1 md:flex-none px-5 py-2 md:py-1.5 rounded-md text-sm font-bold text-text-secondary hover:text-text-primary transition-colors cursor-not-allowed opacity-50" title="Coming soon">
              Week
            </button>
          </div>
        </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden relative">
        {isLoading ? (
          <div className="h-full flex items-center justify-center text-text-tertiary font-medium">Loading schedule...</div>
        ) : (
          <DayView appointments={filteredAppointments} currentDate={currentDate} />
        )}
        </main>
      </div>
      <BottomNav />
    </div>
  );
}
