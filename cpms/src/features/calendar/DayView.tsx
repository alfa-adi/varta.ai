import React, { useState } from 'react';
import type { Appointment } from '../../types';
import { AppointmentCard } from './AppointmentCard';
import { Coffee } from 'lucide-react';

interface DayViewProps {
  appointments: Appointment[];
  currentDate: Date;
}

const START_HOUR = 8;
const END_HOUR = 20; // 8 PM
const HOUR_HEIGHT = 128; // 64px per 30 minutes

export function DayView({ appointments, currentDate }: DayViewProps) {
  const hours = Array.from({ length: END_HOUR - START_HOUR + 1 }, (_, i) => START_HOUR + i);

  // Simple current time indicator logic
  const now = new Date();
  const currentHour = now.getHours();
  const currentMinute = now.getMinutes();
  
  const isToday = currentDate.toDateString() === now.toDateString();
  const showCurrentTime = isToday && currentHour >= START_HOUR && currentHour <= END_HOUR;
  const currentTimeTop = ((currentHour - START_HOUR) + (currentMinute / 60)) * HOUR_HEIGHT;

  const parseTime = (timeStr: string) => {
    // Expects "10:00 AM" format
    const [time, period] = timeStr.split(' ');
    let [h, m] = time.split(':').map(Number);
    if (period === 'PM' && h !== 12) h += 12;
    if (period === 'AM' && h === 12) h = 0;
    return { h, m };
  };

  return (
    <div className="h-full bg-surface-card relative border-l border-border-subtle shadow-inner overflow-hidden">
      {/* Empty State Overlay */}
      {appointments.length === 0 && (
        <div className="absolute inset-0 left-[96px] flex flex-col items-center justify-center pointer-events-none z-30">
          <div className="w-20 h-20 mb-4 bg-surface-ground rounded-full flex items-center justify-center border border-border-subtle shadow-sm">
            <Coffee size={44} className="text-text-tertiary opacity-80" strokeWidth={1.5} />
          </div>
          <h3 className="text-[15px] font-semibold text-text-primary">No appointments scheduled</h3>
          <p className="text-[13px] text-text-secondary mt-1">Enjoy the free time!</p>
        </div>
      )}

      <div className="h-full overflow-y-auto relative">
        <div className="relative min-w-[800px]">
          {/* Time Labels & Grid Lines */}
          {hours.map((hour, index) => {
          const displayHour = hour > 12 ? hour - 12 : hour === 0 ? 12 : hour;
          const ampm = hour >= 12 ? 'PM' : 'AM';
          
          return (
            <div 
              key={hour} 
              className="relative border-b border-border-subtle/30 flex w-full"
              style={{ height: `${HOUR_HEIGHT}px` }}
            >
              {/* Label */}
              <div className="w-[96px] shrink-0 -mt-2 pr-4 text-right">
                <span className="text-xs font-medium text-text-tertiary">
                  {displayHour}:00 {ampm}
                </span>
              </div>
              
              {/* Grid Area */}
              <div className="flex-1 relative border-l border-border-subtle/20">
                {/* 30-min solid subtle line */}
                {index !== hours.length - 1 && (
                  <div className="absolute top-1/2 left-0 right-0 border-b border-border-subtle/10" />
                )}
              </div>
            </div>
          );
        })}

        {/* Current Time Indicator */}
        {showCurrentTime && (
          <div 
            className="absolute right-0 z-20 flex items-center pointer-events-none"
            style={{ top: `${currentTimeTop}px`, left: '96px' }}
          >
            <div className="w-2.5 h-2.5 rounded-full bg-danger -ml-1.5 shadow-sm"></div>
            <div className="h-[2px] bg-danger flex-1 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></div>
          </div>
        )}

        {/* Appointments */}
        {appointments.length > 0 && appointments.map(appointment => {
          const { h, m } = parseTime(appointment.time);
          const topOffset = ((h - START_HOUR) + (m / 60)) * HOUR_HEIGHT;
          const height = (appointment.duration / 60) * HOUR_HEIGHT;
          
          return (
            <div 
              key={appointment.id}
              className="absolute right-8 max-w-[400px] px-2 transition-all hover:z-30 z-10"
              style={{ top: `${topOffset}px`, height: `${height}px`, left: '104px' }}
            >
              <AppointmentCard appointment={appointment} />
            </div>
          );
        })}
      </div>
    </div>
  </div>
  );
}
