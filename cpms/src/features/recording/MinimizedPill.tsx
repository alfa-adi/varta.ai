import React from 'react';
import { useRecordingStore } from '../../stores/recordingStore';
import { useQuery } from '@tanstack/react-query';
import { fetchPatient } from '../../data/mock';
import { Maximize2, Mic } from 'lucide-react';
import { cn } from '../../utils/cn';

export function MinimizedPill() {
  const { isRecording, isMinimized, isPaused, elapsedSeconds, currentPatientId, maximizeRecording } = useRecordingStore();

  const { data: patient } = useQuery({
    queryKey: ['patient', currentPatientId],
    queryFn: () => fetchPatient(currentPatientId!),
    enabled: !!currentPatientId,
  });

  if (!isRecording || !isMinimized) return null;

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-50 animate-slide-up">
      <button 
        onClick={maximizeRecording}
        className="flex items-center gap-3 bg-[#0A1628] text-white px-5 py-3 rounded-full shadow-2xl border border-white/20 hover:scale-105 transition-transform"
      >
        <div className={cn(
          "h-2 w-2 rounded-full",
          isPaused ? "bg-warning" : "bg-danger animate-pulse"
        )} />
        
        <span className="font-bold text-sm">REC</span>
        
        <div className="w-px h-4 bg-white/20"></div>
        
        <span className="text-sm font-medium">{patient?.name || 'Session'}</span>
        
        <span className="font-mono text-sm opacity-80">{formatTime(elapsedSeconds)}</span>
        
        <div className="w-px h-4 bg-white/20"></div>
        
        <span className="text-xs opacity-60 font-medium flex items-center gap-1.5">
          Tap to return <Maximize2 size={14} />
        </span>
      </button>
    </div>
  );
}
