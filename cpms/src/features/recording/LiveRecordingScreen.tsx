import React, { useEffect } from 'react';
import { useRecordingStore } from '../../stores/recordingStore';
import { useQuery } from '@tanstack/react-query';
import { fetchPatient } from '../../data/mock';
import { Waveform } from './Waveform';
import { Minimize2, Square, Pause, Play, PenTool } from 'lucide-react';
import { cn } from '../../utils/cn';

export function LiveRecordingScreen() {
  const { 
    isRecording, isPaused, isMinimized, isReviewing, 
    elapsedSeconds, currentPatientId, mode, liveTranscript,
    pauseRecording, resumeRecording, minimizeRecording, endRecording, tick 
  } = useRecordingStore();

  const { data: patient } = useQuery({
    queryKey: ['patient', currentPatientId],
    queryFn: () => fetchPatient(currentPatientId!),
    enabled: !!currentPatientId,
  });

  // Timer effect
  useEffect(() => {
    if (isRecording && !isPaused && !isMinimized && !isReviewing) {
      const interval = setInterval(() => {
        tick();
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isRecording, isPaused, isMinimized, isReviewing, tick]);

  // Simulated live transcript generation
  useEffect(() => {
    if (isRecording && !isPaused && !isReviewing) {
      const interval = setInterval(() => {
        const dummyLines = [
          { speaker: 'doctor', text: "How have you been feeling since yesterday?", originalText: "कल से कैसा लग रहा है?" },
          { speaker: 'patient', text: "My fever is gone, but the cough persists.", originalText: "बुखार तो चला गया, लेकिन खांसी अभी भी है।" },
          { speaker: 'doctor', text: "Okay, I will prescribe a mild cough syrup for that.", originalText: "ठीक है, मैं उसके लिए कफ सिरप लिख देता हूँ।" }
        ];
        
        useRecordingStore.setState(state => {
          if (state.liveTranscript.length >= 3) return state; // Stop after 3 for demo
          const nextLine = dummyLines[state.liveTranscript.length];
          return {
            liveTranscript: [...state.liveTranscript, {
              id: `live-${state.liveTranscript.length}`,
              speaker: nextLine.speaker as 'doctor' | 'patient',
              text: nextLine.text,
              originalText: nextLine.originalText,
              timestamp: 'LIVE'
            }]
          };
        });
      }, 4000);
      return () => clearInterval(interval);
    }
  }, [isRecording, isPaused, isReviewing]);

  if (!isRecording || isMinimized || isReviewing) return null;

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="fixed inset-0 z-50 bg-[#0A1628] flex flex-col text-white animate-fade-in">
      
      {/* Top Bar */}
      <div className="h-16 flex items-center justify-between px-6 border-b border-white/10 shrink-0">
        <div className="flex items-center gap-4">
          <div className="h-10 w-10 rounded-full bg-brand-primary text-white flex items-center justify-center font-bold text-lg">
            {patient?.name.charAt(0) || 'P'}
          </div>
          <div>
            <h2 className="text-base font-bold text-white">{patient?.name || 'Patient'}</h2>
            <div className="flex items-center gap-2 text-xs">
              <span className={cn(
                "font-mono",
                isPaused ? "text-warning" : "text-danger animate-pulse"
              )}>
                {formatTime(elapsedSeconds)}
              </span>
              <span className="text-white/40">•</span>
              <span className="text-white/60 font-medium uppercase tracking-wider">{mode}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button 
            onClick={endRecording}
            className="flex items-center gap-2 px-4 py-2 bg-danger text-white hover:bg-danger/90 rounded-lg font-bold text-sm transition-colors shadow-sm"
          >
            <Square size={16} className="fill-white" />
            End Session
          </button>
          <div className="w-px h-6 bg-white/10 mx-1"></div>
          <button 
            onClick={minimizeRecording}
            className="p-2 text-white/60 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            title="Minimize"
          >
            <Minimize2 size={20} />
          </button>
        </div>
      </div>

      {/* Center Visualizer Area */}
      <div className="h-1/3 shrink-0 flex flex-col items-center justify-center p-6 relative">
        <Waveform active={!isPaused} />
        <div className="mt-6 text-sm font-medium text-white/60 flex items-center gap-2">
          {!isPaused ? (
            <>
              <div className="h-2 w-2 rounded-full bg-accent-green animate-pulse" />
              Doctor speaking...
            </>
          ) : (
            <span className="text-warning">Session Paused</span>
          )}
        </div>
      </div>

      {/* Bottom Half: Live Transcript */}
      <div className="flex-1 border-t border-white/10 bg-black/20 flex flex-col relative overflow-hidden">
        {/* Gradient fade at top of transcript */}
        <div className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-[#0A1628]/80 to-transparent z-10 pointer-events-none" />
        
        <div className="flex-1 overflow-y-auto p-6 flex flex-col justify-end gap-4">
          {liveTranscript.length === 0 ? (
            <div className="text-center text-white/40 text-sm pb-4">
              Listening...
            </div>
          ) : (
            liveTranscript.map((line) => {
              const isDoctor = line.speaker === 'doctor';
              return (
                <div 
                  key={line.id} 
                  className={cn(
                    "flex w-full max-w-[85%]",
                    isDoctor ? "self-end justify-end" : "self-start justify-start"
                  )}
                >
                  <div className="flex flex-col gap-1">
                    <span className={cn(
                      "text-[10px] font-medium px-1",
                      isDoctor ? "text-right text-brand-primary-light" : "text-left text-white/40"
                    )}>
                      {isDoctor ? 'Doctor' : 'Patient'}
                    </span>
                    
                    <div className={cn(
                      "px-4 py-2.5 rounded-2xl shadow-sm relative",
                      isDoctor 
                        ? "bg-brand-primary text-white rounded-tr-sm" 
                        : "bg-[#1A2639] border border-white/5 text-white/90 rounded-tl-sm"
                    )}>
                      <p className="text-sm leading-relaxed">{line.text}</p>
                      
                      {mode === 'Translate' && line.originalText && (
                        <div className="mt-2 pt-2 border-t border-white/10">
                          <p className="text-xs text-white/60 italic">{line.originalText}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="min-h-[96px] bg-[#050D18] flex items-center justify-between px-8 border-t border-white/5 shrink-0 pb-safe">
        <button className="flex items-center gap-2 text-white/60 hover:text-white transition-colors text-sm font-medium">
          <PenTool size={18} />
          Quick Note
        </button>

        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-6">
          <button 
            onClick={isPaused ? resumeRecording : pauseRecording}
            className="h-12 w-12 rounded-full bg-white/10 border border-white/20 flex items-center justify-center text-white hover:bg-white/20 transition-all"
            title={isPaused ? "Resume" : "Pause"}
          >
            {isPaused ? <Play size={20} className="ml-1" /> : <Pause size={20} />}
          </button>

          <div className="flex flex-col items-center -mt-2">
            <div className={cn(
              "h-16 w-16 rounded-full flex items-center justify-center transition-all",
              isPaused 
                ? "bg-white/10 border-2 border-white/20" 
                : "bg-danger shadow-[0_0_30px_rgba(220,38,38,0.5)] animate-pulse"
            )}>
              <div className={cn("rounded-sm transition-all", isPaused ? "h-5 w-5 bg-white/40" : "h-5 w-5 bg-white")} />
            </div>
          </div>
          
          <div className="w-12"></div> {/* spacer to keep REC strictly centered */}
        </div>
        
        <div className="w-24"></div> {/* Spacer for balance */}
      </div>

    </div>
  );
}
