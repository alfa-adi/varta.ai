import React from 'react';
import type { TranscriptLine } from '../../types';
import { cn } from '../../utils/cn';
import { Search, Download, PlayCircle, Languages } from 'lucide-react';

interface TranscriptBubblesProps {
  transcript: TranscriptLine[];
  mode?: 'Transcribe' | 'Translate';
}

export function TranscriptBubbles({ transcript, mode = 'Transcribe' }: TranscriptBubblesProps) {
  const hasTranslation = mode === 'Translate';

  return (
    <div className="flex flex-col h-full bg-surface-ground">
      
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 border-b border-border-subtle bg-surface-card sticky top-0 z-10 rounded-t-xl">
        <div className="flex items-center gap-2">
          <button className="p-1.5 text-text-tertiary hover:text-brand-primary transition-colors rounded-md hover:bg-brand-primary-xlt" title="Search Transcript">
            <Search size={18} />
          </button>
          <button className="p-1.5 text-text-tertiary hover:text-brand-primary transition-colors rounded-md hover:bg-brand-primary-xlt" title="Play Audio">
            <PlayCircle size={18} />
          </button>
        </div>
        
        <div className="flex items-center gap-3">
          {hasTranslation && (
            <button className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-brand-primary bg-brand-primary-xlt rounded-md border border-brand-primary-light">
              <Languages size={14} />
              Translation ON
            </button>
          )}
          <button className="p-1.5 text-text-tertiary hover:text-brand-primary transition-colors rounded-md hover:bg-brand-primary-xlt" title="Download Transcript">
            <Download size={18} />
          </button>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 p-3 overflow-y-auto flex flex-col gap-4">
        {transcript.map((line) => {
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
                  isDoctor ? "text-right text-brand-primary" : "text-left text-text-secondary"
                )}>
                  {isDoctor ? 'Doctor' : 'Patient'}
                </span>
                
                <div className={cn(
                  "px-3 py-2 rounded-2xl shadow-sm relative",
                  isDoctor 
                    ? "bg-brand-primary-light text-text-primary rounded-tr-sm" 
                    : "bg-surface-card border border-border-subtle text-text-primary rounded-tl-sm"
                )}>
                  <p className="text-sm leading-relaxed">{line.text}</p>
                  
                  {hasTranslation && line.originalText && (
                    <div className="mt-2 pt-2 border-t border-black/5">
                      <p className="text-xs text-text-secondary italic">{line.originalText}</p>
                    </div>
                  )}
                  
                  <span className="block text-[10px] text-text-tertiary mt-1.5 text-right w-full">
                    {line.timestamp}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
        {transcript.length === 0 && (
          <div className="flex-1 flex items-center justify-center text-text-tertiary text-sm">
            No transcript available for this session.
          </div>
        )}
      </div>

    </div>
  );
}
