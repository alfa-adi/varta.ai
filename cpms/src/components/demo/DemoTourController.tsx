import React from 'react';
import { useDemoStore, DEMO_STEPS } from '../../stores/demoStore';
import { useNavigate } from 'react-router-dom';
import { Play, ChevronLeft, ChevronRight, X, Minimize2, Maximize2 } from 'lucide-react';
import { cn } from '../../utils/cn';

export function DemoTourController() {
  const { isActive, isMinimized, currentStep, nextStep, prevStep, endDemo, setMinimized } = useDemoStore();
  const navigate = useNavigate();

  if (!isActive) return null;

  const step = DEMO_STEPS[currentStep];
  const isFirst = currentStep === 0;
  const isLast = currentStep === DEMO_STEPS.length - 1;

  const handleNext = () => {
    if (isLast) {
      endDemo();
      navigate('/demo');
      return;
    }
    
    const nextPath = DEMO_STEPS[currentStep + 1]?.expectedPath;
    if (nextPath) {
      navigate(nextPath);
    }
    nextStep();
  };

  const handlePrev = () => {
    if (isFirst) return;
    
    const prevPath = DEMO_STEPS[currentStep - 1]?.expectedPath;
    if (prevPath) {
      navigate(prevPath);
    }
    prevStep();
  };

  const handleSkip = () => {
    endDemo();
    navigate('/demo');
  };

  if (isMinimized) {
    return (
      <div className="fixed bottom-[4.5rem] md:bottom-6 right-4 md:right-6 z-[100] animate-fade-in">
        <button
          onClick={() => setMinimized(false)}
          className="flex items-center gap-2 bg-brand-primary text-text-on-brand px-4 py-3 rounded-full shadow-xl shadow-brand-primary/20 font-bold hover:scale-105 transition-all focus-ring"
          aria-label="Maximize demo controller"
        >
          <Play size={16} />
          Demo: Step {currentStep + 1}
        </button>
      </div>
    );
  }

  return (
    <div className={cn(
      "fixed z-[100] animate-slide-up shadow-2xl border border-border-subtle",
      // Desktop: bottom right floating card
      "md:bottom-6 md:right-6 md:w-80 md:rounded-2xl bg-surface-card",
      // Mobile: full width banner above bottom nav
      "bottom-[4.5rem] left-2 right-2 rounded-xl md:left-auto"
    )}>
      <div className="bg-brand-primary text-text-on-brand p-3 rounded-t-xl md:rounded-t-2xl flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Play size={16} />
          <span className="font-bold text-sm tracking-wide">
            Step {currentStep + 1} of {DEMO_STEPS.length}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button 
            onClick={() => setMinimized(true)}
            className="p-1 hover:bg-white/20 rounded focus-ring"
            aria-label="Minimize"
          >
            <Minimize2 size={16} />
          </button>
          <button 
            onClick={handleSkip}
            className="p-1 hover:bg-white/20 rounded focus-ring"
            aria-label="End Demo"
          >
            <X size={16} />
          </button>
        </div>
      </div>
      
      <div className="p-4 bg-surface-card rounded-b-xl md:rounded-b-2xl">
        <h3 className="font-bold text-text-primary mb-1">{step.title}</h3>
        <p className="text-sm text-text-secondary mb-4 min-h-[40px]">
          {step.hint}
        </p>
        
        <div className="flex items-center justify-between mt-4">
          <button
            onClick={handleSkip}
            className="text-xs font-bold text-text-tertiary hover:text-text-primary transition-colors focus-ring rounded"
          >
            Skip Demo
          </button>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrev}
              disabled={isFirst}
              className="p-2 rounded-lg border border-border-subtle text-text-secondary hover:bg-surface-ground disabled:opacity-50 disabled:pointer-events-none focus-ring"
              aria-label="Previous step"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={handleNext}
              className="px-4 py-2 bg-brand-primary text-text-on-brand rounded-lg font-bold hover:bg-brand-primary-mid transition-colors flex items-center gap-1 focus-ring"
            >
              {isLast ? 'Finish' : 'Next'}
              {!isLast && <ChevronRight size={16} className="-mr-1" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
