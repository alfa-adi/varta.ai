import React from 'react';
import { Activity, Play, RefreshCw, CheckCircle2 } from 'lucide-react';
import { useDemoStore, resetToCanonicalDemoState } from '../stores/demoStore';

export function DemoLanding() {
  const startDemo = useDemoStore(state => state.startDemo);

  const handleStartDemo = () => {
    // 1. Enable the demo controller globally
    startDemo();
    // 2. Wipes CPMS stores and seeds the canonical presentation state.
    // This will force a hard reload and take the user to /login.
    resetToCanonicalDemoState();
  };

  return (
    <div className="min-h-screen bg-surface-ground flex flex-col items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-surface-card rounded-2xl shadow-xl border border-border-subtle overflow-hidden">
        
        {/* Header */}
        <div className="bg-brand-primary p-8 text-text-on-brand text-center relative overflow-hidden">
          <div className="absolute inset-0 opacity-10 flex items-center justify-center">
            <Activity size={300} />
          </div>
          <div className="relative z-10">
            <div className="bg-white/20 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 backdrop-blur-sm">
              <Activity size={32} />
            </div>
            <h1 className="text-3xl font-bold mb-2">CPMS</h1>
            <p className="text-brand-primary-xlt text-lg font-medium opacity-90">
              Clinical Patient Management System
            </p>
            <p className="mt-4 text-sm font-semibold tracking-wide uppercase opacity-75">
              Document every word. Forget none of it.
            </p>
          </div>
        </div>

        {/* Agenda */}
        <div className="p-8 md:p-10">
          <h2 className="text-xl font-bold text-text-primary mb-6 text-center">
            Guided Demo Agenda
          </h2>
          
          <div className="space-y-4 max-w-md mx-auto">
            {[
              "Sign in to your configured clinic",
              "Start a consultation",
              "AI captures the session",
              "Review patient history",
              "Generate prescription",
              "Manage follow-up and reports",
              "Settings & Identity Config"
            ].map((step, idx) => (
              <div key={idx} className="flex items-center gap-4 bg-surface-ground p-3 rounded-xl border border-border-subtle">
                <div className="bg-brand-primary-light text-brand-primary w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shrink-0">
                  {idx + 1}
                </div>
                <span className="font-semibold text-text-secondary">{step}</span>
              </div>
            ))}
          </div>

          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={handleStartDemo}
              className="flex items-center justify-center gap-2 px-8 py-4 bg-brand-primary text-text-on-brand font-bold rounded-xl shadow-lg shadow-brand-primary/20 hover:bg-brand-primary-mid transition-all hover:scale-105 focus-ring"
            >
              <Play size={20} />
              Start Guided Demo
            </button>
            <button
              onClick={resetToCanonicalDemoState}
              className="flex items-center justify-center gap-2 px-8 py-4 bg-surface-ground text-text-secondary font-bold rounded-xl border border-border-subtle hover:bg-border-subtle hover:text-text-primary transition-all focus-ring"
            >
              <RefreshCw size={20} />
              Reset State Only
            </button>
          </div>

          <p className="text-center text-xs text-text-tertiary mt-8">
            This demo uses simulated local storage. No data is sent to external servers.
          </p>
        </div>
      </div>
    </div>
  );
}
