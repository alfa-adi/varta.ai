import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Search, X, UserPlus, Mic, FileText, User, Calendar, Clock, Activity } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useSearchStore } from '../../stores/searchStore';
import { useRecordingStore } from '../../stores/recordingStore';
import { usePatientDataStore } from '../../stores/patientDataStore';
import { usePatientFormStore } from '../../stores/patientFormStore';
import { mockSessions } from '../../data/mock';
import type { Patient, Session } from '../../types';
import { cn } from '../../utils/cn';

type ResultItem = 
  | { type: 'action'; id: string; label: string; icon: React.ReactNode; onSelect: () => void }
  | { type: 'patient'; id: string; patient: Patient; onSelect: () => void }
  | { type: 'session'; id: string; session: Session; patientName: string; patientId: string; onSelect: () => void };

export function GlobalSearch() {
  const { isOpen, closeSearch, query, setQuery } = useSearchStore();
  const openLauncher = useRecordingStore(state => state.openLauncher);
  const openPatientForm = usePatientFormStore(state => state.openForm);
  const patients = usePatientDataStore(state => state.patients);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [activeIndex, setActiveIndex] = useState(0);

  // Close on Escape or Backdrop Click
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeSearch();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      // Prevent body scrolling
      document.body.style.overflow = 'hidden';
      // Auto focus input
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      document.body.style.overflow = '';
      setActiveIndex(0);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, closeSearch]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) closeSearch();
  };

  // Search Logic
  const results = useMemo(() => {
    const q = query.toLowerCase().trim();
    
    // Base actions
    const quickActions: ResultItem[] = [
      { 
        type: 'action', 
        id: 'action-new-session', 
        label: 'New Session', 
        icon: <Mic size={18} className="text-brand-primary" />,
        onSelect: () => { closeSearch(); openLauncher(); }
      },
      {
        type: 'action',
        id: 'action-add-patient',
        label: 'Add Patient',
        icon: <UserPlus size={18} className="text-brand-primary" />,
        onSelect: () => { closeSearch(); openPatientForm(); }
      },
      {
        type: 'action',
        id: 'action-quick-note',
        label: 'Quick Note',
        icon: <FileText size={18} className="text-brand-primary" />,
        onSelect: () => { closeSearch(); /* placeholder */ }
      }
    ];

    if (!q) {
      // Default State: Recent Patients (first 2), Recent Sessions (first 2 from patient 1), Actions
      const recentPatients: ResultItem[] = patients.slice(0, 2).map(p => ({
        type: 'patient', id: `pat-${p.id}`, patient: p, onSelect: () => { closeSearch(); navigate(`/patients/${p.id}`); }
      }));
      const recentSessions: ResultItem[] = (mockSessions['1'] || []).slice(0, 2).map(s => ({
        type: 'session', id: `sess-${s.id}`, session: s, patientName: patients[0]?.name || 'Unknown', patientId: '1', onSelect: () => { closeSearch(); navigate(`/patients/1/session/${s.id}`); }
      }));
      
      return [
        { group: 'Quick Actions', items: quickActions },
        { group: 'Recent Patients', items: recentPatients },
        { group: 'Recent Sessions', items: recentSessions }
      ];
    }

    // Filter Patients
    const matchedPatients = patients.filter(p => 
      p.name.toLowerCase().includes(q) || 
      p.phone.includes(q) || 
      (p.abhaId && p.abhaId.toLowerCase().includes(q)) ||
      p.chronicConditions.some(c => c.label.toLowerCase().includes(q)) ||
      p.allergies.some(a => a.label.toLowerCase().includes(q))
    ).map(p => ({
      type: 'patient' as const, id: `pat-${p.id}`, patient: p, onSelect: () => { closeSearch(); navigate(`/patients/${p.id}`); }
    }));

    // Filter Sessions
    const matchedSessions: ResultItem[] = [];
    Object.keys(mockSessions).forEach(patientId => {
      const patient = patients.find(p => p.id === patientId);
      if (!patient) return;
      
      const sessions = mockSessions[patientId].filter(s => 
        s.chiefComplaint.toLowerCase().includes(q) ||
        s.doctorNotes.toLowerCase().includes(q) ||
        s.symptoms.some(sym => sym.label.toLowerCase().includes(q))
      );
      
      sessions.forEach(s => {
        matchedSessions.push({
          type: 'session', id: `sess-${s.id}`, session: s, patientName: patient.name, patientId: patient.id,
          onSelect: () => { closeSearch(); navigate(`/patients/${patient.id}/session/${s.id}`); }
        });
      });
    });

    const out = [];
    if (matchedPatients.length > 0) out.push({ group: 'Patients', items: matchedPatients });
    if (matchedSessions.length > 0) out.push({ group: 'Sessions', items: matchedSessions });
    
    return out;
  }, [query, closeSearch, navigate, openLauncher, openPatientForm, patients]);

  const flatList = useMemo(() => results.flatMap(g => g.items), [results]);

  // Reset active index when query changes
  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  // Keyboard Navigation
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex(prev => (prev + 1) % flatList.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex(prev => (prev - 1 + flatList.length) % flatList.length);
      } else if (e.key === 'Enter' && flatList.length > 0) {
        e.preventDefault();
        flatList[activeIndex].onSelect();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, flatList, activeIndex]);

  // Scroll active item into view
  useEffect(() => {
    if (isOpen && containerRef.current) {
      const activeEl = containerRef.current.querySelector('[data-active="true"]');
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [activeIndex, isOpen]);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-end md:items-start justify-center md:pt-[10vh] bg-slate-900/30 backdrop-blur-sm animate-fade-in"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-label="Global Search"
    >
      <div 
        className="w-full h-[90vh] md:h-auto max-w-[720px] md:mx-4 bg-surface-card rounded-t-2xl md:rounded-2xl shadow-2xl overflow-hidden border-t md:border border-border-subtle flex flex-col md:max-h-[80vh] animate-slide-up pb-safe"
      >
        {/* Input Header */}
        <div className="flex items-center px-4 py-4 border-b border-border-subtle relative shrink-0">
          <Search size={22} className="text-brand-primary ml-2 mr-3 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            aria-label="Search query"
            className="flex-1 bg-transparent text-lg text-text-primary placeholder:text-text-tertiary focus:outline-none"
            placeholder="Search patients, phone, ABHA, sessions..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-activedescendant={flatList[activeIndex]?.id}
          />
          <div className="flex items-center gap-2">
            <kbd className="hidden sm:inline-block px-2 py-1 text-xs font-semibold text-text-secondary bg-surface-ground border border-border-subtle rounded uppercase shadow-sm">
              Esc
            </kbd>
            <button 
              onClick={closeSearch}
              className="p-1.5 text-text-tertiary hover:bg-surface-ground rounded-full transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Results Body */}
        <div 
          className="flex-1 overflow-y-auto p-2"
          ref={containerRef}
          role="listbox"
        >
          {flatList.length === 0 ? (
            <div className="py-16 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-brand-primary-xlt rounded-full flex items-center justify-center mb-4">
                <Search size={24} className="text-brand-primary" />
              </div>
              <p className="text-lg font-bold text-text-primary mb-1">No matching results found</p>
              <p className="text-sm text-text-secondary mb-6 max-w-sm">
                We couldn't find any patients or sessions matching "{query}".
              </p>
              <button 
                onClick={() => { closeSearch(); openPatientForm(); }}
                className="flex items-center gap-2 px-5 py-2.5 bg-brand-primary text-text-on-brand rounded-full font-medium hover:bg-brand-primary-hover transition-colors shadow-sm"
              >
                <UserPlus size={18} />
                Add New Patient
              </button>
            </div>
          ) : (
            results.map((group, groupIndex) => {
              // Calculate global offset for this group
              const startIndex = results.slice(0, groupIndex).reduce((acc, g) => acc + g.items.length, 0);
              
              return (
                <div key={group.group} className="mb-4 last:mb-0">
                  <h4 className="px-3 py-1.5 text-xs font-bold text-text-tertiary uppercase tracking-wider">
                    {group.group}
                  </h4>
                  <div className="space-y-1">
                    {group.items.map((item, itemIndex) => {
                      const globalIndex = startIndex + itemIndex;
                      const isActive = activeIndex === globalIndex;
                      
                      if (item.type === 'action') {
                        return (
                          <div
                            key={item.id}
                            id={item.id}
                            role="option"
                            aria-selected={isActive}
                            data-active={isActive}
                            onClick={item.onSelect}
                            onMouseEnter={() => setActiveIndex(globalIndex)}
                            className={cn(
                              "flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all border",
                              isActive ? "bg-brand-primary-xlt border-brand-primary/30 shadow-sm" : "border-transparent hover:bg-surface-ground"
                            )}
                          >
                            <div className="w-8 h-8 rounded-full bg-surface-card flex items-center justify-center border border-border-subtle shadow-sm shrink-0">
                              {item.icon}
                            </div>
                            <span className="font-semibold text-text-primary text-sm">{item.label}</span>
                          </div>
                        );
                      }
                      
                      if (item.type === 'patient') {
                        return (
                          <div
                            key={item.id}
                            id={item.id}
                            role="option"
                            aria-selected={isActive}
                            data-active={isActive}
                            onClick={item.onSelect}
                            onMouseEnter={() => setActiveIndex(globalIndex)}
                            className={cn(
                              "flex items-center gap-4 px-3 py-2.5 rounded-lg cursor-pointer transition-all border",
                              isActive ? "bg-brand-primary-xlt border-brand-primary/30 shadow-sm" : "border-transparent hover:bg-surface-ground"
                            )}
                          >
                            <div className="w-10 h-10 rounded-full bg-brand-primary text-text-on-brand flex items-center justify-center font-bold text-sm shrink-0 shadow-sm">
                              {item.patient.name.split(' ').map(n => n[0]).join('')}
                            </div>
                            <div className="flex-1 min-w-0 flex flex-col justify-center">
                              <div className="flex justify-between items-baseline mb-0.5">
                                <span className="font-bold text-text-primary truncate text-[15px]">{item.patient.name}</span>
                                <span className="text-xs font-medium text-text-secondary shrink-0">{item.patient.phone}</span>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-text-secondary truncate">
                                <span>{item.patient.age}y • {item.patient.gender}</span>
                                {item.patient.chronicConditions.length > 0 && (
                                  <>
                                    <span className="text-text-tertiary">•</span>
                                    <span className="truncate text-warning-strong font-medium">
                                      {item.patient.chronicConditions.map(c => c.label).join(', ')}
                                    </span>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      }
                      
                      if (item.type === 'session') {
                        return (
                          <div
                            key={item.id}
                            id={item.id}
                            role="option"
                            aria-selected={isActive}
                            data-active={isActive}
                            onClick={item.onSelect}
                            onMouseEnter={() => setActiveIndex(globalIndex)}
                            className={cn(
                              "flex items-start gap-4 px-3 py-3 rounded-lg cursor-pointer transition-all border",
                              isActive ? "bg-brand-primary-xlt border-brand-primary/30 shadow-sm" : "border-transparent hover:bg-surface-ground"
                            )}
                          >
                            <div className="w-10 h-10 rounded-full bg-surface-card border border-border-subtle shadow-sm flex items-center justify-center text-text-tertiary shrink-0 mt-0.5">
                              <Activity size={18} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex justify-between items-start mb-1">
                                <span className="font-bold text-text-primary text-[15px] truncate pr-2">
                                  {item.session.chiefComplaint}
                                </span>
                                <span className="text-[11px] font-bold px-1.5 py-0.5 rounded bg-surface-card border border-border-subtle text-text-tertiary shrink-0 uppercase tracking-wider mt-0.5">
                                  {item.session.date}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-xs font-medium text-text-secondary">
                                <div className="flex items-center gap-1.5 text-brand-primary">
                                  <User size={12} />
                                  <span className="truncate max-w-[120px]">{item.patientName}</span>
                                </div>
                                {item.session.languagePair && (
                                  <>
                                    <span className="text-text-tertiary">|</span>
                                    <span className="text-text-secondary">{item.session.languagePair}</span>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      }
                    })}
                  </div>
                </div>
              );
            })
          )}
        </div>
        
        {/* Footer */}
        <div className="border-t border-border-subtle bg-surface-ground px-4 py-2.5 flex items-center justify-between text-xs font-medium text-text-tertiary shrink-0">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5"><kbd className="bg-surface-card border border-border-subtle px-1.5 rounded shadow-sm">↑</kbd> <kbd className="bg-surface-card border border-border-subtle px-1.5 rounded shadow-sm">↓</kbd> Navigate</span>
            <span className="flex items-center gap-1.5"><kbd className="bg-surface-card border border-border-subtle px-1.5 rounded shadow-sm">Enter</kbd> Select</span>
          </div>
          <div>Calm Clinic CPMS</div>
        </div>
      </div>
    </div>
  );
}
