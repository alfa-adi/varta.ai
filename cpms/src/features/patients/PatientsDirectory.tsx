import React, { useState, useMemo } from 'react';
import { UserPlus, SearchX, X } from 'lucide-react';
import { usePatientDataStore } from '../../stores/patientDataStore';
import { usePatientFormStore } from '../../stores/patientFormStore';
import { DirectoryToolbar } from './components/DirectoryToolbar';
import { DirectoryCard } from './components/DirectoryCard';
import { FilterSheet } from './components/FilterSheet';
import type { FilterState } from './components/FilterSheet';
import { AppShell } from '../../components/layout/AppShell';
import type { Patient } from '../../types';

const INITIAL_FILTERS: FilterState = {
  urgency: [],
  conditions: [],
  timeRange: 'all',
  language: [],
  archiveState: 'active'
};

export function PatientsDirectory() {
  const { patients } = usePatientDataStore();
  const { openForm } = usePatientFormStore();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('recent');
  const [filters, setFilters] = useState<FilterState>(INITIAL_FILTERS);
  const [isFilterSheetOpen, setIsFilterSheetOpen] = useState(false);

  // Compute active filter count
  const activeFilterCount = useMemo(() => {
    let count = filters.urgency.length + filters.conditions.length + filters.language.length;
    if (filters.timeRange !== 'all') count++;
    if (filters.archiveState !== 'active') count++;
    return count;
  }, [filters]);

  // Derived filtered & sorted patients
  const displayPatients = useMemo(() => {
    let result = [...patients];

    // 1. Archive Filter
    if (filters.archiveState === 'active') {
      result = result.filter(p => !p.isArchived);
    } else if (filters.archiveState === 'archived') {
      result = result.filter(p => p.isArchived);
    }

    // 2. Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(p => {
        const matchesName = p.name.toLowerCase().includes(q);
        const matchesPhone = p.phone.includes(q);
        const matchesAbha = p.abhaId?.toLowerCase().includes(q);
        const matchesCondition = p.chronicConditions.some(c => c.label.toLowerCase().includes(q)) || p.allergies.some(a => a.label.toLowerCase().includes(q));
        const matchesNotes = p.livingSummary?.toLowerCase().includes(q);
        return matchesName || matchesPhone || matchesAbha || matchesCondition || matchesNotes;
      });
    }

    // 3. Urgency Filter
    if (filters.urgency.length > 0) {
      result = result.filter(p => filters.urgency.includes(p.urgency));
    }

    // 4. Conditions Filter
    if (filters.conditions.length > 0) {
      result = result.filter(p => 
        filters.conditions.some(cond => p.chronicConditions.some(c => c.label === cond))
      );
    }

    // 5. Language Filter
    if (filters.language.length > 0) {
      result = result.filter(p => p.preferredLanguage && filters.language.includes(p.preferredLanguage));
    }

    // 6. Time Range Filter (simplified logic based on string parsing for demo)
    if (filters.timeRange !== 'all') {
      const now = new Date();
      result = result.filter(p => {
        const visitDate = new Date(p.lastVisitDate);
        if (isNaN(visitDate.getTime())) return true; // Keep if invalid date string
        const diffDays = (now.getTime() - visitDate.getTime()) / (1000 * 3600 * 24);
        
        if (filters.timeRange === '30days') return diffDays <= 30;
        if (filters.timeRange === '90days') return diffDays <= 90;
        if (filters.timeRange === '1year') return diffDays <= 365;
        return true;
      });
    }

    // 7. Sort
    result.sort((a, b) => {
      if (sortBy === 'name_asc') {
        return a.name.localeCompare(b.name);
      }
      if (sortBy === 'oldest') {
        return new Date(a.lastVisitDate).getTime() - new Date(b.lastVisitDate).getTime();
      }
      if (sortBy === 'urgency') {
        const order = { urgent: 0, attention: 1, routine: 2 };
        return order[a.urgency] - order[b.urgency];
      }
      // default: recent
      return new Date(b.lastVisitDate).getTime() - new Date(a.lastVisitDate).getTime();
    });

    return result;
  }, [patients, searchQuery, filters, sortBy]);

  const removeFilterChip = (type: 'urgency' | 'conditions' | 'language' | 'timeRange' | 'archiveState', value?: string) => {
    setFilters(prev => {
      if (type === 'timeRange') return { ...prev, timeRange: 'all' };
      if (type === 'archiveState') return { ...prev, archiveState: 'active' };
      
      const arr = prev[type as keyof FilterState] as string[];
      return {
        ...prev,
        [type]: arr.filter(item => item !== value)
      };
    });
  };

  return (
    <AppShell title="Patients">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header Row */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">All Patients</h1>
            <p className="text-text-secondary mt-1">{patients.length} total registered</p>
          </div>
          <button 
            onClick={openForm}
            className="flex items-center gap-2 px-5 py-2.5 bg-brand-primary hover:bg-brand-primary-mid text-text-on-brand rounded-xl font-bold text-sm shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-brand-primary-light"
          >
            <UserPlus size={18} />
            Add Patient
          </button>
        </div>

        {/* Toolbar */}
        <DirectoryToolbar 
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          filterCount={activeFilterCount}
          onToggleFilters={() => setIsFilterSheetOpen(true)}
          sortBy={sortBy}
          onSortChange={setSortBy}
        />

        {/* Active Filter Chips */}
        {activeFilterCount > 0 && (
          <div className="flex flex-wrap items-center gap-2 mb-6">
            <span className="text-xs font-bold text-text-tertiary uppercase tracking-wider mr-2">Active Filters:</span>
            
            {filters.archiveState !== 'active' && (
              <span className="flex items-center gap-1.5 px-3 py-1 bg-surface-card border border-border-subtle rounded-full text-xs font-semibold text-text-secondary shadow-sm">
                Status: {filters.archiveState === 'archived' ? 'Archived' : 'All'}
                <button onClick={() => removeFilterChip('archiveState')} className="hover:text-danger hover:bg-danger-light rounded-full p-0.5 transition-colors"><X size={12} /></button>
              </span>
            )}

            {filters.timeRange !== 'all' && (
              <span className="flex items-center gap-1.5 px-3 py-1 bg-surface-card border border-border-subtle rounded-full text-xs font-semibold text-text-secondary shadow-sm">
                Last Visit: {filters.timeRange}
                <button onClick={() => removeFilterChip('timeRange')} className="hover:text-danger hover:bg-danger-light rounded-full p-0.5 transition-colors"><X size={12} /></button>
              </span>
            )}

            {filters.urgency.map(urg => (
              <span key={urg} className="flex items-center gap-1.5 px-3 py-1 bg-brand-primary-light border border-brand-primary/20 rounded-full text-xs font-semibold text-brand-primary shadow-sm">
                Urgency: {urg}
                <button onClick={() => removeFilterChip('urgency', urg)} className="hover:text-danger hover:bg-danger-light rounded-full p-0.5 transition-colors"><X size={12} /></button>
              </span>
            ))}

            {filters.conditions.map(cond => (
              <span key={cond} className="flex items-center gap-1.5 px-3 py-1 bg-brand-primary-light border border-brand-primary/20 rounded-full text-xs font-semibold text-brand-primary shadow-sm">
                {cond}
                <button onClick={() => removeFilterChip('conditions', cond)} className="hover:text-danger hover:bg-danger-light rounded-full p-0.5 transition-colors"><X size={12} /></button>
              </span>
            ))}

            {filters.language.map(lang => (
              <span key={lang} className="flex items-center gap-1.5 px-3 py-1 bg-brand-primary-light border border-brand-primary/20 rounded-full text-xs font-semibold text-brand-primary shadow-sm">
                {lang}
                <button onClick={() => removeFilterChip('language', lang)} className="hover:text-danger hover:bg-danger-light rounded-full p-0.5 transition-colors"><X size={12} /></button>
              </span>
            ))}

            <button 
              onClick={() => setFilters(INITIAL_FILTERS)}
              className="text-xs font-bold text-text-tertiary hover:text-text-primary px-2 py-1 transition-colors underline underline-offset-2 ml-2"
            >
              Clear All
            </button>
          </div>
        )}

        {/* Directory Grid */}
        {displayPatients.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in pb-12">
            {displayPatients.map(patient => (
              <DirectoryCard key={patient.id} patient={patient} />
            ))}
          </div>
        ) : (
          /* Empty State */
          <div className="flex flex-col items-center justify-center py-20 bg-surface-card rounded-2xl border border-border-default border-dashed">
            <div className="w-16 h-16 bg-surface-ground rounded-full flex items-center justify-center text-text-tertiary mb-4">
              <SearchX size={32} />
            </div>
            <h3 className="text-lg font-bold text-text-primary mb-2">No patients found</h3>
            <p className="text-text-secondary text-sm max-w-sm text-center mb-6">
              Try adjusting your search query or filters. If this is a new patient, you can add them to the directory.
            </p>
            <button 
              onClick={openForm}
              className="px-6 py-2.5 bg-surface-ground hover:bg-brand-primary-light text-brand-primary border border-brand-primary/30 rounded-xl font-bold text-sm transition-colors flex items-center gap-2"
            >
              <UserPlus size={18} />
              Add New Patient
            </button>
          </div>
        )}
      </div>

      <FilterSheet 
        isOpen={isFilterSheetOpen}
        onClose={() => setIsFilterSheetOpen(false)}
        filters={filters}
        onChange={setFilters}
        onReset={() => setFilters(INITIAL_FILTERS)}
      />
    </AppShell>
  );
}
