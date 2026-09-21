import React, { useState, useMemo } from 'react';
import { usePatientDataStore } from '../../../stores/patientDataStore';
import { useReportsStore } from '../../../stores/reportsStore';
import { usePatientFormStore } from '../../../stores/patientFormStore';
import { Search, UserPlus, AlertCircle } from 'lucide-react';

interface PatientMatchControlsProps {
  reportId: string;
}

export function PatientMatchControls({ reportId }: PatientMatchControlsProps) {
  const [query, setQuery] = useState('');
  const patients = usePatientDataStore(state => state.patients);
  const matchPatient = useReportsStore(state => state.matchPatient);
  const openForm = usePatientFormStore(state => state.openForm);

  const filtered = useMemo(() => {
    if (!query.trim()) return patients.slice(0, 3);
    return patients.filter(p => 
      p.name.toLowerCase().includes(query.toLowerCase()) || 
      p.phone.includes(query) || 
      p.abhaId?.includes(query)
    ).slice(0, 5);
  }, [patients, query]);

  return (
    <div className="bg-surface-ground border border-border-subtle rounded-xl p-5 mb-6 shadow-sm">
      <div className="flex items-start gap-3 mb-4">
        <AlertCircle size={20} className="text-warning shrink-0 mt-0.5" />
        <div>
          <h4 className="font-bold text-text-primary">Unmatched Patient</h4>
          <p className="text-sm text-text-secondary mt-0.5">
            This report could not be automatically matched to an existing patient record. Please verify the patient details and select a match, or create a new profile.
          </p>
        </div>
      </div>

      <div className="flex gap-3 mb-4">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
          <input 
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search by name or phone..."
            className="w-full h-10 pl-9 pr-3 rounded-lg border border-border-subtle bg-surface-card text-sm focus:outline-none focus:border-brand-primary"
          />
        </div>
        <button 
          onClick={openForm}
          className="flex items-center gap-2 px-4 h-10 border border-border-subtle rounded-lg bg-surface-card text-sm font-semibold text-brand-primary hover:bg-brand-primary-xlt transition-colors shrink-0"
        >
          <UserPlus size={16} />
          New Patient
        </button>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Suggested Matches</p>
        {filtered.map(p => (
          <div key={p.id} className="flex items-center justify-between p-3 rounded-lg border border-border-subtle bg-surface-card hover:border-brand-primary/50 transition-colors">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-brand-primary text-text-on-brand flex items-center justify-center font-bold text-sm">
                {p.name.charAt(0)}
              </div>
              <div>
                <p className="font-bold text-sm text-text-primary">{p.name}</p>
                <p className="text-xs text-text-secondary">{p.phone} • {p.age}y {p.gender.charAt(0)}</p>
              </div>
            </div>
            <button 
              onClick={() => matchPatient(reportId, p.id, p.name)}
              className="px-4 py-1.5 rounded-full text-xs font-bold text-brand-primary bg-brand-primary-light hover:bg-brand-primary-mid hover:text-text-on-brand transition-colors"
            >
              Match
            </button>
          </div>
        ))}
        {filtered.length === 0 && (
          <p className="text-sm text-text-secondary text-center py-4">No matching patients found.</p>
        )}
      </div>
    </div>
  );
}
