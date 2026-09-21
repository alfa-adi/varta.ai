import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { mockPatients } from '../data/mock';
import type { Patient } from '../types';

interface PatientDataState {
  patients: Patient[];
  addPatient: (patient: Patient) => void;
  updatePatient: (id: string, updates: Partial<Patient>) => void;
  archivePatient: (id: string, isArchived: boolean) => void;
  getPatientById: (id: string) => Patient | undefined;
  findDuplicate: (phone: string, abhaId?: string) => Patient | undefined;
  resetDemoPatients: () => void;
}

// Helper to normalize phone (extract last 10 digits)
export const normalizePhone = (phone: string) => {
  const digits = phone.replace(/\D/g, '');
  return digits.slice(-10);
};

// Helper to normalize ABHA (remove hyphens/spaces)
export const normalizeABHA = (abha: string) => {
  return abha.replace(/[\s-]/g, '').toLowerCase();
};

export const usePatientDataStore = create<PatientDataState>()(
  persist(
    (set, get) => ({
      patients: [...mockPatients], // Initialize with mock data
      
      addPatient: (patient) => set((state) => ({ 
        patients: [...state.patients, { ...patient, isArchived: false }] 
      })),
      
      updatePatient: (id, updates) => set((state) => ({
        patients: state.patients.map(p => p.id === id ? { ...p, ...updates } : p)
      })),
      
      archivePatient: (id, isArchived) => set((state) => ({
        patients: state.patients.map(p => p.id === id ? { ...p, isArchived } : p)
      })),
      
      getPatientById: (id) => get().patients.find(p => p.id === id),
      
      findDuplicate: (phone, abhaId) => {
        const { patients } = get();
        const normalizedPhone = normalizePhone(phone);
        const normalizedAbha = abhaId ? normalizeABHA(abhaId) : null;
        
        return patients.find(p => {
          const pPhone = normalizePhone(p.phone);
          if (pPhone === normalizedPhone && normalizedPhone.length === 10) return true;
          
          if (normalizedAbha && p.abhaId) {
            const pAbha = normalizeABHA(p.abhaId);
            if (pAbha === normalizedAbha && normalizedAbha.length > 5) return true;
          }
          
          return false;
        });
      },
      
      resetDemoPatients: () => set({ patients: [...mockPatients] })
    }),
    {
      name: 'cpms-patients',
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0 || !persistedState?.patients) {
          return { patients: [...mockPatients] };
        }
        // Ensure every patient has required fields with safe defaults
        const patients = (persistedState.patients as any[]).map((p: any) => ({
          isArchived: false,
          allergies: [],
          chronicConditions: [],
          urgency: 'routine',
          livingSummary: '',
          ...p,
        }));
        return { ...persistedState, patients };
      },
    }
  )
);
