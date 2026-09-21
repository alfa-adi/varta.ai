import { create } from 'zustand';

interface PatientFormState {
  isOpen: boolean;
  openForm: () => void;
  closeForm: () => void;
}

export const usePatientFormStore = create<PatientFormState>((set) => ({
  isOpen: false,
  openForm: () => set({ isOpen: true }),
  closeForm: () => set({ isOpen: false }),
}));
