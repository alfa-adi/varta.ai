import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { mockAppointments } from '../data/mock';
import type { Appointment } from '../types';

interface AppointmentsState {
  appointments: Appointment[];
  addAppointment: (appointment: Appointment) => void;
  removeAppointment: (id: string) => void;
  getAppointmentsByDate: (date: string) => Appointment[];
}

export const useAppointmentsStore = create<AppointmentsState>()(
  persist(
    (set, get) => ({
      appointments: [...mockAppointments],
      
      addAppointment: (appointment) => set((state) => ({
        appointments: [...state.appointments, appointment]
      })),
      
      removeAppointment: (id) => set((state) => ({
        appointments: state.appointments.filter(a => a.id !== id)
      })),
      
      getAppointmentsByDate: (date) => {
        return get().appointments.filter(a => a.date === date);
      }
    }),
    {
      name: 'cpms-appointments',
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0 || !persistedState?.appointments) {
          return { appointments: [...mockAppointments] };
        }
        // Ensure each appointment has required fields with safe defaults
        const appointments = (persistedState.appointments as any[]).map((a: any) => ({
          duration: 15,
          urgency: 'routine',
          type: 'Consultation',
          ...a,
        }));
        return { ...persistedState, appointments };
      },
    }
  )
);
