import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type RecordingMode = 'Transcribe' | 'Translate';
export type DateFormat = 'DD/MM/YYYY' | 'MM/DD/YYYY' | 'YYYY-MM-DD';
export type TimeFormat = '12h' | '24h';

export interface ClinicProfile {
  name: string;
  address: string;
  phone: string;
  gstin: string;
  logoUrl?: string;
}

export interface DoctorProfile {
  name: string;
  firstName: string;
  qualification: string;
  regNumber: string;
  specialisation: string;
  photoUrl?: string;
  signatureUrl?: string;
}

export interface RecordingSettings {
  defaultMode: RecordingMode;
  defaultDoctorLanguage: string;
  defaultPatientLanguage: string;
  aiSummaryEnabled: boolean;
  confirmBeforeEnding: boolean;
}

export interface DisplaySettings {
  uiLanguage: string;
  dateFormat: DateFormat;
  timeFormat: TimeFormat;
  highContrast: boolean;
}

export interface NotificationSettings {
  followUpReminders: boolean;
  pendingRxReminders: boolean;
  quietHoursEnabled: boolean;
  quietHoursStart: string;
  quietHoursEnd: string;
}

export interface SettingsState {
  clinic: ClinicProfile;
  doctor: DoctorProfile;
  recording: RecordingSettings;
  display: DisplaySettings;
  notifications: NotificationSettings;
  
  updateClinic: (patch: Partial<ClinicProfile>) => void;
  updateDoctor: (patch: Partial<DoctorProfile>) => void;
  updateRecording: (patch: Partial<RecordingSettings>) => void;
  updateDisplay: (patch: Partial<DisplaySettings>) => void;
  updateNotifications: (patch: Partial<NotificationSettings>) => void;
}

const defaultSettings: Omit<SettingsState, 'updateClinic' | 'updateDoctor' | 'updateRecording' | 'updateDisplay' | 'updateNotifications'> = {
  clinic: {
    name: '',
    address: '',
    phone: '',
    gstin: '',
  },
  doctor: {
    name: '',
    firstName: '',
    qualification: '',
    regNumber: '',
    specialisation: '',
  },
  recording: {
    defaultMode: 'Transcribe',
    defaultDoctorLanguage: 'English',
    defaultPatientLanguage: 'Hindi',
    aiSummaryEnabled: true,
    confirmBeforeEnding: false,
  },
  display: {
    uiLanguage: 'English',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: '12h',
    highContrast: false,
  },
  notifications: {
    followUpReminders: true,
    pendingRxReminders: true,
    quietHoursEnabled: false,
    quietHoursStart: '22:00',
    quietHoursEnd: '07:00',
  },
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...defaultSettings,
      updateClinic: (patch) => set((s) => ({ clinic: { ...s.clinic, ...patch } })),
      updateDoctor: (patch) => set((s) => ({ doctor: { ...s.doctor, ...patch } })),
      updateRecording: (patch) => set((s) => ({ recording: { ...s.recording, ...patch } })),
      updateDisplay: (patch) => set((s) => ({ display: { ...s.display, ...patch } })),
      updateNotifications: (patch) => set((s) => ({ notifications: { ...s.notifications, ...patch } })),
    }),
    { name: 'cpms-settings' }
  )
);
