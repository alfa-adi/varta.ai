import { create } from 'zustand';
import type { TranscriptLine } from '../types';

export type RecordingMode = 'Transcribe' | 'Translate';

interface RecordingState {
  launcherOpen: boolean;
  isRecording: boolean;
  isPaused: boolean;
  isMinimized: boolean;
  isReviewing: boolean;
  elapsedSeconds: number;
  currentPatientId: string | null;
  mode: RecordingMode;
  liveTranscript: TranscriptLine[];
  
  // Actions
  openLauncher: (patientId?: string) => void;
  closeLauncher: () => void;
  setMode: (mode: RecordingMode) => void;
  setPatient: (patientId: string) => void;
  
  startRecording: () => void;
  pauseRecording: () => void;
  resumeRecording: () => void;
  minimizeRecording: () => void;
  maximizeRecording: () => void;
  endRecording: () => void;
  
  discardSession: () => void;
  saveSession: () => void;
  
  tick: () => void; // for timer
}

const initialState = {
  launcherOpen: false,
  isRecording: false,
  isPaused: false,
  isMinimized: false,
  isReviewing: false,
  elapsedSeconds: 0,
  currentPatientId: null,
  mode: 'Transcribe' as RecordingMode,
  liveTranscript: [],
};

export const useRecordingStore = create<RecordingState>((set) => ({
  ...initialState,
  
  openLauncher: (patientId) => set({ launcherOpen: true, currentPatientId: patientId || null }),
  closeLauncher: () => set({ launcherOpen: false }),
  setMode: (mode) => set({ mode }),
  setPatient: (patientId) => set({ currentPatientId: patientId }),
  
  startRecording: () => set({
    launcherOpen: false,
    isRecording: true,
    isPaused: false,
    isMinimized: false,
    isReviewing: false,
    elapsedSeconds: 0,
    liveTranscript: []
  }),
  
  pauseRecording: () => set({ isPaused: true }),
  resumeRecording: () => set({ isPaused: false }),
  minimizeRecording: () => set({ isMinimized: true }),
  maximizeRecording: () => set({ isMinimized: false }),
  
  endRecording: () => set({
    isRecording: false,
    isPaused: false,
    isMinimized: false,
    isReviewing: true
  }),
  
  discardSession: () => set({ ...initialState }),
  
  saveSession: () => set({ ...initialState }), // Will handle navigation externally
  
  tick: () => set((state) => ({
    elapsedSeconds: (state.isRecording && !state.isPaused) ? state.elapsedSeconds + 1 : state.elapsedSeconds
  })),
}));
