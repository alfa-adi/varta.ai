import { create } from 'zustand';

export interface DemoStep {
  title: string;
  hint: string;
  expectedPath?: string;
}

export const DEMO_STEPS: DemoStep[] = [
  {
    title: '1. Login',
    hint: 'Enter any 6-digit OTP to sign in to the pre-configured clinic.',
    expectedPath: '/login'
  },
  {
    title: '2. Dashboard',
    hint: 'Highlight today’s queue, pending reports, and appointments.',
    expectedPath: '/'
  },
  {
    title: '3. Global Search',
    hint: 'Tap the search icon or press Ctrl+K to find a patient quickly.',
  },
  {
    title: '4. Patient Profile',
    hint: 'Open a patient to view their timeline, then tap the central + button to start a session.',
  },
  {
    title: '5. Live Recording',
    hint: 'Show the active recording state. Tap End Session when ready.',
  },
  {
    title: '6. AI Review',
    hint: 'Review the generated summary. Tap Save & Issue Prescription.',
  },
  {
    title: '7. Prescription',
    hint: 'Highlight the structured medicines and print-ready layout.',
  },
  {
    title: '8. Reports Inbox',
    hint: 'Open Reports from the More tab. Show an abnormal lab result.',
  },
  {
    title: '9. Settings (Proof of Config)',
    hint: 'Show that clinic identity and preferences are fully configurable.',
    expectedPath: '/settings'
  },
  {
    title: '10. Summary',
    hint: 'End of demo. Highlight key outcomes: speed, accuracy, continuity.',
    expectedPath: '/demo'
  }
];

interface DemoState {
  isActive: boolean;
  isMinimized: boolean;
  currentStep: number;
  
  startDemo: () => void;
  endDemo: () => void;
  nextStep: () => void;
  prevStep: () => void;
  setMinimized: (min: boolean) => void;
}

export const useDemoStore = create<DemoState>((set) => ({
  isActive: false,
  isMinimized: false,
  currentStep: 0,
  
  startDemo: () => set({ isActive: true, currentStep: 0, isMinimized: false }),
  endDemo: () => set({ isActive: false }),
  nextStep: () => set(state => ({ currentStep: Math.min(state.currentStep + 1, DEMO_STEPS.length - 1) })),
  prevStep: () => set(state => ({ currentStep: Math.max(state.currentStep - 1, 0) })),
  setMinimized: (min) => set({ isMinimized: min })
}));

/**
 * Resets the application to a canonical presentation state.
 * Only targets CPMS namespace to avoid destroying unrelated browser data.
 */
export function resetToCanonicalDemoState() {
  // Clear only CPMS stores
  const keysToClear = [
    'cpms-auth',
    'cpms-settings',
    'cpms-patients',
    'cpms-reports',
    'cpms-appointments'
  ];

  keysToClear.forEach(key => localStorage.removeItem(key));

  // Seed the canonical Auth state so we skip the Setup Wizard
  localStorage.setItem('cpms-auth', JSON.stringify({
    state: {
      isAuthenticated: false,
      isFirstTimeSetup: false, // Bypass setup
      userPhone: null
    },
    version: 1
  }));

  // Seed the canonical Settings state
  localStorage.setItem('cpms-settings', JSON.stringify({
    state: {
      doctor: {
        name: 'Dr. Sarah Chen',
        specialty: 'General Practice',
        registrationNumber: 'MED-884920'
      },
      clinic: {
        name: 'Calm Family Clinic',
        address: '128 Wellness Avenue, Suite 300\nMetropolis, NY 10001',
        phone: '+1 (555) 019-2837',
        email: 'hello@calmclinic.com',
        website: 'www.calmclinic.com'
      },
      preferences: {
        defaultLanguage: 'en',
        printPaperSize: 'A4',
        includeVitals: true
      }
    },
    version: 1
  }));

  // Force a hard reload so Zustand re-hydrates with the new canonical local storage
  window.location.href = '/login';
}
