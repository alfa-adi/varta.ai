import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  isAuthenticated: boolean;
  /** True only until the user completes the one-time clinic setup wizard */
  isFirstTimeSetup: boolean;
  userPhone: string | null;
  /** Runtime-only flag — never persisted */
  _hasHydrated: boolean;
  
  login: (phone: string) => void;
  completeSetup: () => void;
  logout: () => void;
  /** Dev/demo only — resets everything including setup flag for onboarding testing */
  resetDemoSession: () => void;
  setHasHydrated: (state: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      isAuthenticated: false,
      isFirstTimeSetup: true,
      userPhone: null,
      _hasHydrated: false,

      /**
       * Log in. `isFirstTimeSetup` is NOT reset here — it persists from
       * the previous session so returning users bypass setup automatically.
       */
      login: (phone) => set((state) => ({ 
        isAuthenticated: true, 
        userPhone: phone,
        // Keep existing isFirstTimeSetup value from persisted store.
        // If it's already false (setup was completed), stay false.
        isFirstTimeSetup: state.isFirstTimeSetup,
      })),

      completeSetup: () => set({
        isFirstTimeSetup: false
      }),

      logout: () => set({
        isAuthenticated: false,
        userPhone: null,
        // Preserve isFirstTimeSetup so returning users don't redo setup.
      }),

      /**
       * ⚠️ DEMO/DEV ONLY — resets to first-time state for onboarding testing.
       * Should only be called from Settings > Account or the dev tools panel.
       */
      resetDemoSession: () => set({
        isAuthenticated: false,
        isFirstTimeSetup: true,
        userPhone: null
      }),

      setHasHydrated: (state) => set({
        _hasHydrated: state
      })
    }),
    {
      name: 'cpms-auth',
      version: 1,
      // Never persist the runtime hydration flag
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        isFirstTimeSetup: state.isFirstTimeSetup,
        userPhone: state.userPhone,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) state.setHasHydrated(true);
      },
    }
  )
);
