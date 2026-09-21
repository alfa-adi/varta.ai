import { Routes, Route } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { PatientsDirectory } from './features/patients/PatientsDirectory';
import { PatientProfile } from './pages/PatientProfile';
import { RecordingManager } from './features/recording/RecordingManager';
import { PrescriptionPage } from './features/prescription/PrescriptionPage';
import { CalendarPage } from './features/calendar/CalendarPage';
import { GlobalSearch } from './features/search/GlobalSearch';
import { PatientFormDrawer } from './features/patient/PatientFormDrawer';
import { SettingsPage } from './features/settings/SettingsPage';
import { ReportsPage } from './features/reports/ReportsPage';
import { LoginPage } from './features/auth/LoginPage';
import { SetupWizard } from './features/auth/SetupWizard';
import { AuthGuard } from './components/layout/AuthGuard';
import { ToastProvider } from './components/ui/ToastProvider';
import { useGlobalKeyboard } from './hooks/useGlobalKeyboard';
import { DemoLanding } from './pages/DemoLanding';
import { DemoTourController } from './components/demo/DemoTourController';

function App() {
  useGlobalKeyboard();

  return (
    <>
      <Routes>
        <Route path="/demo" element={<DemoLanding />} />
        <Route path="/login" element={<AuthGuard allowPublic><LoginPage /></AuthGuard>} />
        <Route path="/setup" element={<AuthGuard allowPublic><SetupWizard /></AuthGuard>} />
        <Route path="/" element={<AuthGuard><Dashboard /></AuthGuard>} />
        <Route path="/patients" element={<AuthGuard><PatientsDirectory /></AuthGuard>} />
        <Route path="/patients/:id" element={<AuthGuard><PatientProfile /></AuthGuard>} />
        <Route path="/patients/:patientId/prescription/:sessionId" element={<AuthGuard><PrescriptionPage /></AuthGuard>} />
        <Route path="/appointments" element={<AuthGuard><CalendarPage /></AuthGuard>} />
        <Route path="/reports" element={<AuthGuard><ReportsPage /></AuthGuard>} />
        <Route path="/reports/:reportId" element={<AuthGuard><ReportsPage /></AuthGuard>} />
        <Route path="/settings" element={<AuthGuard><SettingsPage /></AuthGuard>} />
        <Route path="/settings/:sectionId" element={<AuthGuard><SettingsPage /></AuthGuard>} />
      </Routes>
      <RecordingManager />
      <GlobalSearch />
      <PatientFormDrawer />
      <ToastProvider />
      <DemoTourController />
    </>
  );
}

export default App;
