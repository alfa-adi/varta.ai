export type Urgency = 'routine' | 'attention' | 'urgent';
export type SessionType = 'Routine' | 'Follow-up' | 'Acute' | 'Review' | 'Labs';
export type RecordingMode = 'Transcribe' | 'Translate';

export interface TagItem {
  id: string;
  label: string;
  variant: 'symptom' | 'medicine' | 'test' | 'allergy' | 'chronic' | 'neutral';
}

export interface Patient {
  id: string;
  patientId: string;
  abhaId?: string;
  name: string;
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  bloodGroup?: string;
  phone: string;
  photo?: string;
  urgency: Urgency;
  allergies: TagItem[];
  chronicConditions: TagItem[];
  lastVisitDate: string;
  livingSummary: string;
  emergencyContact?: { name: string; phone: string };
  sessionCount?: number;
  isArchived?: boolean;
  preferredLanguage?: string;
}

export interface TranscriptLine {
  id: string;
  speaker: 'doctor' | 'patient';
  text: string;
  originalText?: string;
  timestamp: string;
  language?: string;
}

export interface Vital {
  label: string;
  value: string;
}

export interface PrescribedMedicine {
  id: string;
  name: string;
  frequency: string;
  duration: string;
  instructions?: string;
}

export interface Prescription {
  medicines: PrescribedMedicine[];
  tests: TagItem[];
  advice?: string;
  followUpDate?: string;
  qrEnabled?: boolean;
}

export interface LabFinding {
  id: string;
  testName: string;
  value: string;
  unit: string;
  referenceRange: string;
  status: 'normal' | 'high' | 'low';
}

export type ReportWorkflowStatus = 'needs_review' | 'reviewed' | 'unmatched';

export interface LabReport {
  id: string;
  patientId: string | null;
  patientName: string;
  reportType: string;
  labName: string;
  receivedAt: string;
  workflowStatus: ReportWorkflowStatus;
  isAbnormal: boolean;
  findings: LabFinding[];
  addedToRecord?: boolean;
}

export interface Report {
  id: string;
  title: string;
  date: string;
  status: 'pending' | 'reviewed';
  type: 'xray' | 'lab' | 'other';
}

export interface Session {
  id: string;
  sessionNumber: number;
  date: string;
  shortDate: string;
  duration: string;
  type: SessionType;
  mode: RecordingMode;
  languagePair?: string;
  chiefComplaint: string;
  symptoms: TagItem[];
  medicines: TagItem[];
  tests: TagItem[];
  vitals: Vital[];
  doctorNotes: string;
  followUpDate?: string;
  transcript: TranscriptLine[];
  prescription?: Prescription;
  reports: Report[];
}

export interface Appointment {
  id: string;
  patientId: string;
  patientName: string;
  date: string;
  time: string;
  duration: number;
  type: SessionType;
  urgency: Urgency;
  sessionNumber?: number;
  sourceReportId?: string;
  sourceReportContext?: string;
}

export interface DoctorProfile {
  name: string;
  firstName: string;
  qualification: string;
  regNumber: string;
  specialisation: string;
  clinic: {
    name: string;
    address: string;
    phone: string;
    website?: string;
    gstin?: string;
  };
}

export interface DashboardStats {
  seenToday: number;
  nextTwoHours: number;
  pendingRx: number;
  dueThisWeek: number;
}
