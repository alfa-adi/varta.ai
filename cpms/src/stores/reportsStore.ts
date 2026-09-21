import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { LabReport, LabFinding } from '../types';

const mockFindingsNormal: LabFinding[] = [
  { id: 'f1', testName: 'Hemoglobin', value: '14.2', unit: 'g/dL', referenceRange: '13.5 - 17.5', status: 'normal' },
  { id: 'f2', testName: 'WBC Count', value: '7,500', unit: 'cells/mcL', referenceRange: '4,500 - 11,000', status: 'normal' },
  { id: 'f3', testName: 'Platelets', value: '250,000', unit: 'cells/mcL', referenceRange: '150,000 - 450,000', status: 'normal' },
];

const mockFindingsAbnormal: LabFinding[] = [
  { id: 'f4', testName: 'Fasting Blood Sugar', value: '145', unit: 'mg/dL', referenceRange: '70 - 100', status: 'high' },
  { id: 'f5', testName: 'HbA1c', value: '7.2', unit: '%', referenceRange: '< 5.7', status: 'high' },
  { id: 'f6', testName: 'Cholesterol', value: '210', unit: 'mg/dL', referenceRange: '< 200', status: 'high' },
];

const initialReports: LabReport[] = [
  {
    id: 'r1',
    patientId: '3',
    patientName: 'Amit Patel',
    reportType: 'Complete Blood Count & Lipid Profile',
    labName: 'Metropolis Healthcare',
    receivedAt: new Date(Date.now() - 3600000).toISOString(),
    workflowStatus: 'needs_review',
    isAbnormal: true,
    findings: mockFindingsAbnormal,
    addedToRecord: false
  },
  {
    id: 'r2',
    patientId: '1',
    patientName: 'Rajesh Kumar',
    reportType: 'Routine Blood Test',
    labName: 'Thyrocare',
    receivedAt: new Date(Date.now() - 86400000).toISOString(),
    workflowStatus: 'needs_review',
    isAbnormal: false,
    findings: mockFindingsNormal,
    addedToRecord: false
  },
  {
    id: 'r3',
    patientId: null,
    patientName: 'Unknown (Priya S.)',
    reportType: 'Thyroid Profile',
    labName: 'Dr Lal PathLabs',
    receivedAt: new Date(Date.now() - 7200000).toISOString(),
    workflowStatus: 'unmatched',
    isAbnormal: false,
    findings: [
      { id: 'f7', testName: 'TSH', value: '2.5', unit: 'mIU/L', referenceRange: '0.4 - 4.0', status: 'normal' }
    ],
    addedToRecord: false
  },
  {
    id: 'r4',
    patientId: '2',
    patientName: 'Priya Sharma',
    reportType: 'Vitamin D3',
    labName: 'SRL Diagnostics',
    receivedAt: new Date(Date.now() - 172800000).toISOString(),
    workflowStatus: 'reviewed',
    isAbnormal: true,
    findings: [
      { id: 'f8', testName: 'Vitamin D (25-OH)', value: '18', unit: 'ng/mL', referenceRange: '30 - 100', status: 'low' }
    ],
    addedToRecord: true
  },
  {
    id: 'r5',
    patientId: '1',
    patientName: 'Rajesh Kumar',
    reportType: 'Lipid Profile',
    labName: 'Thyrocare',
    receivedAt: new Date(Date.now() - 259200000).toISOString(),
    workflowStatus: 'reviewed',
    isAbnormal: false,
    findings: [
      { id: 'f9', testName: 'Total Cholesterol', value: '180', unit: 'mg/dL', referenceRange: '< 200', status: 'normal' }
    ],
    addedToRecord: true
  }
];

interface ReportsState {
  reports: LabReport[];
  markAsReviewed: (id: string) => void;
  addToPatientRecord: (id: string) => void;
  matchPatient: (reportId: string, patientId: string, patientName: string) => void;
  addMockReport: () => void;
}

export const useReportsStore = create<ReportsState>()(
  persist(
    (set) => ({
      reports: initialReports,
      
      markAsReviewed: (id) => set(state => ({
        reports: state.reports.map(r => r.id === id ? { ...r, workflowStatus: 'reviewed' } : r)
      })),

      addToPatientRecord: (id) => set(state => ({
        reports: state.reports.map(r => r.id === id ? { ...r, addedToRecord: true } : r)
      })),

      matchPatient: (reportId, patientId, patientName) => set(state => ({
        reports: state.reports.map(r => r.id === reportId ? { 
          ...r, 
          patientId, 
          patientName, 
          workflowStatus: r.workflowStatus === 'unmatched' ? 'needs_review' : r.workflowStatus 
        } : r)
      })),

      addMockReport: () => set(state => {
        const newReport: LabReport = {
          id: `r-${Date.now()}`,
          patientId: '1',
          patientName: 'Rajesh Kumar',
          reportType: 'Uploaded Report Demo',
          labName: 'Local Lab',
          receivedAt: new Date().toISOString(),
          workflowStatus: 'needs_review',
          isAbnormal: false,
          findings: mockFindingsNormal,
          addedToRecord: false
        };
        return { reports: [newReport, ...state.reports] };
      })
    }),
    {
      name: 'cpms-reports',
      version: 1,
      // Migration: if stale data has missing fields, fill defaults
      migrate: (persistedState: any, version: number) => {
        if (version === 0 || !persistedState?.reports) {
          return { reports: initialReports };
        }
        // Ensure every report has required fields
        const reports = (persistedState.reports as any[]).map((r: any) => ({
          addedToRecord: false,
          findings: [],
          ...r,
        }));
        return { ...persistedState, reports };
      },
    }
  )
);
