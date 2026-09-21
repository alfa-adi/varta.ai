import type { Patient, DashboardStats, TagItem, Session, Appointment } from '../types';

export const mockTags: Record<string, TagItem[]> = {
  symptoms: [
    { id: 's1', label: 'Fever', variant: 'symptom' },
    { id: 's2', label: 'Cough', variant: 'symptom' },
    { id: 's3', label: 'Headache', variant: 'symptom' },
    { id: 's4', label: 'Nausea', variant: 'symptom' }
  ],
  allergies: [
    { id: 'a1', label: 'Penicillin', variant: 'allergy' },
    { id: 'a2', label: 'Peanuts', variant: 'allergy' },
  ],
  chronic: [
    { id: 'c1', label: 'Hypertension', variant: 'chronic' },
    { id: 'c2', label: 'Type 2 Diabetes', variant: 'chronic' },
  ],
  medicines: [
    { id: 'm1', label: 'Paracetamol 500mg', variant: 'medicine' },
    { id: 'm2', label: 'Amoxicillin 250mg', variant: 'medicine' },
    { id: 'm3', label: 'Metformin 500mg', variant: 'medicine' }
  ],
  tests: [
    { id: 't1', label: 'CBC', variant: 'test' },
    { id: 't2', label: 'HbA1c', variant: 'test' }
  ]
};

export const mockPatients: Patient[] = [
  {
    id: '1',
    patientId: 'PT-2023-001',
    abhaId: 'ABHA-1234-5678-9012',
    name: 'Rajesh Kumar',
    age: 45,
    gender: 'Male',
    bloodGroup: 'O+',
    phone: '+91 98765 43210',
    urgency: 'routine',
    allergies: [mockTags.allergies[0]],
    chronicConditions: [mockTags.chronic[0]],
    lastVisitDate: '2026-06-12',
    livingSummary: 'Patient has a history of hypertension, managed with medication. Recent sessions focus on BP monitoring and mild acute symptoms.',
    sessionCount: 7,
  },
  {
    id: '2',
    patientId: 'PT-2023-002',
    abhaId: 'ABHA-9876-5432-1098',
    name: 'Priya Sharma',
    age: 32,
    gender: 'Female',
    bloodGroup: 'A+',
    phone: '+91 98765 12345',
    urgency: 'attention',
    allergies: [mockTags.allergies[1]],
    chronicConditions: [],
    lastVisitDate: '2026-06-14',
    livingSummary: 'Experiencing severe acute headaches. Requires follow-up imaging if symptoms persist.',
    sessionCount: 3,
  },
  {
    id: '3',
    patientId: 'PT-2023-003',
    name: 'Amit Patel',
    age: 58,
    gender: 'Male',
    bloodGroup: 'B-',
    phone: '+91 91234 56780',
    urgency: 'urgent',
    allergies: [],
    chronicConditions: [mockTags.chronic[0], mockTags.chronic[1]],
    lastVisitDate: '2026-06-15',
    livingSummary: 'Diabetic and hypertensive. Elevated blood sugar levels observed. Needs immediate consultation.',
    sessionCount: 12,
  }
];

export const mockSessions: Record<string, Session[]> = {
  '1': [
    {
      id: 's-1-1',
      sessionNumber: 7,
      date: '12 June 2026',
      shortDate: '12 Jun',
      duration: '14 min',
      type: 'Follow-up',
      mode: 'Translate',
      languagePair: 'Hindi → English',
      chiefComplaint: 'Mild fever and cough for 2 days.',
      symptoms: [mockTags.symptoms[0], mockTags.symptoms[1]],
      medicines: [mockTags.medicines[0]],
      tests: [],
      vitals: [
        { label: 'BP', value: '128/82' },
        { label: 'Temp', value: '99.2°F' },
        { label: 'SpO2', value: '98%' }
      ],
      doctorNotes: 'BP is stable. Mild viral infection suspected. Advised rest and hydration.',
      followUpDate: '19 June 2026',
      reports: [],
      prescription: {
        medicines: [
          {
            id: 'm1',
            name: 'Paracetamol 500mg',
            frequency: '1-0-1',
            duration: '3d',
            instructions: 'After meals'
          }
        ],
        tests: [mockTags.tests[0]],
        advice: 'Rest for 3 days. Drink plenty of fluids.',
        followUpDate: '2024-03-23',
        qrEnabled: true
      },
      transcript: [
        { id: 't1', speaker: 'doctor', text: 'Hello Rajesh, how are you feeling today? Any changes in your BP?', originalText: 'नमस्ते राजेश, आज कैसा लग रहा है?', timestamp: '10:00 AM' },
        { id: 't2', speaker: 'patient', text: 'My BP is fine, but I have had a mild fever and cough since yesterday.', originalText: 'बीपी ठीक है, लेकिन कल से हल्का बुखार और खांसी है।', timestamp: '10:01 AM' },
        { id: 't3', speaker: 'doctor', text: 'Let me check your temperature. It\'s 99.2. I\'ll give you Paracetamol for the fever.', originalText: 'मैं आपका तापमान चेक करता हूँ। 99.2 है।', timestamp: '10:02 AM' }
      ]
    },
    {
      id: 's-1-2',
      sessionNumber: 6,
      date: '15 May 2026',
      shortDate: '15 May',
      duration: '10 min',
      type: 'Routine',
      mode: 'Transcribe',
      chiefComplaint: 'Routine BP checkup.',
      symptoms: [],
      medicines: [],
      tests: [mockTags.tests[0]],
      vitals: [
        { label: 'BP', value: '135/85' },
        { label: 'Weight', value: '78kg' }
      ],
      doctorNotes: 'BP slightly elevated. Advised to reduce sodium intake.',
      reports: [
        { id: 'r1', title: 'Routine Blood Test', date: '14 May 2026', status: 'reviewed', type: 'lab' }
      ],
      transcript: [
        { id: 't1', speaker: 'doctor', text: 'Your reports look mostly fine, but the BP is a bit on the higher side today.', timestamp: '11:15 AM' },
        { id: 't2', speaker: 'patient', text: 'I had some salty food over the weekend, maybe that is why.', timestamp: '11:16 AM' }
      ]
    }
  ]
};

export const mockStats: DashboardStats = {
  seenToday: 14,
  nextTwoHours: 6,
  pendingRx: 3,
  dueThisWeek: 28
};

export const mockAppointments: Appointment[] = [
  {
    id: 'a1',
    patientId: '1',
    patientName: 'Rajesh Kumar',
    date: new Date().toISOString().split('T')[0], // Today
    time: '10:00 AM',
    duration: 30, // minutes
    type: 'Follow-up',
    urgency: 'routine',
    sessionNumber: 8,
  },
  {
    id: 'a2',
    patientId: '2',
    patientName: 'Priya Sharma',
    date: new Date().toISOString().split('T')[0], // Today
    time: '11:30 AM',
    duration: 45,
    type: 'Acute',
    urgency: 'attention',
    sessionNumber: 4,
  },
  {
    id: 'a3',
    patientId: '3',
    patientName: 'Amit Patel',
    date: new Date().toISOString().split('T')[0], // Today
    time: '02:00 PM',
    duration: 30,
    type: 'Follow-up',
    urgency: 'urgent',
    sessionNumber: 13,
  },
  {
    id: 'a4',
    patientId: '1',
    patientName: 'Rajesh Kumar',
    date: new Date(Date.now() + 86400000).toISOString().split('T')[0], // Tomorrow
    time: '09:00 AM',
    duration: 30,
    type: 'Routine',
    urgency: 'routine',
  }
];

// API Fetchers
export const fetchPatients = async (): Promise<Patient[]> => {
  return new Promise((resolve) => setTimeout(() => resolve(mockPatients), 300));
};

export const fetchPatient = async (id: string): Promise<Patient> => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const patient = mockPatients.find(p => p.id === id);
      if (patient) resolve(patient);
      else reject(new Error('Patient not found'));
    }, 200);
  });
};

export const fetchSessions = async (patientId: string): Promise<Session[]> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockSessions[patientId] || []);
    }, 300);
  });
};

export const fetchStats = async (): Promise<DashboardStats> => {
  return new Promise((resolve) => setTimeout(() => resolve(mockStats), 200));
};

export const fetchAppointments = async (): Promise<Appointment[]> => {
  return new Promise((resolve) => setTimeout(() => resolve(mockAppointments), 300));
};
