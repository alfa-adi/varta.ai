# Frontend Data Fetching (Getpoints) Schema

Based on an analysis of the frontend codebase (specifically src/app/data/mockData.ts), the UI currently relies on static mock data for its models. To make this frontend fully dynamic, it will require a backend with the following REST API endpoints ("getpoints") to manage its data structures.

## 1. Patients API

Endpoints related to managing patient records, profiles, and their medical history.

| Endpoint | Method | Expected Query Params | Expected Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| /api/patients | GET | ?search=str, ?limit=num | Patient[] | Fetch a list of patients (used for Global Search and Dashboard). |
| /api/patients/{id} | GET | None | Patient | Fetch a full patient profile, including their chronicConditions, allergies, and livingSummary. |
| /api/patients | POST | None | Patient | Create a new patient record (used by the NewPatientForm component). |

### Patient Model Schema Requirements
- id, patientId, abhaId, name, age, gender, bloodGroup, phone, photo, urgency
- allergies, chronicConditions (Array of TagItem)
- lastVisitDate, livingSummary, emergencyContact

---

## 2. Sessions & Clinical Notes API

Endpoints related to recording and retrieving specific doctor-patient encounters, transcripts, and prescriptions.

| Endpoint | Method | Expected Query Params | Expected Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| /api/patients/{id}/sessions | GET | ?limit=num | Session[] | Fetch history of clinical sessions for a specific patient. |
| /api/sessions/{id} | GET | None | Session | Fetch detailed session data, including the transcript and prescription. |
| /api/sessions | POST | None | Session | Save a newly completed session (triggered after the RecordingScreen finishes). |

### Session Model Schema Requirements
- id, sessionNumber, date, shortDate, duration, type, mode (Transcribe / Translate)
- chiefComplaint, doctorNotes, followUpDate
- symptoms, medicines, tests (Array of TagItem)
- vitals (Array of { label, value })
- transcript (Array of TranscriptLine with timestamps and translations)
- prescription (Object containing medicines, tests, advice, followUp)
- reports (Array of Report items)

---

## 3. Appointments & Calendar API

Endpoints to drive the Dashboard scheduling and Calendar screens.

| Endpoint | Method | Expected Query Params | Expected Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| /api/appointments | GET | ?date=YYYY-MM-DD | Appointment[] | Fetch scheduled appointments for a specific day or date range. |

### Appointment Model Schema Requirements
- id, patientId, patientName
- date, time, duration (in minutes)
- type, urgency (routine / attention / urgent)
- sessionNumber

---

## 4. Doctor Profile & Clinic Config API

Endpoints to fetch the authenticated doctor's profile and clinic letterhead details (used for generating prescriptions).

| Endpoint | Method | Expected Query Params | Expected Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| /api/doctor/profile | GET | None | DoctorProfile | Fetch doctor and clinic details for the authenticated user. |

### DoctorProfile Model Schema Requirements
- name, firstName, qualification, regNumber, specialisation
- clinic: { name, address, phone, website, gstin }

---

## 5. Dashboard & Analytics API

Endpoints strictly for aggregating metrics displayed on the Dashboard (e.g., "Seen Today", "Pending Rx").

| Endpoint | Method | Expected Query Params | Expected Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| /api/dashboard/stats | GET | None | DashboardStats | Fetch aggregate counts for the top dashboard cards. |

### DashboardStats Model Schema Requirements
- seenToday (number)
- nextTwoHours (number)
- pendingRx (number)
- dueThisWeek (number)

---

> [!TIP]
> *Next Steps for Backend Integration:*
> Currently, the frontend components (like App.tsx and DashboardScreen) import static data synchronously from mockData.ts. To integrate these APIs, you will need to:
> 1. Set up a data fetching library like *React Query (useQuery), **SWR*, or standard useEffect hooks.
> 2. Replace the synchronous states (e.g. mockPatients.find) with async API calls targeting the endpoints outlined above.