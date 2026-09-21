# CPMS Demo Script

**Target Audience:** Doctors, Clinic Owners, Investors.
**Goal:** Prove that CPMS eliminates documentation overhead while improving record quality and follow-up care.
**Duration:** ~6 minutes.

## The Pitch (30 Seconds)
"Doctors today spend as much time typing notes as they do talking to patients. It leads to burnout and fractured attention. CPMS changes that. Our system listens to your consultation, structures the medical data in real-time, and generates a complete, print-ready prescription before the patient even stands up to leave. Document every word, forget none of it, and focus entirely on the patient."

---

## 1. Introduction & Login (1 min)
*   **Action:** Open `/demo`, click "Start Guided Demo". Wait for it to route to `/login`.
*   **Script:** "Welcome to CPMS. I'm going to walk you through a typical day. We'll start by logging into our configured clinic."
*   **Action:** Enter `000000` as the OTP to show the error.
*   **Script:** "Security is strict. Invalid attempts are blocked. Let's use our valid access code."
*   **Action:** Enter a valid OTP (e.g., `123456`) and wait for the Dashboard to load.

## 2. The Dashboard (1 min)
*   **Action:** Once on the Dashboard, point out the metrics.
*   **Script:** "This is the doctor's command center. You immediately see who is waiting, what lab reports need your review, and upcoming follow-ups. Everything is actionable right from here."
*   **Action:** Press `Ctrl+K` or click the search bar. Type "Rajesh".
*   **Script:** "Global search lets you pull up a patient file in milliseconds, whether they are calling in or walking through the door."
*   **Action:** Click on Rajesh's search result to open his profile.

## 3. The Consultation & AI Capture (1.5 mins)
*   **Action:** In Rajesh's profile, scroll through the timeline.
*   **Script:** "Here we have Rajesh's complete longitudinal history. Let's start today's session."
*   **Action:** Click the central `+` FAB and select `New Session`.
*   **Script:** "This is where the magic happens. The system is now securely recording our conversation."
*   **Action:** Speak a brief mock consultation (e.g., "Rajesh, you've had a fever for three days. I'm prescribing Paracetamol 500mg twice a day for 5 days. Let's also do a blood test to check for Dengue.").
*   **Action:** Click `End Session`.
*   **Script:** "When the session ends, our AI immediately processes the transcript, structuring symptoms, vitals, and extracting the prescribed medicines and tests. No manual typing required."
*   **Action:** Click `Save & Issue Prescription`.

## 4. The Prescription (1 min)
*   **Action:** Show the generated prescription screen.
*   **Script:** "Before the patient has even stood up, their prescription is ready. It's professionally formatted with the clinic's identity, includes all the structured medicines we just discussed, and even features a QR code for the patient to easily refill their meds or book their next appointment."

## 5. Lab Reports & Follow-ups (1 min)
*   **Action:** Click the "More" tab, then click "Reports".
*   **Script:** "But patient care doesn't end when they leave the clinic. Let's look at the Reports Inbox."
*   **Action:** Click on an abnormal report (e.g., Fasting Blood Sugar).
*   **Script:** "Labs flow directly into CPMS. The system automatically flags abnormal findings in red, so you know exactly what needs urgent attention."
*   **Action:** Click "Create Follow-up", set a date, and confirm.
*   **Script:** "With one click, we can schedule a follow-up appointment. This ensures continuity of care and prevents patients from falling through the cracks."
*   **Action:** Navigate to the Calendar tab to show the new appointment.

## 6. Configurable Identity & Conclusion (30s)
*   **Action:** Go to "Settings".
*   **Script:** "Finally, everything you've seen is fully customizable. From the doctor's registration number to the clinic's branding and AI defaults, CPMS adapts to your practice—not the other way around."
*   **Action:** Click "Next" on the demo controller to finish the tour and return to the landing page.
*   **Script:** "Faster documentation, structured records, and better follow-up. That's CPMS."
