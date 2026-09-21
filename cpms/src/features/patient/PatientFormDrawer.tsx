import React, { useState, useEffect, useRef } from 'react';
import { X, User, Phone, MapPin, Search, AlertCircle, CheckCircle2, ChevronRight, Activity, Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { usePatientFormStore } from '../../stores/patientFormStore';
import { usePatientDataStore } from '../../stores/patientDataStore';
import { useToastStore } from '../../stores/toastStore';
import { DemoBadge } from '../../components/ui/DemoBadge';
import type { Patient, TagItem, Urgency } from '../../types';
import { cn } from '../../utils/cn';

// Minimal Custom Discard Dialog
function DiscardDialog({ onConfirm, onCancel }: { onConfirm: () => void, onCancel: () => void }) {
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in px-4">
      <div className="bg-surface-card w-full max-w-sm rounded-xl shadow-xl p-6 border border-border-subtle">
        <h3 className="text-lg font-bold text-text-primary mb-2">Discard unsaved changes?</h3>
        <p className="text-sm text-text-secondary mb-6">If you leave now, you'll lose the patient details you've entered.</p>
        <div className="flex justify-end gap-3">
          <button 
            onClick={onCancel}
            className="px-4 py-2 text-sm font-semibold text-text-secondary hover:bg-surface-ground rounded-lg transition-colors"
          >
            Keep editing
          </button>
          <button 
            onClick={onConfirm}
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            className="px-4 py-2 text-sm font-semibold text-white bg-danger hover:bg-danger/90 rounded-lg shadow-sm transition-colors"
          >
            Discard changes
          </button>
        </div>
      </div>
    </div>
  );
}

export function PatientFormDrawer() {
  const isOpen = usePatientFormStore(state => state.isOpen);
  const closeForm = usePatientFormStore(state => state.closeForm);
  const { addPatient, findDuplicate } = usePatientDataStore();
  const addToast = useToastStore(state => state.addToast);
  const navigate = useNavigate();
  
  const drawerRef = useRef<HTMLDivElement>(null);
  const initialFocusRef = useRef<HTMLInputElement>(null);

  // Form State
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<Patient['gender'] | 'Prefer not to say' | ''>('');
  const [phone, setPhone] = useState('');
  const [abhaId, setAbhaId] = useState('');
  const [address, setAddress] = useState('');
  const [bloodGroup, setBloodGroup] = useState('');
  const [allergies, setAllergies] = useState<string>('');
  const [chronic, setChronic] = useState<string>('');
  const [urgency, setUrgency] = useState<Urgency>('routine');
  const [language, setLanguage] = useState('English');
  const [translationEnabled, setTranslationEnabled] = useState(false);
  const [doctorLanguage, setDoctorLanguage] = useState('English');

  // UI State
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [duplicateWarning, setDuplicateWarning] = useState<Patient | null>(null);
  const [showDiscard, setShowDiscard] = useState(false);

  const isDirty = name || age || gender || phone || abhaId || address;

  // Reset form when opened
  useEffect(() => {
    if (isOpen) {
      setName('');
      setAge('');
      setGender('');
      setPhone('');
      setAbhaId('');
      setAddress('');
      setBloodGroup('');
      setAllergies('');
      setChronic('');
      setUrgency('routine');
      setLanguage('English');
      setTranslationEnabled(false);
      setTouched({});
      setDuplicateWarning(null);
      setShowDiscard(false);
      
      document.body.style.overflow = 'hidden';
      setTimeout(() => initialFocusRef.current?.focus(), 100);
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // Handle Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !showDiscard) {
        handleCloseAttempt();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, showDiscard, isDirty]);

  const handleCloseAttempt = () => {
    if (isDirty) {
      setShowDiscard(true);
    } else {
      closeForm();
    }
  };

  const handleBlur = (field: string) => {
    setTouched(prev => ({ ...prev, [field]: true }));
    if (field === 'phone' || field === 'abhaId') {
      checkForDuplicates();
    }
  };

  const checkForDuplicates = () => {
    if (phone.length >= 10 || abhaId.length > 5) {
      const dup = findDuplicate(phone, abhaId);
      setDuplicateWarning(dup || null);
    } else {
      setDuplicateWarning(null);
    }
  };

  const validatePhone = (p: string) => {
    const digits = p.replace(/\D/g, '');
    return digits.length === 10;
  };

  const handleSave = () => {
    setTouched({ name: true, age: true, gender: true, phone: true });
    
    if (!name || !age || !gender || !validatePhone(phone) || duplicateWarning) {
      return; // Invalid
    }

    // Convert comma separated strings to TagItems
    const parsedAllergies: TagItem[] = allergies.split(',').map(s => s.trim()).filter(Boolean).map(label => ({
      id: Math.random().toString(), label, variant: 'allergy'
    }));
    const parsedChronic: TagItem[] = chronic.split(',').map(s => s.trim()).filter(Boolean).map(label => ({
      id: Math.random().toString(), label, variant: 'chronic'
    }));

    const newPatient: Patient = {
      id: Math.random().toString(36).substr(2, 9),
      patientId: `PT-${new Date().getFullYear()}-${Math.floor(Math.random() * 1000)}`,
      name,
      age: parseInt(age, 10),
      gender: gender as any, // fallback
      phone: `+91 ${phone.replace(/\D/g, '').slice(-10)}`, // Standardize
      abhaId: abhaId || undefined,
      bloodGroup: bloodGroup || undefined,
      urgency,
      allergies: parsedAllergies,
      chronicConditions: parsedChronic,
      lastVisitDate: new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase(),
      livingSummary: 'New patient registered today.',
    };

    addPatient(newPatient);
    closeForm();
    addToast({ type: 'success', message: `Patient "${newPatient.name}" added successfully.` });
    navigate(`/patients/${newPatient.id}`);
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Add New Patient"
    >
      <div 
        className="absolute inset-0" 
        onClick={handleCloseAttempt}
      />
      
      <div 
        ref={drawerRef}
        className="relative w-full md:w-[560px] h-full bg-surface-card md:rounded-l-2xl shadow-2xl flex flex-col animate-slide-up md:animate-slide-in-right z-10 pb-safe"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-subtle shrink-0">
          <div>
            <h2 className="text-xl font-bold text-text-primary tracking-tight">Add Patient</h2>
            <p className="text-sm text-text-secondary">Create a patient profile</p>
          </div>
          <button 
            onClick={handleCloseAttempt}
            className="p-2 text-text-tertiary hover:bg-surface-ground rounded-full transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Scrollable Form Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {/* Avatar Section */}
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-brand-primary-xlt text-brand-primary flex items-center justify-center font-bold text-xl border border-brand-primary/20">
              {name ? name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() : <User size={28} />}
            </div>
            <div>
              <button className="text-sm font-semibold text-brand-primary hover:text-brand-primary-mid px-3 py-1.5 bg-brand-primary-xlt rounded-lg transition-colors">
                Upload Photo
              </button>
            </div>
          </div>

          {/* Section 1: Basic Details */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-wider">Basic Details</h3>
            
            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Full Name <span className="text-danger">*</span></label>
              <input 
                ref={initialFocusRef}
                type="text" 
                value={name}
                onChange={e => setName(e.target.value)}
                onBlur={() => handleBlur('name')}
                placeholder="e.g. Priya Sharma"
                className={cn("w-full h-11 px-3 bg-white border rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all", 
                  touched.name && !name ? "border-danger bg-danger-xlt" : "border-border-subtle focus:border-brand-primary"
                )}
              />
              {touched.name && !name && <p className="text-xs text-danger font-medium mt-1">Name is required</p>}
            </div>

            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Age / DOB <span className="text-danger">*</span></label>
              <input 
                type="text" 
                value={age}
                onChange={e => setAge(e.target.value)}
                onBlur={() => handleBlur('age')}
                placeholder="e.g. 32"
                className={cn("w-full h-11 px-3 bg-white border rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all",
                  touched.age && !age ? "border-danger bg-danger-xlt" : "border-border-subtle focus:border-brand-primary"
                )}
              />
            </div>

            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Gender <span className="text-danger">*</span></label>
              <div
                role="group"
                aria-label="Gender"
                className={cn(
                  "grid grid-cols-4 gap-1 p-1 rounded-xl border transition-all",
                  touched.gender && !gender
                    ? "border-danger bg-danger-xlt"
                    : "border-border-subtle bg-surface-ground"
                )}
              >
                {(['Female', 'Male', 'Other', 'Prefer not to say'] as const).map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    role="radio"
                    aria-checked={gender === opt}
                    onClick={() => { setGender(opt as any); handleBlur('gender'); }}
                    className={cn(
                      "py-2 px-1 rounded-lg text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-brand-primary-light truncate",
                      gender === opt
                        ? "bg-brand-primary-xlt text-brand-primary border border-brand-primary/30 shadow-sm"
                        : "text-text-secondary hover:bg-surface-card hover:text-text-primary"
                    )}
                  >
                    {opt}
                  </button>
                ))}
              </div>
              {touched.gender && !gender && <p className="text-xs text-danger font-medium mt-1">Gender is required</p>}
            </div>

            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Mobile Number <span className="text-danger">*</span></label>
              <div className="flex">
                <div className="h-11 px-3 bg-surface-ground border border-border-subtle border-r-0 rounded-l-lg flex items-center text-text-secondary font-medium shrink-0">
                  +91
                </div>
                <input 
                  type="tel" 
                  value={phone}
                  onChange={e => {
                    const val = e.target.value.replace(/\D/g, '').slice(0, 10);
                    setPhone(val);
                  }}
                  onBlur={() => handleBlur('phone')}
                  placeholder="98765 43210"
                  className={cn("w-full h-11 px-3 bg-white border rounded-r-lg focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all",
                    touched.phone && !validatePhone(phone) ? "border-danger bg-danger-xlt" : "border-border-subtle focus:border-brand-primary"
                  )}
                />
              </div>
              {touched.phone && !validatePhone(phone) && <p className="text-xs text-danger font-medium mt-1">Valid 10-digit number required</p>}
            </div>
            
            {/* Duplicate Warning */}
            {duplicateWarning && (
              <div className="p-4 bg-warning-xlt border border-warning/30 rounded-lg flex items-start gap-3 mt-2 animate-fade-in">
                <AlertCircle className="text-warning mt-0.5 shrink-0" size={18} />
                <div className="flex-1">
                  <p className="text-sm font-bold text-text-primary mb-1">Patient already exists</p>
                  <p className="text-xs text-text-secondary mb-3">A profile with this phone or ABHA ID is already registered to {duplicateWarning.name}.</p>
                  <button 
                    onClick={() => { closeForm(); navigate(`/patients/${duplicateWarning.id}`); }}
                    className="text-sm font-semibold text-brand-primary hover:text-brand-primary-mid flex items-center gap-1"
                  >
                    View existing patient <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            )}
          </section>

          {/* Section 2: Identity */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-wider">Identity (Optional)</h3>
            
            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">ABHA Number</label>
              <input 
                type="text" 
                value={abhaId}
                onChange={e => setAbhaId(e.target.value)}
                onBlur={() => handleBlur('abhaId')}
                placeholder="14-digit ABHA ID"
                className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all"
              />
              <p className="text-xs text-text-tertiary mt-1">Allows fetching past medical records.</p>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Address</label>
              <textarea 
                value={address}
                onChange={e => setAddress(e.target.value)}
                placeholder="Residential address"
                rows={2}
                className="w-full px-3 py-2 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all resize-none"
              />
            </div>
          </section>

          {/* Section 3: Clinical Snapshot */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-wider">Clinical Snapshot (Optional)</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-sm font-semibold text-text-secondary">Blood Group</label>
                <select 
                  value={bloodGroup}
                  onChange={e => setBloodGroup(e.target.value)}
                  className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all appearance-none"
                >
                  <option value="">Unknown</option>
                  <option value="A+">A+</option>
                  <option value="A-">A-</option>
                  <option value="B+">B+</option>
                  <option value="B-">B-</option>
                  <option value="O+">O+</option>
                  <option value="O-">O-</option>
                  <option value="AB+">AB+</option>
                  <option value="AB-">AB-</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-sm font-semibold text-text-secondary">Urgency Level</label>
                <select 
                  value={urgency}
                  onChange={e => setUrgency(e.target.value as Urgency)}
                  className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all appearance-none"
                >
                  <option value="routine">Routine</option>
                  <option value="attention">Attention</option>
                  <option value="urgent">Urgent</option>
                </select>
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Allergies (comma separated)</label>
              <input 
                type="text" 
                value={allergies}
                onChange={e => setAllergies(e.target.value)}
                placeholder="e.g. Penicillin, Peanuts, No known allergies"
                className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all"
              />
            </div>
            
            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Chronic Conditions (comma separated)</label>
              <input 
                type="text" 
                value={chronic}
                onChange={e => setChronic(e.target.value)}
                placeholder="e.g. Hypertension, Type 2 Diabetes"
                className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all"
              />
            </div>
          </section>

          {/* Section 4: Communication */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-wider">Communication</h3>
            
            <div className="space-y-1">
              <label className="text-sm font-semibold text-text-secondary">Patient's Spoken Language</label>
              <select 
                value={language}
                onChange={e => setLanguage(e.target.value)}
                className="w-full h-11 px-3 bg-white border border-border-subtle rounded-lg focus:border-brand-primary focus:outline-none focus:ring-2 focus:ring-brand-primary-light transition-all appearance-none"
              >
                <option value="English">English</option>
                <option value="Hindi">Hindi</option>
                <option value="Marathi">Marathi</option>
                <option value="Gujarati">Gujarati</option>
                <option value="Tamil">Tamil</option>
                <option value="Telugu">Telugu</option>
                <option value="Kannada">Kannada</option>
                <option value="Bengali">Bengali</option>
                <option value="Malayalam">Malayalam</option>
                <option value="Punjabi">Punjabi</option>
              </select>
            </div>

            {language !== doctorLanguage && (
              <div className="p-4 bg-brand-primary-xlt border border-brand-primary/20 rounded-lg mt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-brand-primary">Enable Translation</span>
                  <div 
                    className={cn("w-10 h-6 rounded-full p-1 cursor-pointer transition-colors", translationEnabled ? "bg-brand-primary" : "bg-border-strong")}
                    onClick={() => setTranslationEnabled(!translationEnabled)}
                  >
                    <div className={cn("w-4 h-4 bg-white rounded-full shadow transition-transform", translationEnabled ? "translate-x-4" : "translate-x-0")} />
                  </div>
                </div>
                {translationEnabled && (
                  <p className="text-xs text-brand-primary/80 font-medium">Transcripts will be translated to English for you.</p>
                )}
              </div>
            )}
          </section>

          {/* Bottom Padding */}
          <div className="h-8" />
        </div>

        {/* Sticky Footer */}
        <div className="p-4 border-t border-border-subtle bg-surface-ground flex justify-end gap-3 shrink-0 rounded-bl-2xl">
          <button 
            onClick={handleCloseAttempt}
            className="px-6 py-2.5 text-sm font-semibold text-text-secondary hover:bg-surface-card hover:text-text-primary rounded-full transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={handleSave}
            disabled={!!duplicateWarning}
            className="px-6 py-2.5 text-sm font-bold text-white bg-brand-primary hover:bg-brand-primary-hover rounded-full shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            Save Patient
          </button>
        </div>
        
        {/* Custom Discard Dialog */}
        {showDiscard && (
          <DiscardDialog 
            onConfirm={closeForm} 
            onCancel={() => setShowDiscard(false)} 
          />
        )}
      </div>
    </div>
  );
}
