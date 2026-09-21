import React, { useState } from 'react';
import type { Patient, Session, PrescribedMedicine } from '../../types';
import { useSettingsStore } from '../../stores/settingsStore';
import { MedicineRow } from './MedicineRow';
import { MedicineForm } from './MedicineForm';
import { TagPill } from '../../components/ui/TagPill';
import { DemoBadge } from '../../components/ui/DemoBadge';
import { Plus, CheckSquare, X } from 'lucide-react';
import { cn } from '../../utils/cn';

interface PrescriptionDocumentProps {
  patient: Patient;
  session: Session;
  isExtracted: boolean;
  // Local state mapped from page
  medicines: PrescribedMedicine[];
  onAddMedicine: (m: PrescribedMedicine) => void;
  onRemoveMedicine: (id: string) => void;
  tests: string[];
  onAddTest: (test: string) => void;
  onRemoveTest: (test: string) => void;
  advice: string;
  onChangeAdvice: (advice: string) => void;
  followUp: string;
  onChangeFollowUp: (date: string) => void;
  chiefComplaint: string;
  onChangeChiefComplaint: (cc: string) => void;
}

export function PrescriptionDocument({
  patient,
  session,
  isExtracted,
  medicines,
  onAddMedicine,
  onRemoveMedicine,
  tests,
  onAddTest,
  onRemoveTest,
  advice,
  onChangeAdvice,
  followUp,
  onChangeFollowUp,
  chiefComplaint,
  onChangeChiefComplaint
}: PrescriptionDocumentProps) {
  
  const [showForm, setShowForm] = useState(false);
  const [newTest, setNewTest] = useState('');
  const { doctor, clinic } = useSettingsStore();

  const handleAddTest = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && newTest.trim()) {
      onAddTest(newTest.trim());
      setNewTest('');
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md border border-border-subtle max-w-3xl mx-auto flex flex-col text-text-primary">
      
      {/* Clinic Header */}
      <div className="p-4 md:p-8 border-b-2 border-brand-primary-light flex flex-col md:flex-row justify-between items-start gap-4">
        <div>
          <h1 className="text-2xl font-bold text-brand-primary">{doctor.name}</h1>
          <p className="text-sm font-semibold text-text-secondary mt-1">{doctor.qualification}</p>
          <p className="text-sm text-text-tertiary">Reg No: {doctor.regNumber}</p>
        </div>
        <div className="text-left md:text-right w-full md:w-auto mt-2 md:mt-0 pt-2 border-t md:border-t-0 border-border-subtle md:pt-0">
          <h2 className="text-base md:text-lg font-bold text-text-primary">{clinic.name}</h2>
          <p className="text-xs md:text-sm text-text-secondary mt-1 max-w-[200px] leading-relaxed md:ml-auto">
            {clinic.address}
          </p>
          <p className="text-xs md:text-sm font-medium text-text-secondary mt-2">{clinic.phone}</p>
        </div>
      </div>

      {/* Patient Context */}
      <div className="px-4 md:px-8 py-3 md:py-4 border-b border-border-subtle bg-surface-ground/50 flex flex-col sm:flex-row justify-between sm:items-center text-xs md:text-sm gap-2">
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <div>
            <span className="text-text-tertiary font-medium">Patient:</span>
            <span className="font-bold ml-1 md:ml-2 text-sm md:text-base">{patient.name}</span>
          </div>
          <div>
            <span className="text-text-tertiary font-medium">Age/Sex:</span>
            <span className="font-bold ml-1 md:ml-2">{patient.age}y {patient.gender.charAt(0)}</span>
          </div>
        </div>
        <div>
          <span className="text-text-tertiary font-medium">Date:</span>
          <span className="font-bold ml-1 md:ml-2">{session.shortDate}</span>
        </div>
      </div>

      <div className="p-4 md:p-8 space-y-8 md:space-y-10 flex-1">
        
        {/* Chief Complaint / Diagnosis */}
        <section>
          <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-widest mb-3">Diagnosis / Chief Complaint</h3>
          <input 
            type="text" 
            value={chiefComplaint}
            onChange={e => onChangeChiefComplaint(e.target.value)}
            className="w-full text-base font-semibold text-text-primary focus:outline-none border-b border-transparent focus:border-brand-primary pb-1 transition-colors bg-transparent placeholder:text-text-tertiary"
            placeholder="Enter diagnosis or chief complaint..."
          />
        </section>

        {/* Rx Section */}
        <section>
          <div className="flex items-center gap-4 mb-6">
            <span className="text-4xl font-serif italic text-brand-primary opacity-80 select-none">Rx</span>
            <div className="h-px bg-border-subtle flex-1"></div>
          </div>
          
          <div className="space-y-2 mb-4">
            {medicines.map((med, idx) => (
              <MedicineRow 
                key={med.id} 
                index={idx + 1} 
                medicine={med} 
                onRemove={() => onRemoveMedicine(med.id)} 
              />
            ))}
          </div>

          {showForm ? (
            <MedicineForm 
              onAdd={(m) => { onAddMedicine(m); setShowForm(false); }} 
              onCancel={() => setShowForm(false)} 
            />
          ) : (
            <button 
              onClick={() => setShowForm(true)}
              className="flex items-center gap-2 text-brand-primary font-bold hover:bg-brand-primary-light px-4 py-2 rounded-lg transition-colors mt-2"
            >
              <Plus size={18} />
              Add Medicine
            </button>
          )}
        </section>

        {/* Tests */}
        <section>
          <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-widest mb-3 border-b border-border-subtle pb-2">Investigations Advised</h3>
          <div className="flex flex-wrap gap-2 mb-3">
            {tests.map(test => (
              <div key={test} className="flex items-center gap-2 bg-surface-ground border border-border-subtle px-3 py-1.5 rounded-lg text-sm font-medium">
                <CheckSquare size={16} className="text-brand-primary" />
                {test}
                <button onClick={() => onRemoveTest(test)} className="ml-1 text-text-tertiary hover:text-danger"><X size={14}/></button>
              </div>
            ))}
          </div>
          <input 
            type="text" 
            value={newTest}
            onChange={e => setNewTest(e.target.value)}
            onKeyDown={handleAddTest}
            placeholder="Type a test name and press Enter..."
            className="w-full text-sm focus:outline-none border-b border-border-subtle focus:border-brand-primary pb-1 transition-colors bg-transparent placeholder:text-text-tertiary"
          />
        </section>

        {/* Advice */}
        <section>
          <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-widest mb-3 border-b border-border-subtle pb-2">General Advice</h3>
          <textarea 
            value={advice}
            onChange={e => onChangeAdvice(e.target.value)}
            className="w-full text-sm leading-relaxed text-text-primary focus:outline-none bg-transparent placeholder:text-text-tertiary min-h-[80px] resize-y"
            placeholder="Enter general advice, diet, or precautions..."
          />
        </section>

        {/* Follow Up */}
        <section>
          <div className="inline-flex items-center gap-3">
            <h3 className="text-xs font-bold text-text-tertiary uppercase tracking-widest">Follow Up:</h3>
            <input 
              type="text" 
              value={followUp}
              onChange={e => onChangeFollowUp(e.target.value)}
              placeholder="e.g. 5 days, 23rd March"
              className="text-sm font-bold text-text-primary focus:outline-none border-b border-transparent focus:border-brand-primary bg-transparent"
            />
          </div>
        </section>

      </div>

      {/* Footer */}
      <div className="p-4 md:p-8 mt-4 pt-12 md:pt-16 flex flex-col md:flex-row justify-between items-center md:items-end gap-8 bg-surface-ground/30 rounded-b-xl relative">
        {session.prescription?.qrEnabled !== false && (
          <div className="text-center order-2 md:order-1">
            <div className="h-20 w-20 md:h-24 md:w-24 bg-white border border-border-subtle shadow-sm flex items-center justify-center p-2 mb-2 mx-auto">
              <div className="w-full h-full border-2 md:border-4 border-dashed border-border-subtle opacity-50 flex items-center justify-center text-[8px] md:text-[10px] text-text-tertiary font-bold text-center">
                QR Code
              </div>
            </div>
            <DemoBadge label="QR demo" className="mt-1" />
          </div>
        )}
        
        <div className="text-center order-1 md:order-2 w-full md:w-auto">
          <div className="h-12 border-b border-text-tertiary w-40 md:w-48 mb-2 mx-auto md:mx-0"></div>
          <p className="font-bold text-brand-primary">{doctor.name}</p>
          <p className="text-[9px] md:text-[10px] text-text-tertiary uppercase tracking-widest mt-1">Signature</p>
        </div>
      </div>
      
    </div>
  );
}
