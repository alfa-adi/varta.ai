import React, { useState } from 'react';
import type { PrescribedMedicine } from '../../types';
import { Plus, X } from 'lucide-react';
import { cn } from '../../utils/cn';

interface MedicineFormProps {
  onAdd: (med: PrescribedMedicine) => void;
  onCancel: () => void;
}

const FREQUENCIES = ['1-0-0', '0-0-1', '1-0-1', '1-1-1', 'SOS'];
const DURATIONS = ['3d', '5d', '7d', '14d', '30d'];

export function MedicineForm({ onAdd, onCancel }: MedicineFormProps) {
  const [name, setName] = useState('');
  const [frequency, setFrequency] = useState('');
  const [duration, setDuration] = useState('');
  const [instructions, setInstructions] = useState('');

  const handleAdd = () => {
    if (!name || !frequency || !duration) return;
    
    onAdd({
      id: Math.random().toString(36).substr(2, 9),
      name,
      frequency,
      duration,
      instructions
    });
  };

  const isComplete = name && frequency && duration;

  return (
    <div className="bg-brand-primary-xlt border border-brand-primary/20 rounded-xl p-5 mb-4 animate-fade-in shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h4 className="font-bold text-brand-primary">Add Medicine</h4>
        <button onClick={onCancel} className="text-text-tertiary hover:text-text-primary transition-colors">
          <X size={20} />
        </button>
      </div>

      <div className="space-y-4">
        {/* Name input */}
        <div>
          <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Drug Name & Strength</label>
          <input 
            type="text" 
            placeholder="e.g. Paracetamol 500mg"
            className="w-full p-3 rounded-lg border border-border-subtle bg-surface-card text-text-primary focus:outline-none focus:ring-2 focus:ring-brand-primary focus:border-brand-primary shadow-sm"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {/* Frequency */}
        <div>
          <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Frequency</label>
          <div className="flex flex-wrap gap-2">
            {FREQUENCIES.map(f => (
              <button
                key={f}
                onClick={() => setFrequency(f)}
                className={cn(
                  "px-3 py-1.5 rounded-full text-sm font-semibold border transition-colors shadow-sm",
                  frequency === f 
                    ? "bg-brand-primary text-text-on-brand border-brand-primary" 
                    : "bg-surface-card text-text-secondary border-border-subtle hover:border-brand-primary"
                )}
              >
                {f}
              </button>
            ))}
            <input 
              type="text" 
              placeholder="Custom"
              className={cn(
                "px-3 py-1.5 rounded-full text-sm border focus:outline-none focus:border-brand-primary shadow-sm",
                !FREQUENCIES.includes(frequency) && frequency ? "bg-brand-primary text-text-on-brand border-brand-primary placeholder:text-text-on-brand/70" : "bg-surface-card text-text-primary border-border-subtle"
              )}
              value={!FREQUENCIES.includes(frequency) ? frequency : ''}
              onChange={(e) => setFrequency(e.target.value)}
            />
          </div>
        </div>

        {/* Duration */}
        <div>
          <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Duration</label>
          <div className="flex flex-wrap gap-2">
            {DURATIONS.map(d => (
              <button
                key={d}
                onClick={() => setDuration(d)}
                className={cn(
                  "px-3 py-1.5 rounded-full text-sm font-semibold border transition-colors shadow-sm",
                  duration === d 
                    ? "bg-brand-primary text-text-on-brand border-brand-primary" 
                    : "bg-surface-card text-text-secondary border-border-subtle hover:border-brand-primary"
                )}
              >
                {d}
              </button>
            ))}
            <input 
              type="text" 
              placeholder="Custom"
              className={cn(
                "px-3 py-1.5 rounded-full text-sm border focus:outline-none focus:border-brand-primary shadow-sm",
                !DURATIONS.includes(duration) && duration ? "bg-brand-primary text-text-on-brand border-brand-primary placeholder:text-text-on-brand/70" : "bg-surface-card text-text-primary border-border-subtle"
              )}
              value={!DURATIONS.includes(duration) ? duration : ''}
              onChange={(e) => setDuration(e.target.value)}
            />
          </div>
        </div>

        {/* Instructions */}
        <div>
          <label className="block text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Instructions (Optional)</label>
          <input 
            type="text" 
            placeholder="e.g. After meals"
            className="w-full p-3 rounded-lg border border-border-subtle bg-surface-card text-text-primary focus:outline-none focus:ring-2 focus:ring-brand-primary focus:border-brand-primary shadow-sm"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
          />
        </div>
      </div>

      <div className="mt-6 flex justify-end">
        <button 
          onClick={handleAdd}
          disabled={!isComplete}
          className="flex items-center gap-2 px-6 py-2.5 bg-brand-primary text-text-on-brand font-bold rounded-lg shadow-sm hover:bg-brand-primary-mid transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Plus size={18} />
          Add to prescription
        </button>
      </div>
    </div>
  );
}
