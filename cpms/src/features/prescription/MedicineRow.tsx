import React from 'react';
import type { PrescribedMedicine } from '../../types';
import { X } from 'lucide-react';

interface MedicineRowProps {
  index: number;
  medicine: PrescribedMedicine;
  onRemove: () => void;
}

export function MedicineRow({ index, medicine, onRemove }: MedicineRowProps) {
  return (
    <div className="flex items-start gap-4 p-3 rounded-lg hover:bg-surface-ground transition-colors group">
      <span className="font-bold text-text-tertiary w-6">{index}.</span>
      
      <div className="flex-1">
        <h4 className="font-bold text-text-primary text-base">{medicine.name}</h4>
        <div className="flex items-center gap-2 mt-1 text-sm font-medium">
          <span className="bg-brand-primary-light text-brand-primary px-2 py-0.5 rounded uppercase tracking-wider">
            {medicine.frequency}
          </span>
          <span className="text-text-tertiary">•</span>
          <span className="text-text-secondary">{medicine.duration}</span>
        </div>
        {medicine.instructions && (
          <p className="text-sm text-text-tertiary mt-1 italic">
            Instructions: {medicine.instructions}
          </p>
        )}
      </div>

      <button 
        onClick={onRemove}
        className="text-text-tertiary hover:text-danger hover:bg-danger-light p-2 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
        title="Remove"
      >
        <X size={18} />
      </button>
    </div>
  );
}
