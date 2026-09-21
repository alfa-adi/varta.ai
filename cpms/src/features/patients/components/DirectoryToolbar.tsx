import React from 'react';
import { Search, Filter, ArrowUpDown } from 'lucide-react';
import { cn } from '../../../utils/cn';

interface DirectoryToolbarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  filterCount: number;
  onToggleFilters: () => void;
  sortBy: string;
  onSortChange: (sort: string) => void;
}

export function DirectoryToolbar({
  searchQuery,
  onSearchChange,
  filterCount,
  onToggleFilters,
  sortBy,
  onSortChange
}: DirectoryToolbarProps) {
  return (
    <div className="flex flex-col md:flex-row gap-4 items-center justify-between bg-surface-card p-4 rounded-xl border border-border-subtle shadow-sm mb-6">
      
      {/* Search Bar */}
      <div className="relative w-full md:w-[320px] lg:w-[400px]">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" size={18} />
        <input 
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search name, phone, ABHA, condition..."
          className="w-full h-10 pl-10 pr-4 bg-surface-ground border border-border-subtle rounded-lg text-sm focus:outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary transition-all"
        />
      </div>

      <div className="flex items-center gap-3 w-full md:w-auto">
        {/* Filter Toggle */}
        <button 
          onClick={onToggleFilters}
          className={cn(
            "flex-1 md:flex-none flex items-center justify-center gap-2 h-10 px-4 rounded-lg text-sm font-semibold border transition-all",
            filterCount > 0 
              ? "bg-brand-primary-light border-brand-primary text-brand-primary" 
              : "bg-surface-ground border-border-subtle text-text-secondary hover:text-text-primary hover:border-brand-primary/30"
          )}
        >
          <Filter size={16} />
          Filters
          {filterCount > 0 && (
            <span className="w-5 h-5 rounded-full bg-brand-primary text-text-on-brand text-[10px] flex items-center justify-center font-bold">
              {filterCount}
            </span>
          )}
        </button>

        {/* Sort Dropdown */}
        <div className="relative flex-1 md:flex-none">
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none">
            <ArrowUpDown size={14} />
          </div>
          <select 
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value)}
            className="w-full h-10 pl-9 pr-8 bg-surface-ground border border-border-subtle rounded-lg text-sm font-medium text-text-secondary focus:outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary transition-all appearance-none cursor-pointer"
          >
            <option value="recent">Recently seen</option>
            <option value="name_asc">Name A–Z</option>
            <option value="oldest">Last visit oldest</option>
            <option value="urgency">Urgency</option>
          </select>
        </div>
      </div>
    </div>
  );
}
