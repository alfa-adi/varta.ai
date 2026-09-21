import { create } from 'zustand';

interface SearchState {
  isOpen: boolean;
  query: string;
  openSearch: () => void;
  closeSearch: () => void;
  toggleSearch: () => void;
  setQuery: (query: string) => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  isOpen: false,
  query: '',
  openSearch: () => set({ isOpen: true }),
  closeSearch: () => set({ isOpen: false, query: '' }),
  toggleSearch: () => set((state) => ({ isOpen: !state.isOpen, query: state.isOpen ? '' : state.query })),
  setQuery: (query) => set({ query }),
}));
