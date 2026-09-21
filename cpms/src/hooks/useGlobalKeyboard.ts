import { useEffect } from 'react';
import { useSearchStore } from '../stores/searchStore';

export function useGlobalKeyboard() {
  const toggleSearch = useSearchStore((state) => state.toggleSearch);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        // Check if the user is typing in a real input, if so, we usually still override for global search,
        // but the user requested: "The Ctrl/Cmd+K handler must ignore typing contexts (input, textarea, select, and contenteditable) 
        // and only prevent the browser default when it opens CPMS search."
        // Wait, if they ARE in an input, should we ignore it? Yes, the prompt says "must ignore typing contexts".
        const activeElement = document.activeElement as HTMLElement;
        const isInput = activeElement && (
          activeElement.tagName === 'INPUT' ||
          activeElement.tagName === 'TEXTAREA' ||
          activeElement.tagName === 'SELECT' ||
          activeElement.isContentEditable
        );

        if (isInput) {
          return; // Ignore and let default browser behavior or input behavior happen
        }

        e.preventDefault();
        toggleSearch();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleSearch]);
}
