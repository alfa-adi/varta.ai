import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { BottomNav } from './BottomNav';
import { cn } from '../../utils/cn';

interface AppShellProps {
  children: React.ReactNode;
  title: string;
}

export function AppShell({ children, title }: AppShellProps) {
  const [isSidebarExpanded, setSidebarExpanded] = useState(false);

  return (
    <div className="h-screen w-full flex bg-transparent overflow-hidden">
      <Sidebar 
        isExpanded={isSidebarExpanded} 
        onToggle={() => setSidebarExpanded(!isSidebarExpanded)} 
      />
      
      <div 
        className={cn(
          "flex-1 flex flex-col transition-all duration-300 w-full",
          isSidebarExpanded ? "md:ml-[240px]" : "md:ml-[72px]",
          "mb-16 md:mb-0" // Add margin bottom for mobile nav
        )}
      >
        <TopBar title={title} />
        
        <main className="flex-1 p-4 md:p-6 overflow-y-auto">
          <div className="max-w-[1080px] mx-auto w-full pb-safe">
            {children}
          </div>
        </main>
      </div>
      <BottomNav />
    </div>
  );
}
