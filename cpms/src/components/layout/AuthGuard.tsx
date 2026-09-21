import React, { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { Activity } from 'lucide-react';

interface AuthGuardProps {
  children: React.ReactNode;
  allowPublic?: boolean;
}

export function AuthGuard({ children, allowPublic = false }: AuthGuardProps) {
  const { isAuthenticated, isFirstTimeSetup, _hasHydrated } = useAuthStore();
  const location = useLocation();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Wait for client mount and store hydration before making redirect decisions
  if (!mounted || !_hasHydrated) {
    return (
      <div
        role="status"
        aria-label="Loading CPMS"
        className="h-screen w-screen bg-surface-ground flex flex-col items-center justify-center gap-4"
      >
        <div className="flex items-center gap-3 text-brand-primary mb-2">
          <Activity size={28} className="animate-pulse" />
          <span className="text-xl font-bold tracking-tight">CPMS</span>
        </div>
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="h-2 w-2 rounded-full bg-brand-primary animate-pulse"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
        <span className="sr-only">Loading, please wait…</span>
      </div>
    );
  }

  const isLoginRoute = location.pathname === '/login';
  const isSetupRoute = location.pathname === '/setup';

  // Public routes (login / setup)
  if (allowPublic) {
    if (isAuthenticated) {
      if (isFirstTimeSetup && !isSetupRoute) {
        return <Navigate to="/setup" replace />;
      }
      if (!isFirstTimeSetup && (isLoginRoute || isSetupRoute)) {
        return <Navigate to="/" replace />;
      }
    }
    return <>{children}</>;
  }

  // Protected routes
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (isFirstTimeSetup) {
    return <Navigate to="/setup" replace />;
  }

  return <>{children}</>;
}
