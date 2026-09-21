import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { useToastStore } from '../../stores/toastStore';
import { Activity, Globe, BrainCircuit, CheckCircle2, FlaskConical } from 'lucide-react';
import { cn } from '../../utils/cn';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isFirstTimeSetup } = useAuthStore();
  const addToast = useToastStore(state => state.addToast);
  
  const [step, setStep] = useState<'phone' | 'otp'>('phone');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [timeLeft, setTimeLeft] = useState(30);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (step === 'otp' && timeLeft > 0) {
      timer = setTimeout(() => setTimeLeft(prev => prev - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [step, timeLeft]);

  const handlePhoneSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (phone.replace(/\D/g, '').length !== 10) {
      setError('Please enter a valid 10-digit number');
      return;
    }
    setError('');
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setStep('otp');
      setTimeLeft(30);
    }, 1000);
  };

  const handleOtpChange = (index: number, value: string) => {
    if (!/^\d*$/.test(value)) return;
    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-advance
    if (value && index < 5) {
      const nextInput = document.getElementById(`otp-${index + 1}`);
      if (nextInput) nextInput.focus();
    }
  };

  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      const prevInput = document.getElementById(`otp-${index - 1}`);
      if (prevInput) prevInput.focus();
    }
  };

  const handleOtpSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const otpValue = otp.join('');
    if (otpValue.length !== 6) {
      setError('Please enter all 6 digits');
      return;
    }
    setError('');
    setLoading(true);
    
    setTimeout(() => {
      setLoading(false);
      // For demo: '000000' triggers invalid OTP error
      if (otpValue === '000000') {
        setError('Invalid OTP code. Please try again.');
        setOtp(['', '', '', '', '', '']);
        document.getElementById('otp-0')?.focus();
        return;
      }
      
      // Login — isFirstTimeSetup is preserved from the persisted store.
      // Returning users who completed setup will skip /setup automatically.
      login('+91 ' + phone);
      navigate(isFirstTimeSetup ? '/setup' : '/', { replace: true });
    }, 1200);
  };

  return (
    <div className="flex h-screen w-full bg-surface-ground overflow-hidden">
      {/* Left Panel: Branding */}
      <div className="hidden md:flex w-[45%] bg-brand-primary flex-col justify-between p-12 text-white relative overflow-hidden">
        {/* Topographic subtle background lines */}
        <div className="absolute inset-0 opacity-5 pointer-events-none" 
             style={{ backgroundImage: 'radial-gradient(circle at 50% 50%, #ffffff 1px, transparent 1px)', backgroundSize: '24px 24px' }} 
        />
        
        <div className="relative z-10">
          <div className="flex items-center gap-3 text-white mb-16">
            <div className="bg-white/20 p-2 rounded-xl backdrop-blur-md shadow-sm border border-white/30">
              <Activity className="text-white" size={24} />
            </div>
            <span className="font-bold text-2xl tracking-tight">CPMS</span>
          </div>

          <h1 className="text-5xl font-bold leading-[1.1] mb-6">
            Document every word.<br/>
            <span className="text-white/70">Forget none of it.</span>
          </h1>
          <p className="text-lg text-white/80 max-w-sm mb-16">
            The intelligent clinic instrument that transforms conversations into structured, reliable records.
          </p>
        </div>

        <div className="space-y-6 relative z-10">
          <div className="flex items-center gap-4 bg-white/10 backdrop-blur-sm p-4 rounded-xl border border-white/10 w-fit">
            <Globe className="text-brand-primary-light" size={24} />
            <div>
              <p className="font-bold text-sm text-white">14 languages</p>
              <p className="text-xs text-white/70">Seamless local dialects</p>
            </div>
          </div>
          <div className="flex items-center gap-4 bg-white/10 backdrop-blur-sm p-4 rounded-xl border border-white/10 w-fit">
            <BrainCircuit className="text-brand-primary-light" size={24} />
            <div>
              <p className="font-bold text-sm text-white">AI documentation</p>
              <p className="text-xs text-white/70">Instant clinical summaries</p>
            </div>
          </div>
          <div className="flex items-center gap-4 bg-white/10 backdrop-blur-sm p-4 rounded-xl border border-white/10 w-fit">
            <CheckCircle2 className="text-brand-primary-light" size={24} />
            <div>
              <p className="font-bold text-sm text-white">Less admin</p>
              <p className="text-xs text-white/70">Focus on the patient</p>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel: Form */}
      <div className="w-full md:w-[55%] bg-surface-card flex flex-col p-6 sm:p-12 md:p-16 relative shadow-[-10px_0_30px_rgba(0,0,0,0.05)] z-10">
        <div className="absolute top-6 right-6">
          <select className="bg-surface-ground border border-border-subtle rounded-lg px-3 py-1.5 text-xs font-semibold text-text-secondary focus:outline-none focus:ring-1 focus:ring-brand-primary cursor-pointer">
            <option>English (India)</option>
            <option>हिंदी</option>
          </select>
        </div>

        <div className="flex-1 flex flex-col justify-center max-w-[400px] w-full mx-auto relative">
          
          <div className={cn("transition-all duration-500 absolute w-full", step === 'phone' ? 'opacity-100 z-10 translate-x-0' : 'opacity-0 z-0 -translate-x-10 pointer-events-none')}>
            <h2 className="text-3xl font-bold text-text-primary mb-2">Welcome back</h2>
            <p className="text-text-secondary mb-10">Sign in to your clinic to continue.</p>

            <form onSubmit={handlePhoneSubmit} className="space-y-6">
              <div className="space-y-2">
                <label className="text-xs font-bold text-text-secondary uppercase tracking-wider">Mobile Number</label>
                <div className="relative flex items-center">
                  <div className="absolute left-0 top-0 bottom-0 flex items-center px-4 bg-surface-ground border border-r-0 border-border-subtle rounded-l-xl text-text-secondary font-semibold">
                    +91
                  </div>
                  <input 
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    placeholder="98765 43210"
                    className="w-full h-14 pl-[72px] pr-4 bg-surface-ground border border-border-subtle rounded-xl text-text-primary font-semibold text-lg focus:outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary transition-colors"
                  />
                </div>
                {error && <p className="text-danger text-sm font-semibold mt-1">{error}</p>}
              </div>

              <button 
                type="submit"
                disabled={loading || !phone}
                className="w-full h-14 bg-brand-primary text-text-on-brand rounded-xl font-bold text-lg hover:bg-brand-primary-mid transition-colors disabled:opacity-50 shadow-sm flex items-center justify-center"
              >
                {loading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : 'Send OTP'}
              </button>
            </form>

            {/* Demo OTP hint */}
            <div className="mt-8 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
              <FlaskConical size={16} className="text-amber-600 shrink-0 mt-0.5" />
              <p className="text-xs font-medium text-amber-800">
                <span className="font-bold">Demo mode</span> — any 6-digit OTP works. Use <code className="bg-amber-100 px-1 rounded font-mono">000000</code> to test the invalid-OTP error.
              </p>
            </div>

            <div className="flex items-center gap-4 my-6">
              <div className="h-px flex-1 bg-border-subtle" />
              <span className="text-xs font-bold text-text-tertiary uppercase tracking-wider">Or</span>
              <div className="h-px flex-1 bg-border-subtle" />
            </div>

            <button
              type="button"
              onClick={() => addToast({ type: 'info', message: 'Google Sign-In is not available in demo mode.' })}
              className="w-full h-14 bg-surface-card border border-border-subtle text-text-primary rounded-xl font-bold hover:bg-surface-ground transition-colors flex items-center justify-center gap-3"
            >
              <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google" className="w-5 h-5" />
              Continue with Google
            </button>

            <p className="mt-8 text-center text-sm font-medium text-text-secondary">
              New to CPMS? <button onClick={() => navigate('/setup')} className="text-brand-primary hover:underline font-bold">Set up your account</button>
            </p>
          </div>

          <div className={cn("transition-all duration-500 absolute w-full", step === 'otp' ? 'opacity-100 z-10 translate-x-0' : 'opacity-0 z-0 translate-x-10 pointer-events-none')}>
            <button 
              type="button"
              onClick={() => { setStep('phone'); setOtp(['','','','','','']); setError(''); }}
              className="text-brand-primary text-sm font-bold hover:underline mb-8 inline-block"
            >
              ← Back
            </button>
            <h2 className="text-3xl font-bold text-text-primary mb-2">Verify your number</h2>
            <p className="text-text-secondary mb-10">
              We sent a 6-digit code to <span className="font-bold text-text-primary">+91 {phone}</span>
            </p>

            <form onSubmit={handleOtpSubmit} className="space-y-8">
              <div>
                <div className="flex justify-between gap-2 md:gap-3">
                  {otp.map((digit, i) => (
                    <input
                      key={i}
                      id={`otp-${i}`}
                      type="text"
                      maxLength={1}
                      value={digit}
                      onChange={(e) => handleOtpChange(i, e.target.value)}
                      onKeyDown={(e) => handleOtpKeyDown(i, e)}
                      className={cn(
                        "w-12 h-14 md:w-14 md:h-16 text-center text-2xl font-bold rounded-xl border bg-surface-ground focus:outline-none focus:ring-2 transition-all",
                        error ? "border-danger focus:border-danger focus:ring-danger/20" : "border-border-subtle focus:border-brand-primary focus:ring-brand-primary/20",
                        digit && !error ? "border-brand-primary" : ""
                      )}
                    />
                  ))}
                </div>
                {error && <p className="text-danger text-sm font-semibold mt-3 text-center animate-fade-in">{error}</p>}
              </div>

              <button 
                type="submit"
                disabled={loading || otp.join('').length !== 6}
                className="w-full h-14 bg-brand-primary text-text-on-brand rounded-xl font-bold text-lg hover:bg-brand-primary-mid transition-colors disabled:opacity-50 shadow-sm flex items-center justify-center"
              >
                {loading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : 'Verify'}
              </button>
            </form>

            <p className="mt-8 text-center text-sm font-medium text-text-secondary">
              Didn't receive the code?{' '}
              {timeLeft > 0 ? (
                <span className="text-text-tertiary">Resend in 0:{timeLeft.toString().padStart(2, '0')}</span>
              ) : (
                <button onClick={() => setTimeLeft(30)} className="text-brand-primary hover:underline font-bold">Resend now</button>
              )}
            </p>
          </div>

        </div>
      </div>
    </div>
  );
}
