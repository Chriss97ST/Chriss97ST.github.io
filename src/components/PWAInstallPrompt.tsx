import React, { useState, useEffect } from 'react';
import { Download, X, WifiOff } from 'lucide-react';

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface PWAInstallPromptProps {
  deferredPrompt: BeforeInstallPromptEvent | null;
  onInstallComplete: () => void;
}

export const PWAInstallPrompt: React.FC<PWAInstallPromptProps> = ({
  deferredPrompt,
  onInstallComplete,
}) => {
  const [showBanner, setShowBanner] = useState<boolean>(false);
  const [isIOS, setIsIOS] = useState<boolean>(false);
  const [isStandalone, setIsStandalone] = useState<boolean>(false);
  const [isOffline, setIsOffline] = useState<boolean>(!navigator.onLine);

  useEffect(() => {
    // Check if already running in standalone PWA mode
    const isRunningStandalone =
      window.matchMedia('(display-mode: standalone)').matches ||
      (window.navigator as unknown as { standalone?: boolean }).standalone === true;
    setIsStandalone(isRunningStandalone);

    // Check iOS Safari
    const userAgent = window.navigator.userAgent.toLowerCase();
    const isIosDevice = /iphone|ipad|ipod/.test(userAgent);
    setIsIOS(isIosDevice);

    if (deferredPrompt && !isRunningStandalone) {
      setShowBanner(true);
    }

    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [deferredPrompt]);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setShowBanner(false);
      onInstallComplete();
    }
  };

  if (isStandalone && !isOffline) return null;

  return (
    <>
      {/* Offline Status Badge */}
      {isOffline && (
        <div className="offline-pill">
          <WifiOff size={14} />
          <span>Offline-Modus aktiv (Voll funktionsfähig)</span>
        </div>
      )}

      {/* Install Banner */}
      {showBanner && !isStandalone && (
        <div className="pwa-install-banner">
          <div className="pwa-banner-left">
            <div className="pwa-icon-box">📷</div>
            <div>
              <h4 className="pwa-banner-title">RetroLens Studio als App installieren</h4>
              <p className="pwa-banner-desc">
                Direkt vom Homescreen starten, vollständige Offline-Nutzung & blitzschnelle Kamera-Reaktionszeit.
              </p>
            </div>
          </div>

          <div className="pwa-banner-actions">
            {deferredPrompt ? (
              <button className="primary-btn pwa-btn" onClick={handleInstallClick}>
                <Download size={16} />
                <span>Jetzt Installieren</span>
              </button>
            ) : isIOS ? (
              <div className="ios-instructions">
                <span>Tippe auf Teilen ⎋ und dann <strong>„Zum Home-Bildschirm“</strong></span>
              </div>
            ) : null}

            <button className="pwa-close-btn" onClick={() => setShowBanner(false)}>
              <X size={16} />
            </button>
          </div>
        </div>
      )}
    </>
  );
};
