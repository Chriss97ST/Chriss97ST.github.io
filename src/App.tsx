import { useState, useEffect } from 'react'
import { Header } from './components/Header'
import { StudioView } from './components/StudioView'
import { CameraView } from './components/CameraView'
import { PresetSelector } from './components/PresetSelector'
import { AdjustmentsPanel } from './components/AdjustmentsPanel'
import { ExportModal } from './components/ExportModal'
import { PWAInstallPrompt } from './components/PWAInstallPrompt'
import { RETRO_PRESETS } from './constants/presets'
import { generateSampleImages } from './constants/sampleImages'
import { audioEffects } from './services/audioEffects'
import type { RetroPreset, RetroSettings } from './types/retro'
import { Sliders, Sparkles } from 'lucide-react'
import './App.css'

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

interface SeedHistoryItem {
  seed: number
  lightLeakPosition: number
}

export function App() {
  const [mode, setMode] = useState<'studio' | 'camera'>('studio')
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null)
  const [selectedPresetId, setSelectedPresetId] = useState<string>('90s-disposable-cam')
  const [settings, setSettings] = useState<RetroSettings>(() => {
    const defaultPreset = RETRO_PRESETS.find((p) => p.id === '90s-disposable-cam') || RETRO_PRESETS[0]
    return { ...defaultPreset.settings, seed: Math.floor(Math.random() * 9999) }
  })

  // Seed history for Undo functionality
  const [seedHistory, setSeedHistory] = useState<SeedHistoryItem[]>(() => [
    { seed: settings.seed, lightLeakPosition: settings.lightLeakPosition }
  ])
  const [seedHistoryIndex, setSeedHistoryIndex] = useState<number>(0)

  const [sidebarTab, setSidebarTab] = useState<'presets' | 'adjustments'>('presets')
  const [audioEnabled, setAudioEnabled] = useState<boolean>(true)
  const [showExportModal, setShowExportModal] = useState<boolean>(false)
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null)

  // Load default sample photo on initial mount
  useEffect(() => {
    const samples = generateSampleImages()
    if (samples.length > 0) {
      const img = new Image()
      img.onload = () => {
        setImageElement(img)
      }
      img.src = samples[0].dataUrl
    }
  }, [])

  // Listen for PWA BeforeInstallPrompt
  useEffect(() => {
    const handleBeforeInstall = (e: Event) => {
      e.preventDefault()
      setDeferredPrompt(e as BeforeInstallPromptEvent)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstall)
    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstall)
    }
  }, [])

  // Toggle Audio
  const handleToggleAudio = () => {
    const next = !audioEnabled
    setAudioEnabled(next)
    audioEffects.enabled = next
    if (next) audioEffects.playDialTick()
  }

  // Preset Selection
  const handleSelectPreset = (preset: RetroPreset) => {
    setSelectedPresetId(preset.id)
    setSettings({
      ...preset.settings,
      seed: settings.seed,
    })
    audioEffects.playDialTick()
  }

  // Re-roll dynamic seed & push to history
  const handleRandomizeSeed = () => {
    const newSeed = Math.floor(Math.random() * 99999)
    const newPos = Math.floor(Math.random() * 5)
    
    setSeedHistory((prev) => {
      const next = prev.slice(0, seedHistoryIndex + 1)
      next.push({ seed: newSeed, lightLeakPosition: newPos })
      return next
    })
    setSeedHistoryIndex((prev) => prev + 1)

    setSettings((prev) => ({
      ...prev,
      seed: newSeed,
      lightLeakPosition: newPos,
    }))
    audioEffects.playDialTick()
  }

  // Undo seed change
  const handleUndoSeed = () => {
    if (seedHistoryIndex > 0) {
      const prevIndex = seedHistoryIndex - 1
      const prevItem = seedHistory[prevIndex]
      setSeedHistoryIndex(prevIndex)
      setSettings((prev) => ({
        ...prev,
        seed: prevItem.seed,
        lightLeakPosition: prevItem.lightLeakPosition,
      }))
      audioEffects.playDialTick()
    }
  }

  const canUndoSeed = seedHistoryIndex > 0

  // Reset to default preset
  const handleResetSettings = () => {
    const currentPreset = RETRO_PRESETS.find((p) => p.id === selectedPresetId) || RETRO_PRESETS[0]
    setSettings({ ...currentPreset.settings })
    audioEffects.playDialTick()
  }

  // Photo captured from Live Camera
  const handlePhotoCaptured = (img: HTMLImageElement) => {
    setImageElement(img)
    setMode('studio')
    audioEffects.playDialTick()
  }

  return (
    <div className="app-layout">
      {/* Top Header */}
      <Header
        mode={mode}
        onModeChange={(newMode) => {
          setMode(newMode)
          audioEffects.playDialTick()
        }}
        audioEnabled={audioEnabled}
        onToggleAudio={handleToggleAudio}
        onRandomizeSeed={handleRandomizeSeed}
        onUndoSeed={handleUndoSeed}
        canUndoSeed={canUndoSeed}
        onResetSettings={handleResetSettings}
        onOpenExport={() => setShowExportModal(true)}
        canExport={!!imageElement}
        canInstallPWA={!!deferredPrompt}
        onInstallPWA={() => {
          if (deferredPrompt) {
            deferredPrompt.prompt()
          }
        }}
      />

      {/* Main Studio / Camera Container */}
      <main className="main-content-row">
        {mode === 'camera' ? (
          <div className="camera-fullscreen-container">
            <CameraView
              settings={settings}
              selectedPresetId={selectedPresetId}
              onSelectPreset={handleSelectPreset}
              onPhotoCaptured={handlePhotoCaptured}
              onClose={() => setMode('studio')}
              onRandomizeSeed={handleRandomizeSeed}
              onUndoSeed={handleUndoSeed}
              canUndoSeed={canUndoSeed}
            />
          </div>
        ) : (
          <div className="workspace-split">
            {/* Left: Viewport */}
            <section className="viewport-section">
              <StudioView
                imageElement={imageElement}
                onImageLoaded={(img) => setImageElement(img)}
                settings={settings}
                onRandomizeSeed={handleRandomizeSeed}
                onUndoSeed={handleUndoSeed}
                canUndoSeed={canUndoSeed}
              />
            </section>

            {/* Right: Sidebar Controls */}
            <aside className="sidebar-section">
              {/* Sidebar Header Tabs */}
              <div className="sidebar-nav-tabs">
                <button
                  className={`sidebar-nav-btn ${sidebarTab === 'presets' ? 'active' : ''}`}
                  onClick={() => {
                    setSidebarTab('presets')
                    audioEffects.playDialTick()
                  }}
                >
                  <Sparkles size={16} />
                  <span>Jahrzehnte-Presets</span>
                </button>
                <button
                  className={`sidebar-nav-btn ${sidebarTab === 'adjustments' ? 'active' : ''}`}
                  onClick={() => {
                    setSidebarTab('adjustments')
                    audioEffects.playDialTick()
                  }}
                >
                  <Sliders size={16} />
                  <span>Feinjustierung</span>
                </button>
              </div>

              {/* Sidebar Body */}
              <div className="sidebar-scrollable-body">
                {sidebarTab === 'presets' ? (
                  <PresetSelector
                    selectedPresetId={selectedPresetId}
                    onSelectPreset={handleSelectPreset}
                  />
                ) : (
                  <AdjustmentsPanel
                    settings={settings}
                    onChange={(updated) => setSettings(updated)}
                    onRandomizeSeed={handleRandomizeSeed}
                    onUndoSeed={handleUndoSeed}
                    canUndoSeed={canUndoSeed}
                  />
                )}
              </div>
            </aside>
          </div>
        )}
      </main>

      {/* PWA Install Banner & Offline Notice */}
      <PWAInstallPrompt
        deferredPrompt={deferredPrompt}
        onInstallComplete={() => setDeferredPrompt(null)}
      />

      {/* High-Res Export & Share Modal */}
      {showExportModal && imageElement && (
        <ExportModal
          imageElement={imageElement}
          settings={settings}
          onClose={() => setShowExportModal(false)}
        />
      )}
    </div>
  )
}

export default App
