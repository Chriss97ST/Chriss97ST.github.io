# 📷 RetroLens Studio – GitHub Pages Deployment Guide

Diese Anleitung zeigt dir Schritt für Schritt, wie du **RetroLens Studio** auf GitHub veröffentlichst und über **GitHub Pages** als PWA online bereitstellst.

---

## 🚀 Schritt 1: GitHub Repository erstellen

1. Gehe auf [github.com](https://github.com) und logge dich ein.
2. Klicke oben rechts auf das **`+`**-Symbol und wähle **`New repository`**.
3. Vergib einen Namen für das Repository (z. B. `retrolens-studio` oder `MultiTool`).
4. Wähle **Public** (oder Private, wenn du GitHub Pro hast).
5. **Wichtig:** Setze *keine* Haken bei „Add a README file“, „.gitignore“ oder „license“, da das Projekt lokal bereits initialisiert ist.
6. Klicke auf **`Create repository`**.

---

## 💻 Schritt 2: Lokalen Code zu GitHub pushen

Führe in deinem Terminal im Projektordner folgende Befehle aus (ersetze `<DEIN-BENUTZERNAME>` und `<DEIN-REPO>`):

```bash
# 1. Remote-Origin hinzufügen (falls noch nicht vorhanden)
git remote add origin https://github.com/<DEIN-BENUTZERNAME>/<DEIN-REPO>.git

# 2. Aktuellen Branch als main setzen
git branch -M main

# 3. Alle Änderungen committen und hochladen
git add .
git commit -m "Configure GitHub Pages automated deployment workflow"
git push -u origin main
```

---

## ⚙️ Schritt 3: GitHub Pages in den Einstellungen aktivieren

1. Öffne dein Repository auf GitHub.
2. Klicke oben auf **`Settings`** (Zahnrad).
3. Navigiere in der linken Seitenleiste zu **`Pages`** (unter dem Bereich *Code and automation*).
4. Wähle unter **Build and deployment** bei **Source** die Option:
   - **`GitHub Actions`** (nicht `Deploy from a branch`).

---

## ⏱️ Schritt 4: Automatisches Deployment prüfen

Sobald du die GitHub Actions ausgewählt hast und Code auf den `main`-Branch pushst:

1. Klicke in deinem GitHub-Repository auf den Tab **`Actions`**.
2. Du siehst den Workflow **`Deploy RetroLens Studio to GitHub Pages`**.
3. Nach ca. 1 Minute ist der Workflow grün abgehakt (`Success`).
4. Unter **`Settings` > `Pages`** (oder direkt im Actions-Lauf) siehst du deine fertige Live-URL:
   ```
   https://<DEIN-BENUTZERNAME>.github.io/<DEIN-REPO>/
   ```

---

## 📱 PWA & HTTPS Hinweis

- **HTTPS:** GitHub Pages stellt automatisch ein gültiges SSL-Zertifikat bereit. Dies ist die Voraussetzung dafür, dass der **Service Worker**, die **PWA-Installation** und die **Kamera-Berechtigung (WebRTC)** im Browser funktionieren.
- **PWA-Installation:** Öffne die generierte URL auf deinem Smartphone oder Desktop, um RetroLens Studio direkt über die Schaltfläche als native App zu installieren.
