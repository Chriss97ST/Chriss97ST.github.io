const DB_NAME = "worknotiz-db";
const DB_VERSION = 2;
const USERS_STORE = "users";
const ENTRIES_STORE = "entries";
const LEGACY_STORAGE_KEY = "work-notes.v1";
const SESSION_USER_KEY = "worknotiz-current-user";
const THEME_KEY = "worknotiz-theme";

const defaultComplianceRange = getDefaultComplianceRange();

const state = {
  db: null,
  currentUser: null,
  authView: "actions",
  menuOpen: false,
  theme: "light",
  selectedDate: toDateInputValue(new Date()),
  entries: [],
  searchQuery: "",
  editingEntryId: null,
  entryFormOpen: false,
  searchOpen: false,
  complianceRange: "last2weeks",
  complianceFrom: defaultComplianceRange.from,
  complianceTo: defaultComplianceRange.to,
  complianceMissingTime: true,
  complianceMissingInternalOrder: true,
  openWeeks: new Set(),
  treeInitialized: false,
};

const elements = {
  appShell: document.getElementById("appShell"),
  appLayout: document.getElementById("appLayout"),
  authTitle: document.getElementById("authTitle"),
  authMessage: document.getElementById("authMessage"),
  authActions: document.getElementById("authActions"),
  showRegisterButton: document.getElementById("showRegisterButton"),
  showLoginButton: document.getElementById("showLoginButton"),
  registerPanel: document.getElementById("registerPanel"),
  loginPanel: document.getElementById("loginPanel"),
  sessionPanel: document.getElementById("sessionPanel"),
  registerForm: document.getElementById("registerForm"),
  loginForm: document.getElementById("loginForm"),
  cancelRegisterButton: document.getElementById("cancelRegisterButton"),
  cancelLoginButton: document.getElementById("cancelLoginButton"),
  registerUsername: document.getElementById("registerUsername"),
  registerPassword: document.getElementById("registerPassword"),
  loginUsername: document.getElementById("loginUsername"),
  loginPassword: document.getElementById("loginPassword"),
  moreMenuButton: document.getElementById("moreMenuButton"),
  moreMenu: document.getElementById("moreMenu"),
  themeToggleButton: document.getElementById("themeToggleButton"),
  currentUserLabel: document.getElementById("currentUserLabel"),
  logoutButton: document.getElementById("logoutButton"),
  form: document.getElementById("entryForm"),
  entryFormPanel: document.getElementById("entryFormPanel"),
  showEntryFormButton: document.getElementById("showEntryFormButton"),
  searchPanel: document.getElementById("searchPanel"),
  showSearchButton: document.getElementById("showSearchButton"),
  formTitle: document.getElementById("entryFormTitle"),
  submitEntryButton: document.getElementById("submitEntryButton"),
  cancelEditButton: document.getElementById("cancelEditButton"),
  date: document.getElementById("entryDate"),
  customer: document.getElementById("customer"),
  internalOrder: document.getElementById("internalOrder"),
  customerOrder: document.getElementById("customerOrder"),
  activity: document.getElementById("activity"),
  duration: document.getElementById("duration"),
  quickDay: document.getElementById("quickDay"),
  searchInput: document.getElementById("searchInput"),
  clearSearch: document.getElementById("clearSearch"),
  complianceRange: document.getElementById("complianceRange"),
  complianceFrom: document.getElementById("complianceFrom"),
  complianceTo: document.getElementById("complianceTo"),
  complianceMissingTime: document.getElementById("complianceMissingTime"),
  complianceMissingInternalOrder: document.getElementById("complianceMissingInternalOrder"),
  complianceApply: document.getElementById("complianceApply"),
  complianceSummary: document.getElementById("complianceSummary"),
  complianceList: document.getElementById("complianceList"),
  searchHint: document.getElementById("searchHint"),
  treeRoot: document.getElementById("treeRoot"),
  dayEntries: document.getElementById("dayEntries"),
  dayTitle: document.getElementById("selectedDayTitle"),
  weekTitle: document.getElementById("selectedWeekTitle"),
  entryTemplate: document.getElementById("entryTemplate"),
};

init();

async function init() {
  state.db = await openDatabase();
  applyTheme(localStorage.getItem(THEME_KEY) || "light");

  elements.showRegisterButton.addEventListener("click", () => setAuthView("register"));
  elements.showLoginButton.addEventListener("click", () => setAuthView("login"));
  elements.registerForm.addEventListener("submit", handleRegister);
  elements.loginForm.addEventListener("submit", handleLogin);
  elements.cancelRegisterButton.addEventListener("click", () => setAuthView("actions"));
  elements.cancelLoginButton.addEventListener("click", () => setAuthView("actions"));
  elements.moreMenuButton.addEventListener("click", toggleMoreMenu);
  elements.showEntryFormButton.addEventListener("click", toggleEntryForm);
  elements.showSearchButton.addEventListener("click", toggleSearchPanel);
  elements.themeToggleButton.addEventListener("click", handleThemeToggle);
  elements.logoutButton.addEventListener("click", handleLogout);
  document.addEventListener("click", handleDocumentClick);

  const datePicker = flatpickr(elements.date, {
    dateFormat: "Y-m-d",
    altInput: true,
    altInputClass: "date-display",
    altFormat: "d.m.Y",
    locale: "de",
    defaultDate: state.selectedDate,
    allowInput: false,
  });
  elements._datePicker = datePicker;
  elements.quickDay.addEventListener("change", () => handleQuickDayChange(datePicker));
  elements.form.addEventListener("submit", handleSubmit);
  elements.cancelEditButton.addEventListener("click", resetEntryFormMode);
  elements.searchInput.addEventListener("input", handleSearchInput);
  elements.clearSearch.addEventListener("click", clearSearch);
  elements.complianceRange.addEventListener("change", handleComplianceRangeChange);
  elements.complianceFrom.addEventListener("change", handleComplianceDateInputChange);
  elements.complianceTo.addEventListener("change", handleComplianceDateInputChange);
  elements.complianceMissingTime.addEventListener("change", handleComplianceToggleChange);
  elements.complianceMissingInternalOrder.addEventListener("change", handleComplianceToggleChange);
  elements.complianceApply.addEventListener("click", handleComplianceApply);

  const sessionUserId = localStorage.getItem(SESSION_USER_KEY);
  if (sessionUserId) {
    const user = await getUserById(sessionUserId);
    if (user) {
      state.currentUser = user;
      state.authView = "session";
      await migrateLegacyDataForUser(state.currentUser.id);
      await refreshEntries();
      setAuthMessage(`Willkommen, ${state.currentUser.username}.`, "info");
    }
  }

  render();
}

async function handleRegister(event) {
  event.preventDefault();

  const username = elements.registerUsername.value.trim();
  const usernameLower = username.toLowerCase();
  const password = elements.registerPassword.value;

  if (username.length < 3) {
    setAuthMessage("Benutzername muss mindestens 3 Zeichen haben.", "error");
    return;
  }

  if (password.length < 6) {
    setAuthMessage("Passwort muss mindestens 6 Zeichen haben.", "error");
    return;
  }

  const existing = await getUserByUsername(usernameLower);
  if (existing) {
    setAuthMessage("Benutzername ist bereits vergeben.", "error");
    return;
  }

  const passwordHash = await hashPassword(password);
  const user = {
    id: crypto.randomUUID(),
    username,
    usernameLower,
    passwordHash,
    createdAt: new Date().toISOString(),
  };

  await putUser(user);
  state.currentUser = user;
  localStorage.setItem(SESSION_USER_KEY, user.id);
  state.searchQuery = "";
  elements.searchInput.value = "";

  await migrateLegacyDataForUser(user.id);
  await refreshEntries();

  elements.registerForm.reset();
  elements.loginForm.reset();
  state.authView = "session";
  setAuthMessage(`Konto erstellt. Angemeldet als ${user.username}.`, "ok");
  render();
}

async function handleLogin(event) {
  event.preventDefault();

  const username = elements.loginUsername.value.trim();
  const password = elements.loginPassword.value;

  if (!username || !password) {
    setAuthMessage("Bitte Benutzername und Passwort eingeben.", "error");
    return;
  }

  const user = await getUserByUsername(username.toLowerCase());
  if (!user || !user.passwordHash) {
    setAuthMessage("Login fehlgeschlagen: Benutzer oder Passwort falsch.", "error");
    return;
  }

  const passwordHash = await hashPassword(password);
  if (passwordHash !== user.passwordHash) {
    setAuthMessage("Login fehlgeschlagen: Benutzer oder Passwort falsch.", "error");
    return;
  }

  state.currentUser = user;
  localStorage.setItem(SESSION_USER_KEY, user.id);
  state.searchQuery = "";
  elements.searchInput.value = "";

  await migrateLegacyDataForUser(user.id);
  await refreshEntries();

  elements.loginForm.reset();
  state.authView = "session";
  setAuthMessage(`Angemeldet als ${user.username}.`, "ok");
  render();
}

function handleLogout() {
  state.currentUser = null;
  state.authView = "actions";
  state.menuOpen = false;
  state.entries = [];
  state.searchQuery = "";
  state.openWeeks = new Set();
  state.treeInitialized = false;
  elements.searchInput.value = "";
  elements.registerForm.reset();
  elements.loginForm.reset();
  localStorage.removeItem(SESSION_USER_KEY);
  setAuthMessage("Abgemeldet.", "info");
  render();
}

async function handleSubmit(event) {
  event.preventDefault();

  if (!state.currentUser) {
    setAuthMessage("Bitte zuerst anmelden.", "error");
    return;
  }

  const date = elements.date.value;
  const customer = elements.customer.value.trim();
  const internalOrder = elements.internalOrder.value.trim();
  const customerOrder = elements.customerOrder.value.trim();
  const activity = elements.activity.value.trim();
  const durationMinutes = Number(elements.duration.value);

  if (
    !date ||
    !customer ||
    !internalOrder ||
    !customerOrder ||
    !activity ||
    !durationMinutes
  ) {
    return;
  }

  const existingEntry = state.editingEntryId
    ? state.entries.find((entry) => entry.id === state.editingEntryId)
    : null;

  const note = {
    id: existingEntry ? existingEntry.id : crypto.randomUUID(),
    userId: state.currentUser.id,
    date,
    weekKey: getWeekKey(date),
    customer,
    customerLower: customer.toLowerCase(),
    internalOrder,
    internalOrderLower: internalOrder.toLowerCase(),
    customerOrder,
    customerOrderLower: customerOrder.toLowerCase(),
    activity,
    activityLower: activity.toLowerCase(),
    durationMinutes,
    createdAt: existingEntry ? existingEntry.createdAt : new Date().toISOString(),
  };

  if (existingEntry) {
    note.updatedAt = new Date().toISOString();
  }

  await putEntry(note);
  state.selectedDate = date;

  await refreshEntries();
  resetEntryFormMode();
  setEntryDate(date);
  render();
}

function handleQuickDayChange(datePicker) {
  const today = new Date();

  if (elements.quickDay.value === "today") {
    datePicker.setDate(today, true);
  }

  if (elements.quickDay.value === "yesterday") {
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    datePicker.setDate(yesterday, true);
  }
}

function handleSearchInput(event) {
  state.searchQuery = event.target.value.trim().toLowerCase();
  render();
}

function clearSearch() {
  state.searchQuery = "";
  state.searchOpen = false;
  elements.searchInput.value = "";
  render();
}

function handleComplianceRangeChange(event) {
  state.complianceRange = event.target.value;
  if (state.complianceRange !== "custom") {
    const range = getComplianceRangeByPreset(state.complianceRange);
    state.complianceFrom = range.from;
    state.complianceTo = range.to;
  }
  render();
}

function handleComplianceDateInputChange() {
  state.complianceRange = "custom";
  state.complianceFrom = elements.complianceFrom.value || state.complianceFrom;
  state.complianceTo = elements.complianceTo.value || state.complianceTo;
}

function handleComplianceToggleChange() {
  state.complianceMissingTime = elements.complianceMissingTime.checked;
  state.complianceMissingInternalOrder = elements.complianceMissingInternalOrder.checked;
  render();
}

function handleComplianceApply() {
  state.complianceRange = "custom";
  state.complianceFrom = elements.complianceFrom.value || state.complianceFrom;
  state.complianceTo = elements.complianceTo.value || state.complianceTo;

  if (state.complianceFrom > state.complianceTo) {
    const temp = state.complianceFrom;
    state.complianceFrom = state.complianceTo;
    state.complianceTo = temp;
  }

  render();
}

function toggleSearchPanel() {
  state.searchOpen = !state.searchOpen;
  if (!state.searchOpen) {
    state.searchQuery = "";
    elements.searchInput.value = "";
  }
  render();
}

function toggleEntryForm() {
  state.entryFormOpen = !state.entryFormOpen;
  if (!state.entryFormOpen) {
    resetEntryFormMode();
  }
  render();
}

function startEditingEntry(entryId) {
  const entry = state.entries.find((item) => item.id === entryId);
  if (!entry) {
    return;
  }

  state.editingEntryId = entry.id;
  state.entryFormOpen = true;
  setEntryDate(entry.date);
  elements.customer.value = entry.customer || "";
  elements.internalOrder.value = entry.internalOrder || "";
  elements.customerOrder.value = entry.customerOrder || "";
  elements.activity.value = entry.activity || "";
  elements.duration.value = entry.durationMinutes || "";
  elements.quickDay.value = "manual";
  updateEntryFormModeUI();
  elements.customer.focus();
}

function resetEntryFormMode() {
  state.editingEntryId = null;
  state.entryFormOpen = false;
  elements.form.reset();
  setEntryDate(state.selectedDate);
  elements.quickDay.value = "manual";
  updateEntryFormModeUI();
}

function updateEntryFormModeUI() {
  const isEditing = Boolean(state.editingEntryId);
  elements.formTitle.textContent = isEditing ? "Tagesnotiz bearbeiten" : "Tagesnotiz erstellen";
  elements.submitEntryButton.textContent = isEditing ? "Änderungen speichern" : "Eintrag speichern";
  elements.cancelEditButton.hidden = !isEditing;
  elements.showEntryFormButton.textContent = state.entryFormOpen ? "Formular schließen" : "+ Eintrag erstellen";
  elements.showSearchButton.textContent = state.searchOpen ? "Suche schließen" : "Suchen";
}

function setEntryDate(date) {
  if (elements._datePicker) {
    elements._datePicker.setDate(date, true);
    return;
  }

  elements.date.value = date;
}

function render() {
  updateAuthUI();
  updateEntryFormModeUI();
  updateComplianceFilterUI();
  renderTree();
  renderSelectedDay();
}

function updateComplianceFilterUI() {
  elements.complianceRange.value = state.complianceRange;
  elements.complianceFrom.value = state.complianceFrom;
  elements.complianceTo.value = state.complianceTo;
  elements.complianceMissingTime.checked = state.complianceMissingTime;
  elements.complianceMissingInternalOrder.checked = state.complianceMissingInternalOrder;
}

function updateAuthUI() {
  const loggedIn = Boolean(state.currentUser);
  elements.appShell.hidden = !loggedIn;
  elements.authTitle.textContent = loggedIn ? "WorkNotizApp - " + state.currentUser.username : "Willkommen";
  elements.currentUserLabel.textContent = loggedIn
    ? `Aktiver Benutzer: ${state.currentUser.username}`
    : "Nicht angemeldet";
  elements.moreMenuButton.hidden = !loggedIn;
  elements.showEntryFormButton.hidden = !loggedIn;
  elements.showSearchButton.hidden = !loggedIn;
  elements.entryFormPanel.hidden = !loggedIn || !state.entryFormOpen;
  elements.searchPanel.hidden = !loggedIn || !state.searchOpen;
  elements.logoutButton.disabled = !loggedIn;
  elements.moreMenu.hidden = !loggedIn || !state.menuOpen;
  elements.themeToggleButton.textContent =
    state.theme === "dark" ? "Normalmodus aktivieren" : "Darkmode aktivieren";

  const effectiveView = loggedIn ? "session" : state.authView;
  elements.authActions.hidden = effectiveView !== "actions";
  elements.registerPanel.hidden = effectiveView !== "register";
  elements.loginPanel.hidden = effectiveView !== "login";
}

function setAuthView(view) {
  state.authView = view;
  state.menuOpen = false;
  render();
}

function toggleMoreMenu(event) {
  event.stopPropagation();
  if (!state.currentUser) {
    return;
  }

  state.menuOpen = !state.menuOpen;
  render();
}

function handleDocumentClick(event) {
  if (!state.menuOpen) {
    return;
  }

  const clickedInsideMenu = elements.moreMenu.contains(event.target);
  const clickedMenuButton = elements.moreMenuButton.contains(event.target);

  if (!clickedInsideMenu && !clickedMenuButton) {
    state.menuOpen = false;
    render();
  }
}

function handleThemeToggle() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
  localStorage.setItem(THEME_KEY, state.theme);
  state.menuOpen = false;
  render();
}

function applyTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.body.classList.toggle("theme-dark", state.theme === "dark");
}

function renderTree() {
  elements.treeRoot.innerHTML = "";

  if (!state.currentUser) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Bitte anmelden, um deine Wochenordner zu sehen.";
    elements.treeRoot.appendChild(empty);
    return;
  }

  const grouped = groupEntriesByWeekAndDay(state.entries);
  const weekKeys = Object.keys(grouped).sort().reverse();
  const today = toDateInputValue(new Date());
  const todayWeekKey = getWeekKey(today);
  const selectedWeekKey = getWeekKey(state.selectedDate);

  // Die Woche des gewählten Tages ist immer offen
  state.openWeeks.add(selectedWeekKey);
  // Beim ersten Aufbau zusätzlich die aktuelle Woche öffnen
  if (!state.treeInitialized) {
    state.openWeeks.add(todayWeekKey);
    state.treeInitialized = true;
  }

  if (weekKeys.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Noch keine Einträge vorhanden.";
    elements.treeRoot.appendChild(empty);
    return;
  }

  weekKeys.forEach((weekKey) => {
    const isOpen = state.openWeeks.has(weekKey);
    const isCurrentWeek = weekKey === todayWeekKey;
    const [, weekNum] = weekKey.split("-W");
    const totalEntries = Object.values(grouped[weekKey]).reduce(
      (sum, days) => sum + days.length,
      0
    );

    const weekBlock = document.createElement("section");
    weekBlock.className = "week-block" + (isCurrentWeek ? " week-current" : "");

    // ── Toggle-Kopfzeile ─────────────────────────────────
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "week-toggle";

    const info = document.createElement("span");
    info.className = "week-toggle-info";

    const kwSpan = document.createElement("span");
    kwSpan.className = "week-kw";
    kwSpan.textContent = `KW ${weekNum}`;

    const rangeSpan = document.createElement("span");
    rangeSpan.className = "week-range";
    rangeSpan.textContent = getWeekRangeCompact(weekKey);

    info.appendChild(kwSpan);
    info.appendChild(rangeSpan);

    const meta = document.createElement("span");
    meta.className = "week-toggle-meta";

    const totalBadge = document.createElement("span");
    totalBadge.className = "week-total";
    totalBadge.textContent = totalEntries;

    const arrow = document.createElement("span");
    arrow.className = "week-arrow" + (isOpen ? " open" : "");
    arrow.textContent = "›";

    meta.appendChild(totalBadge);
    meta.appendChild(arrow);
    toggle.appendChild(info);
    toggle.appendChild(meta);

    toggle.addEventListener("click", () => {
      if (state.openWeeks.has(weekKey)) {
        // Aktive Woche kann nicht zugeklappt werden
        if (weekKey !== selectedWeekKey) {
          state.openWeeks.delete(weekKey);
        }
      } else {
        state.openWeeks.add(weekKey);
      }
      render();
    });

    weekBlock.appendChild(toggle);

    // ── Tagesliste ───────────────────────────────────────
    if (isOpen) {
      const body = document.createElement("div");
      body.className = "week-body";

      const dayKeys = Object.keys(grouped[weekKey]).sort().reverse();
      dayKeys.forEach((dayKey) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "day-item";
        if (dayKey === today) btn.classList.add("day-today");
        if (state.selectedDate === dayKey && !state.searchQuery) btn.classList.add("active");

        const count = grouped[weekKey][dayKey].length;

        const label = document.createElement("span");
        label.className = "day-label";
        label.textContent = formatDateCompact(dayKey);

        const badge = document.createElement("span");
        badge.className = "day-badge";
        badge.textContent = count;

        btn.appendChild(label);
        btn.appendChild(badge);
        btn.addEventListener("click", () => {
          state.selectedDate = dayKey;
          render();
        });

        body.appendChild(btn);
      });

      weekBlock.appendChild(body);
    }

    elements.treeRoot.appendChild(weekBlock);
  });
}

function renderSelectedDay() {
  elements.dayEntries.innerHTML = "";
  renderComplianceAlerts();

  if (!state.currentUser) {
    elements.dayTitle.textContent = "Bitte anmelden";
    elements.weekTitle.textContent = "Notizen sind pro Benutzer getrennt gespeichert.";
    elements.searchHint.textContent =
      "Nach Login kannst du nach Auftragsnummer, Kunde und Tätigkeit suchen.";

    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Keine Ansicht ohne Anmeldung.";
    elements.dayEntries.appendChild(empty);
    return;
  }

  if (state.searchQuery) {
    const results = searchEntries(state.entries, state.searchQuery);
    elements.dayTitle.textContent = `Suchergebnisse (${results.length})`;
    elements.weekTitle.textContent = `Suchbegriff: ${elements.searchInput.value.trim()}`;
    elements.searchHint.textContent =
      "Treffer in interner Auftragsnummer, Kunden-Auftragsnummer, Kunde oder Tätigkeit.";

    if (results.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Keine Treffer gefunden.";
      elements.dayEntries.appendChild(empty);
      return;
    }

    results.forEach((entry) => {
      elements.dayEntries.appendChild(createEntryNode(entry, true));
    });

    return;
  }

  const selectedDate = state.selectedDate;
  const weekKey = getWeekKey(selectedDate);
  const dayEntries = state.entries.filter((entry) => entry.date === selectedDate);

  elements.dayTitle.textContent = formatDate(selectedDate);
  elements.weekTitle.textContent = `Wochenordner: ${weekKey}`;
  elements.searchHint.textContent =
    "Suche über interne/kundenbezogene Auftragsnummer, Kundenname und Tätigkeit.";

  if (dayEntries.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    const today = toDateInputValue(new Date());
    empty.textContent = selectedDate === today
      ? "Heute wurden noch keine Notizen erfasst."
      : "An diesem Tag gibt es noch keine Notizen.";
    elements.dayEntries.appendChild(empty);
    return;
  }

  dayEntries
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .forEach((entry) => {
      elements.dayEntries.appendChild(createEntryNode(entry, false));
    });
}

function createEntryNode(entry, showMeta) {
  const node = elements.entryTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".entry-customer").textContent = entry.customer;
  node.querySelector(".entry-duration").textContent = `${entry.durationMinutes} Min`;
  node.querySelector(".entry-order").textContent = formatOrderText(entry);
  node.querySelector(".entry-activity").textContent = entry.activity;
  node.querySelector(".entry-meta").textContent = showMeta
    ? `${formatDate(entry.date)} | ${entry.weekKey}`
    : "";
  node.querySelector(".entry-edit").addEventListener("click", () => startEditingEntry(entry.id));
  return node;
}

function formatOrderText(entry) {
  if (entry.internalOrder || entry.customerOrder) {
    const internalOrder = entry.internalOrder || "-";
    const customerOrder = entry.customerOrder || "-";
    return `Auftrag BEM: ${internalOrder} | Auftrag Kunde: ${customerOrder}`;
  }

  return `Auftrag: ${entry.order || "-"}`;
}

function groupEntriesByWeekAndDay(entries) {
  return entries.reduce((acc, entry) => {
    if (!acc[entry.weekKey]) {
      acc[entry.weekKey] = {};
    }

    if (!acc[entry.weekKey][entry.date]) {
      acc[entry.weekKey][entry.date] = [];
    }

    acc[entry.weekKey][entry.date].push(entry);
    return acc;
  }, {});
}

function searchEntries(entries, query) {
  return entries
    .filter((entry) => {
      const customer = (entry.customer || "").toLowerCase();
      const activity = (entry.activity || "").toLowerCase();
      const internalOrder = (entry.internalOrder || entry.order || "").toLowerCase();
      const customerOrder = (entry.customerOrder || "").toLowerCase();

      return (
        customer.includes(query) ||
        activity.includes(query) ||
        internalOrder.includes(query) ||
        customerOrder.includes(query)
      );
    })
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function renderComplianceAlerts() {
  elements.complianceList.innerHTML = "";

  if (!state.currentUser) {
    elements.complianceSummary.textContent = "";
    return;
  }

  if (!state.complianceMissingTime && !state.complianceMissingInternalOrder) {
    elements.complianceSummary.textContent = "Mindestens einen Prüfpunkt auswählen.";
    return;
  }

  const alerts = getComplianceAlerts(
    state.entries,
    state.complianceFrom,
    state.complianceTo,
    {
      checkMissingTime: state.complianceMissingTime,
      checkMissingInternalOrder: state.complianceMissingInternalOrder,
    }
  );

  const checkedDays = getWorkdayCountInRange(state.complianceFrom, state.complianceTo);
  if (alerts.length === 0) {
    elements.complianceSummary.textContent =
      `Zeitraum ${formatDate(state.complianceFrom, true)} bis ${formatDate(state.complianceTo, true)}: ` +
      `Keine Auffälligkeiten in ${checkedDays} Arbeitstagen.`;
    return;
  }

  elements.complianceSummary.textContent =
    `Zeitraum ${formatDate(state.complianceFrom, true)} bis ${formatDate(state.complianceTo, true)}: ` +
    `${alerts.length} von ${checkedDays} Arbeitstagen mit Auffälligkeiten.`;

  alerts.forEach((alert) => {
    const row = document.createElement("article");
    row.className = "compliance-item";

    const jump = document.createElement("button");
    jump.type = "button";
    jump.className = "compliance-jump";
    jump.textContent = formatDate(alert.date, true);
    jump.addEventListener("click", () => {
      state.selectedDate = alert.date;
      state.searchQuery = "";
      elements.searchInput.value = "";
      render();
    });

    const text = document.createElement("p");
    text.className = "compliance-text";
    text.textContent = alert.issues.join(" | ");

    row.appendChild(jump);
    row.appendChild(text);
    elements.complianceList.appendChild(row);
  });
}

function getComplianceAlerts(entries, fromDate, toDate, options) {
  const byDate = entries.reduce((acc, entry) => {
    if (!acc[entry.date]) {
      acc[entry.date] = [];
    }

    acc[entry.date].push(entry);
    return acc;
  }, {});

  const alerts = [];
  const workdays = getWorkdaysInRange(fromDate, toDate);

  workdays.forEach((date) => {
    const dayEntries = byDate[date] || [];
    const totalMinutes = dayEntries.reduce((sum, entry) => sum + Number(entry.durationMinutes || 0), 0);
    const targetMinutes = getTargetMinutesForDate(date);
    const issues = [];

    if (options.checkMissingTime && totalMinutes < targetMinutes) {
      const missing = targetMinutes - totalMinutes;
      issues.push(
        `Fehlzeit ${formatMinutesCompact(missing)} (erfasst ${formatMinutesCompact(totalMinutes)} von ${formatMinutesCompact(targetMinutes)})`
      );
    }

    if (options.checkMissingInternalOrder) {
      if (dayEntries.length === 0) {
        issues.push("Keine Notiz mit interner Auftragsnummer vorhanden");
      } else {
        const missingOrderCount = dayEntries.filter((entry) => {
          const value = (entry.internalOrder || "").trim();
          return value.length === 0;
        }).length;

        if (missingOrderCount > 0) {
          issues.push(`${missingOrderCount} Eintrag${missingOrderCount === 1 ? "" : "e"} ohne interne Auftragsnummer`);
        }
      }
    }

    if (issues.length > 0) {
      alerts.push({ date, issues });
    }
  });

  return alerts;
}

function getWorkdayCountInRange(fromDate, toDate) {
  return getWorkdaysInRange(fromDate, toDate).length;
}

function getWorkdaysInRange(fromDate, toDate) {
  const dates = [];
  const cursor = new Date(`${fromDate}T00:00:00`);
  const end = new Date(`${toDate}T00:00:00`);

  while (cursor <= end) {
    const weekday = cursor.getDay();
    if (weekday >= 1 && weekday <= 5) {
      dates.push(toDateInputValue(cursor));
    }
    cursor.setDate(cursor.getDate() + 1);
  }

  return dates;
}

function getTargetMinutesForDate(dateString) {
  const weekday = new Date(`${dateString}T00:00:00`).getDay();
  if (weekday >= 1 && weekday <= 4) {
    return 8 * 60;
  }

  if (weekday === 5) {
    return 5 * 60;
  }

  return 0;
}

function formatMinutesCompact(totalMinutes) {
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (minutes === 0) {
    return `${hours}h`;
  }

  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}

function getDefaultComplianceRange() {
  const today = new Date();
  const from = new Date(today);
  from.setDate(today.getDate() - 13);
  return {
    from: toDateInputValue(from),
    to: toDateInputValue(today),
  };
}

function getComplianceRangeByPreset(preset) {
  const today = new Date();

  if (preset === "thisWeek") {
    const monday = getWeekStartDate(today);
    return {
      from: toDateInputValue(monday),
      to: toDateInputValue(today),
    };
  }

  if (preset === "last4weeks") {
    const from = new Date(today);
    from.setDate(today.getDate() - 27);
    return {
      from: toDateInputValue(from),
      to: toDateInputValue(today),
    };
  }

  const from = new Date(today);
  from.setDate(today.getDate() - 13);
  return {
    from: toDateInputValue(from),
    to: toDateInputValue(today),
  };
}

function getWeekStartDate(date) {
  const monday = new Date(date);
  const dayIndex = (monday.getDay() + 6) % 7;
  monday.setDate(monday.getDate() - dayIndex);
  monday.setHours(0, 0, 0, 0);
  return monday;
}

async function refreshEntries() {
  if (!state.currentUser) {
    state.entries = [];
    return;
  }

  const entries = await getEntriesForUser(state.currentUser.id);
  state.entries = entries.map(normalizeEntry);

  const today = toDateInputValue(new Date());
  const hasSelectedDay = state.entries.some((entry) => entry.date === state.selectedDate);
  if (!hasSelectedDay) {
    state.selectedDate = today;
  }
}

function normalizeEntry(entry) {
  const internalOrder = entry.internalOrder || entry.order || "";
  const customerOrder = entry.customerOrder || "";
  const customer = entry.customer || "";
  const activity = entry.activity || "";

  return {
    ...entry,
    weekKey: entry.weekKey || getWeekKey(entry.date),
    internalOrder,
    customerOrder,
    customer,
    activity,
    customerLower: entry.customerLower || customer.toLowerCase(),
    internalOrderLower: entry.internalOrderLower || internalOrder.toLowerCase(),
    customerOrderLower: entry.customerOrderLower || customerOrder.toLowerCase(),
    activityLower: entry.activityLower || activity.toLowerCase(),
  };
}

async function migrateLegacyDataForUser(userId) {
  const legacyRaw = localStorage.getItem(LEGACY_STORAGE_KEY);
  if (!legacyRaw) {
    return;
  }

  const existingEntries = await getEntriesForUser(userId);
  if (existingEntries.length > 0) {
    return;
  }

  try {
    const parsed = JSON.parse(legacyRaw);
    if (!parsed || !parsed.weeks || typeof parsed.weeks !== "object") {
      return;
    }

    const importEntries = [];
    Object.entries(parsed.weeks).forEach(([weekKey, weekData]) => {
      const days = weekData.days || {};
      Object.entries(days).forEach(([date, dayData]) => {
        const entries = Array.isArray(dayData.entries) ? dayData.entries : [];
        entries.forEach((entry) => {
          const internalOrder = entry.internalOrder || entry.order || "";
          const customerOrder = entry.customerOrder || "";
          const customer = entry.customer || "";
          const activity = entry.activity || "";
          importEntries.push({
            id: entry.id || crypto.randomUUID(),
            userId,
            date,
            weekKey: weekKey || getWeekKey(date),
            customer,
            customerLower: customer.toLowerCase(),
            internalOrder,
            internalOrderLower: internalOrder.toLowerCase(),
            customerOrder,
            customerOrderLower: customerOrder.toLowerCase(),
            activity,
            activityLower: activity.toLowerCase(),
            durationMinutes: Number(entry.durationMinutes || 0),
            createdAt: entry.createdAt || new Date().toISOString(),
          });
        });
      });
    });

    await putEntriesBatch(importEntries);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch {
    // ignore malformed legacy payload
  }
}

function setAuthMessage(text, level) {
  elements.authMessage.textContent = text;
  elements.authMessage.dataset.level = level;
}

async function hashPassword(password) {
  const encoder = new TextEncoder();
  const data = encoder.encode(password);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const bytes = Array.from(new Uint8Array(digest));
  return bytes.map((value) => value.toString(16).padStart(2, "0")).join("");
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;

      let users;
      if (!db.objectStoreNames.contains(USERS_STORE)) {
        users = db.createObjectStore(USERS_STORE, { keyPath: "id" });
      } else {
        users = request.transaction.objectStore(USERS_STORE);
      }

      if (!users.indexNames.contains("usernameLower")) {
        users.createIndex("usernameLower", "usernameLower", { unique: true });
      }

      if (!db.objectStoreNames.contains(ENTRIES_STORE)) {
        const entries = db.createObjectStore(ENTRIES_STORE, { keyPath: "id" });
        entries.createIndex("userId", "userId", { unique: false });
        entries.createIndex("weekKey", "weekKey", { unique: false });
        entries.createIndex("date", "date", { unique: false });
        entries.createIndex("customerLower", "customerLower", { unique: false });
        entries.createIndex("activityLower", "activityLower", { unique: false });
        entries.createIndex("internalOrderLower", "internalOrderLower", { unique: false });
        entries.createIndex("customerOrderLower", "customerOrderLower", { unique: false });
        entries.createIndex("createdAt", "createdAt", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function putUser(user) {
  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(USERS_STORE, "readwrite");
    const store = transaction.objectStore(USERS_STORE);
    store.put(user);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

function getUserById(userId) {
  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(USERS_STORE, "readonly");
    const store = transaction.objectStore(USERS_STORE);
    const request = store.get(userId);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

function getUserByUsername(usernameLower) {
  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(USERS_STORE, "readonly");
    const store = transaction.objectStore(USERS_STORE);
    const index = store.index("usernameLower");
    const request = index.get(usernameLower);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

function putEntry(entry) {
  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(ENTRIES_STORE, "readwrite");
    const store = transaction.objectStore(ENTRIES_STORE);
    store.put(entry);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

function putEntriesBatch(entries) {
  if (entries.length === 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(ENTRIES_STORE, "readwrite");
    const store = transaction.objectStore(ENTRIES_STORE);
    entries.forEach((entry) => store.put(entry));
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

function getEntriesForUser(userId) {
  return new Promise((resolve, reject) => {
    const transaction = state.db.transaction(ENTRIES_STORE, "readonly");
    const store = transaction.objectStore(ENTRIES_STORE);
    const request = store.getAll();

    request.onsuccess = () => {
      const allEntries = Array.isArray(request.result) ? request.result : [];
      const userEntries = allEntries
        .filter((entry) => entry.userId === userId)
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
      resolve(userEntries);
    };

    request.onerror = () => reject(request.error);
  });
}

function getWeekKey(dateString) {
  const date = new Date(`${dateString}T00:00:00`);
  const day = (date.getDay() + 6) % 7;
  date.setDate(date.getDate() - day + 3);
  const firstThursday = new Date(date.getFullYear(), 0, 4);
  const firstDay = (firstThursday.getDay() + 6) % 7;
  firstThursday.setDate(firstThursday.getDate() - firstDay + 3);
  const weekNumber = 1 + Math.round((date - firstThursday) / 604800000);
  return `${date.getFullYear()}-W${String(weekNumber).padStart(2, "0")}`;
}

function getWeekDateRangeLabel(weekKey) {
  const [yearText, weekText] = weekKey.split("-W");
  const year = Number(yearText);
  const week = Number(weekText);
  const jan4 = new Date(year, 0, 4);
  const day = (jan4.getDay() + 6) % 7;
  const monday = new Date(jan4);
  monday.setDate(jan4.getDate() - day + (week - 1) * 7);
  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);
  return `${formatDate(toDateInputValue(monday), true)} bis ${formatDate(toDateInputValue(sunday), true)}`;
}

function formatDate(dateString, compact = false) {
  const date = new Date(`${dateString}T00:00:00`);
  return new Intl.DateTimeFormat("de-DE", {
    weekday: compact ? undefined : "long",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  }).format(date);
}

function toDateInputValue(date) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function getWeekRangeCompact(weekKey) {
  const [yearText, weekText] = weekKey.split("-W");
  const year = Number(yearText);
  const week = Number(weekText);
  const jan4 = new Date(year, 0, 4);
  const firstDay = (jan4.getDay() + 6) % 7;
  const monday = new Date(jan4);
  monday.setDate(jan4.getDate() - firstDay + (week - 1) * 7);
  const friday = new Date(monday);
  friday.setDate(monday.getDate() + 4);
  const fmt = (d) =>
    new Intl.DateTimeFormat("de-DE", { day: "2-digit", month: "2-digit" }).format(d);
  return `${fmt(monday)} – ${fmt(friday)}`;
}

function formatDateCompact(dateString) {
  const date = new Date(`${dateString}T00:00:00`);
  return new Intl.DateTimeFormat("de-DE", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
  }).format(date);
}
