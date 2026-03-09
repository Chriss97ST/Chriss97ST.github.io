const API = "http://192.168.221.88:8000"

const ui = document.getElementById("ui")
const game = document.getElementById("game")
const pauseMenu = document.getElementById("pauseMenu")
const inventoryUI = document.getElementById("inventoryUI")
const mobileControls = document.getElementById("mobileControls")
const touchPad = document.getElementById("touchPad")
const touchKnob = document.getElementById("touchKnob")
const touchInteract = document.getElementById("touchInteract")
const touchInventory = document.getElementById("touchInventory")
const touchSprint = document.getElementById("touchSprint")
const touchJump = document.getElementById("touchJump")
const touchChat = document.getElementById("touchChat")
const touchPause = document.getElementById("touchPause")

const authMessage = document.getElementById("authMessage")
const tabLogin = document.getElementById("tabLogin")
const tabRegister = document.getElementById("tabRegister")
const loginForm = document.getElementById("loginForm")
const registerForm = document.getElementById("registerForm")
const rememberLogin = document.getElementById("rememberLogin")

let camera
let renderer
let clock
let keys = {}
let gameActive = false
const touchMove = { x: 0, y: 0, active: false }
window.touchMove = touchMove
window.touchSprintActive = false
let touchPointerId = null
let interactHoldTimer = null

const REMEMBER_LOGIN_KEY = "animal_mmo_remember_login"

function loadRememberedLogin() {
  try {
    const raw = localStorage.getItem(REMEMBER_LOGIN_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || !parsed.username || !parsed.password) return null
    return parsed
  } catch {
    return null
  }
}

function saveRememberedLogin(username, password) {
  localStorage.setItem(REMEMBER_LOGIN_KEY, JSON.stringify({ username, password }))
}

function clearRememberedLogin() {
  localStorage.removeItem(REMEMBER_LOGIN_KEY)
}

function isTouchDevice() {
  return (
    window.matchMedia("(hover: none) and (pointer: coarse)").matches ||
    "ontouchstart" in window ||
    (navigator && navigator.maxTouchPoints > 0)
  )
}

function toggleInventory() {
  if (inventoryUI.style.display === "none") {
    closeWorkbenchCrafting()
  }
  inventoryUI.style.display = inventoryUI.style.display === "none" ? "block" : "none"
  if (inventoryUI.style.display === "none") {
    closeItemMenu()
    closeWorkbenchCrafting()
  }
}

function updateTouchKnob(xNorm, yNorm) {
  if (!touchKnob) return
  const maxOffset = 36
  const px = xNorm * maxOffset
  const py = yNorm * maxOffset
  touchKnob.style.transform = `translate(${px}px, ${py}px)`
}

function setTouchMoveByEvent(event) {
  const rect = touchPad.getBoundingClientRect()
  const cx = rect.left + rect.width / 2
  const cy = rect.top + rect.height / 2
  const dx = event.clientX - cx
  const dy = event.clientY - cy
  const maxDist = rect.width * 0.32
  const dist = Math.hypot(dx, dy)

  let xNorm = 0
  let yNorm = 0

  if (dist > 0) {
    const clamped = Math.min(dist, maxDist)
    xNorm = (dx / dist) * (clamped / maxDist)
    yNorm = (dy / dist) * (clamped / maxDist)
  }

  touchMove.x = xNorm
  touchMove.y = yNorm
  touchMove.active = true
  updateTouchKnob(xNorm, yNorm)
}

function resetTouchMove() {
  touchMove.x = 0
  touchMove.y = 0
  touchMove.active = false
  updateTouchKnob(0, 0)
}

function setupTouchControls() {
  if (!touchPad || !touchKnob || !touchInteract || !touchInventory || !touchSprint || !touchJump || !touchChat || !touchPause) {
    return
  }

  touchPad.addEventListener("pointerdown", (event) => {
    touchPointerId = event.pointerId
    touchPad.setPointerCapture(event.pointerId)
    setTouchMoveByEvent(event)
  })

  touchPad.addEventListener("pointermove", (event) => {
    if (touchPointerId !== event.pointerId) return
    setTouchMoveByEvent(event)
  })

  touchPad.addEventListener("pointerup", (event) => {
    if (touchPointerId !== event.pointerId) return
    touchPointerId = null
    resetTouchMove()
  })

  touchPad.addEventListener("pointercancel", (event) => {
    if (touchPointerId !== event.pointerId) return
    touchPointerId = null
    resetTouchMove()
  })

  touchInteract.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    interact()
    if (interactHoldTimer) clearInterval(interactHoldTimer)
    interactHoldTimer = setInterval(() => {
      if (!gameActive) return
      interact()
    }, 230)
  })

  const stopInteractHold = () => {
    if (!interactHoldTimer) return
    clearInterval(interactHoldTimer)
    interactHoldTimer = null
  }

  touchInteract.addEventListener("pointerup", stopInteractHold)
  touchInteract.addEventListener("pointercancel", stopInteractHold)
  touchInteract.addEventListener("pointerleave", stopInteractHold)

  touchInventory.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    toggleInventory()
  })

  const setSprint = (active) => {
    window.touchSprintActive = active
    touchSprint.classList.toggle("active", active)
  }

  touchSprint.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    setSprint(true)
  })

  touchSprint.addEventListener("pointerup", () => setSprint(false))
  touchSprint.addEventListener("pointercancel", () => setSprint(false))
  touchSprint.addEventListener("pointerleave", () => setSprint(false))

  touchJump.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    queueJump()
  })

  touchChat.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    openChatInput()
  })

  touchPause.addEventListener("pointerdown", (event) => {
    event.preventDefault()
    if (!gameActive) return
    pauseMenu.style.display = "block"
  })
}

function setAuthMode(mode) {
  const loginActive = mode === "login"

  loginForm.classList.toggle("active", loginActive)
  registerForm.classList.toggle("active", !loginActive)

  tabLogin.classList.toggle("active", loginActive)
  tabRegister.classList.toggle("active", !loginActive)

  tabLogin.setAttribute("aria-selected", String(loginActive))
  tabRegister.setAttribute("aria-selected", String(!loginActive))

  clearAuthMessage()
}

function clearAuthMessage() {
  authMessage.textContent = ""
  authMessage.className = "auth-message"
}

function showAuthMessage(text, type = "info") {
  authMessage.textContent = text
  authMessage.className = `auth-message ${type}`
}

async function request(path, payload) {
  const response = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })

  return response.json()
}

async function register(username, password) {
  const data = await request("/register", { username, password })

  if (data.status !== "ok") {
    if (data.message === "user_exists") {
      showAuthMessage("Username ist bereits vergeben.", "error")
      return
    }

    showAuthMessage("Registrierung fehlgeschlagen. Bitte Eingaben prüfen.", "error")
    return
  }

  showAuthMessage("Account erstellt. Jetzt einloggen.", "success")
  setAuthMode("login")

  document.getElementById("loginUser").value = username
  document.getElementById("loginPass").value = ""
  registerForm.reset()
}

async function login(username, password, rememberChoice = null) {
  const data = await request("/login", { username, password })

  if (data.status !== "ok") {
    if (rememberChoice === true) {
      clearRememberedLogin()
      if (rememberLogin) rememberLogin.checked = false
    }
    showAuthMessage("Login fehlgeschlagen. Username oder Passwort falsch.", "error")
    return
  }

  if (rememberChoice === true) {
    saveRememberedLogin(username, password)
  }

  if (rememberChoice === false) {
    clearRememberedLogin()
  }

  ui.style.display = "none"
  game.style.display = "block"
  gameActive = true
  if (isTouchDevice() && mobileControls) {
    mobileControls.classList.add("active")
  }

  init3D(data)
}

function init3D(data) {
  scene = new THREE.Scene()
  clock = new THREE.Clock()

  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  )

  renderer = new THREE.WebGLRenderer({ canvas: game, antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(10, 20, 10)
  scene.add(light)

  const ambient = new THREE.AmbientLight(0xffffff, 0.55)
  scene.add(ambient)

  createGround()

  player = createPlayerModel()
  player.position.set(data.x, data.y, data.z)
  scene.add(player)

  spawnAnimals()

  inventory = Array.isArray(data.inventory) ? data.inventory : []
  updateInventory()

  connectWS(data.id)
  animate()
}

function cameraFollow() {
  const target = new THREE.Vector3(
    player.position.x,
    player.position.y + 4,
    player.position.z + 8
  )

  camera.position.lerp(target, 0.08)
  camera.lookAt(player.position)
}

function animate() {
  requestAnimationFrame(animate)

  const delta = clock ? clock.getDelta() : 0.016

  move()
  physics()
  updatePlayerAnimation(delta)
  updateRemotePlayers(delta)
  updateAnimals(delta)

  cameraFollow()
  sendPos()

  renderer.render(scene, camera)
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && gameActive) {
    e.preventDefault()
    setOnlineUsersVisible(true)
  }

  if (typeof chatInput !== "undefined" && document.activeElement === chatInput) {
    return
  }

  keys[e.key] = true
  keys[e.code] = true

  if (e.key === "i") {
    toggleInventory()
  }

  if (e.key === "e") {
    interact()
  }

  if (e.key === "Escape") {
    pauseMenu.style.display = "block"
  }

  if (e.key === "t" || e.key === "T") {
    e.preventDefault()
    openChatInput()
  }
})

document.addEventListener("keyup", (e) => {
  if (e.key === "Tab" && gameActive) {
    e.preventDefault()
    setOnlineUsersVisible(false)
  }

  if (typeof chatInput !== "undefined" && document.activeElement === chatInput) {
    return
  }

  keys[e.key] = false
  keys[e.code] = false
})

game.addEventListener("pointerdown", (event) => {
  if (event.pointerType === "mouse" && event.button !== 0) return
  if (ui.style.display !== "none") return

  if (event.pointerType === "touch" && tryPickupByTouch(event, camera, game)) {
    return
  }

  handleWorldLeftClick(event, camera, game)
})

window.addEventListener("resize", () => {
  if (!renderer || !camera) return

  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault()

  const username = document.getElementById("loginUser").value.trim()
  const password = document.getElementById("loginPass").value

  if (!username || !password) {
    showAuthMessage("Bitte Username und Passwort eingeben.", "error")
    return
  }

  const keepLoggedIn = rememberLogin ? rememberLogin.checked : false
  await login(username, password, keepLoggedIn)
})

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault()

  const username = document.getElementById("registerUser").value.trim()
  const password = document.getElementById("registerPass").value
  const repeat = document.getElementById("registerPass2").value

  if (password !== repeat) {
    showAuthMessage("Passwörter stimmen nicht überein.", "error")
    return
  }

  await register(username, password)
})

tabLogin.addEventListener("click", () => setAuthMode("login"))
tabRegister.addEventListener("click", () => setAuthMode("register"))
setupTouchControls()
resetTouchMove()

const remembered = loadRememberedLogin()
if (remembered) {
  document.getElementById("loginUser").value = remembered.username
  document.getElementById("loginPass").value = remembered.password
  if (rememberLogin) rememberLogin.checked = true
  login(remembered.username, remembered.password, true)
}

function resume() {
  pauseMenu.style.display = "none"
}
