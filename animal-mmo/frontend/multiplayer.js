let ws
let players = {}
let myPlayerId = null

const remotePlayers = new Map()

const chatUI = document.getElementById("chatUI")
const chatMessages = document.getElementById("chatMessages")
const chatInput = document.getElementById("chatInput")
const onlineUsers = document.getElementById("onlineUsers")

let chatVisibleTimeout = null
let chatInputActive = false
let typingPulse = 0
let lastPlayerInteractSent = 0

function createNameTag(text) {
  const canvas = document.createElement("canvas")
  canvas.width = 256
  canvas.height = 64
  const ctx = canvas.getContext("2d")

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = "rgba(8, 16, 25, 0.8)"
  ctx.fillRect(0, 8, canvas.width, 48)
  ctx.strokeStyle = "rgba(153, 197, 235, 0.7)"
  ctx.strokeRect(0.5, 8.5, canvas.width - 1, 47)
  ctx.fillStyle = "#dff2ff"
  ctx.font = "bold 24px Segoe UI"
  ctx.textAlign = "center"
  ctx.fillText(text, canvas.width / 2, 40)

  const texture = new THREE.CanvasTexture(canvas)
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set(2.4, 0.6, 1)
  sprite.position.set(0, 2.25, 0)
  return sprite
}

function createRemotePlayer(name) {
  const group = createPlayerModel()

  const tint = new THREE.Color(0xdabf6a)
  for (const child of group.children) {
    if (child.isMesh && child.material && child.material.color) {
      child.material = child.material.clone()
      child.material.color.lerp(tint, 0.18)
    }
  }

  const nameTag = createNameTag(name)
  group.add(nameTag)

  const typingTag = createNameTag("...")
  typingTag.scale.set(1.1, 0.46, 1)
  typingTag.position.set(0, 3.0, 0)
  typingTag.visible = false
  group.add(typingTag)

  scene.add(group)

  return {
    mesh: group,
    target: new THREE.Vector3(),
    moving: false,
    nameTag,
    typingTag,
    typing: false
  }
}

function updateTagText(sprite, text) {
  const canvas = sprite.material.map.image
  const ctx = canvas.getContext("2d")
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = "rgba(8, 16, 25, 0.8)"
  ctx.fillRect(0, 8, canvas.width, 48)
  ctx.strokeStyle = "rgba(153, 197, 235, 0.7)"
  ctx.strokeRect(0.5, 8.5, canvas.width - 1, 47)
  ctx.fillStyle = "#dff2ff"
  ctx.font = "bold 24px Segoe UI"
  ctx.textAlign = "center"
  ctx.fillText(text, canvas.width / 2, 40)
  sprite.material.map.needsUpdate = true
}

function updateOnlineUsersPanel() {
  if (!onlineUsers) return

  const entries = Object.entries(players || {}).sort((a, b) => Number(a[0]) - Number(b[0]))
  onlineUsers.innerHTML = ""

  const title = document.createElement("div")
  title.className = "online-users-title"
  title.textContent = `Online: ${entries.length}`
  onlineUsers.appendChild(title)

  for (const [uid, info] of entries) {
    const line = document.createElement("div")
    line.className = "online-user-line"

    const left = document.createElement("span")
    left.textContent = uid == myPlayerId ? `${info.name} (Du)` : info.name

    const right = document.createElement("span")
    right.className = "online-user-typing"
    right.textContent = info.typing ? "schreibt..." : ""

    line.appendChild(left)
    line.appendChild(right)
    onlineUsers.appendChild(line)
  }
}

function setOnlineUsersVisible(visible) {
  if (!onlineUsers) return
  onlineUsers.classList.toggle("visible", visible)
}

function sendTypingStatus(active) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "typing",
    active: Boolean(active)
  }))
}

function updateRemotePlayersSnapshot(snapshot) {
  const seen = new Set()

  for (const [uidRaw, info] of Object.entries(snapshot || {})) {
    const uid = Number(uidRaw)
    if (uid === myPlayerId) continue

    seen.add(uid)

    let entry = remotePlayers.get(uid)
    if (!entry) {
      entry = createRemotePlayer(info.name || `Player${uid}`)
      entry.mesh.position.set(info.x || 0, info.y || 1, info.z || 0)
      remotePlayers.set(uid, entry)
    }

    entry.target.set(info.x || 0, info.y || 1, info.z || 0)
    entry.typing = Boolean(info.typing)
    entry.typingTag.visible = entry.typing
  }

  for (const [uid, entry] of remotePlayers.entries()) {
    if (seen.has(uid)) continue
    scene.remove(entry.mesh)
    remotePlayers.delete(uid)
  }

  updateOnlineUsersPanel()
}

function updateRemotePlayers(delta) {
  typingPulse += delta * 3.4
  const dotPhase = Math.floor(typingPulse % 3) + 1
  const dots = ".".repeat(dotPhase)

  for (const entry of remotePlayers.values()) {
    const before = entry.mesh.position.clone()
    entry.mesh.position.lerp(entry.target, Math.min(1, delta * 12))

    const moveDist = entry.mesh.position.distanceTo(before)
    entry.moving = moveDist > 0.002

    const dx = entry.target.x - entry.mesh.position.x
    const dz = entry.target.z - entry.mesh.position.z
    if (Math.abs(dx) + Math.abs(dz) > 0.01) {
      entry.mesh.rotation.y = Math.atan2(dx, dz)
    }

    if (entry.mesh.userData && entry.mesh.userData.anim) {
      const anim = entry.mesh.userData.anim
      const speed = entry.moving ? 12 : 3
      anim.cycle += delta * speed
      const walk = entry.moving ? Math.sin(anim.cycle) : Math.sin(anim.cycle) * 0.1
      const swing = entry.moving ? 0.9 : 0.06

      anim.legL.rotation.x = walk * swing
      anim.legR.rotation.x = -walk * swing
      anim.armL.rotation.x = -walk * swing * 0.8
      anim.armR.rotation.x = walk * swing * 0.8
    }

    entry.typingTag.visible = entry.typing
    if (entry.typing) {
      updateTagText(entry.typingTag, dots)
    }
  }
}

function setChatVisible(visible) {
  chatUI.classList.toggle("visible", visible)
}

function scheduleChatAutoHide() {
  if (chatVisibleTimeout) {
    clearTimeout(chatVisibleTimeout)
  }

  if (chatInputActive) return

  chatVisibleTimeout = setTimeout(() => {
    setChatVisible(false)
  }, 5000)
}

function appendChatMessage(name, text) {
  const line = document.createElement("div")
  line.className = "chat-line"

  const nameNode = document.createElement("span")
  nameNode.className = "chat-name"
  nameNode.textContent = name

  const textNode = document.createElement("span")
  textNode.textContent = text

  line.appendChild(nameNode)
  line.appendChild(textNode)
  chatMessages.appendChild(line)

  while (chatMessages.children.length > 30) {
    chatMessages.removeChild(chatMessages.firstChild)
  }

  chatMessages.scrollTop = chatMessages.scrollHeight

  setChatVisible(true)
  scheduleChatAutoHide()
}

function openChatInput() {
  chatInputActive = true
  chatUI.classList.add("active")
  setChatVisible(true)
  sendTypingStatus(true)

  if (chatVisibleTimeout) {
    clearTimeout(chatVisibleTimeout)
    chatVisibleTimeout = null
  }

  chatInput.focus()
}

function closeChatInput() {
  chatInputActive = false
  chatUI.classList.remove("active")
  chatInput.blur()
  sendTypingStatus(false)
  scheduleChatAutoHide()
}

function sendChatMessage(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "chat",
    text
  }))
}

function requestPlayerInteract() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "player_interact",
    data: {
      x: player.position.x,
      y: player.position.y,
      z: player.position.z
    }
  }))
}

function connectWS(id) {
  myPlayerId = Number(id)
  ws = new WebSocket(`wss://chriss97st.ddns.net/animmo/api/ws/${id}`)
  let disconnected = false

  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data)

    if (data.type === "players") {
      players = data.players
      updateRemotePlayersSnapshot(players)
    }

    if (data.type === "world_snapshot") {
      applyWorldSnapshot(data)
      applyAnimalsSnapshot(data.animals || [])
    }

    if (data.type === "world_patch") {
      applyWorldPatch(data)
    }

    if (data.type === "inventory_update") {
      inventory = Array.isArray(data.inventory) ? data.inventory : []
      updateInventory()
    }

    if (data.type === "chat") {
      appendChatMessage(data.name || "Player", data.text || "")
    }

    if (data.type === "animals") {
      applyAnimalsSnapshot(data.animals || [])
    }
  }

  const handleDisconnect = () => {
    if (disconnected) return
    disconnected = true

    players = {}
    updateRemotePlayersSnapshot({})
    clearServerAnimals()

    if (typeof gameActive !== "undefined") gameActive = false
    if (typeof mobileControls !== "undefined" && mobileControls) {
      mobileControls.classList.remove("active")
    }

    if (typeof game !== "undefined" && game) {
      game.style.display = "none"
    }

    if (typeof ui !== "undefined" && ui) {
      ui.style.display = "flex"
    }

    if (typeof showAuthMessage === "function") {
      showAuthMessage("Verbindung zum Server getrennt. Bitte erneut einloggen.", "error")
    }
  }

  ws.onclose = handleDisconnect
  ws.onerror = handleDisconnect
}

function sendPos() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "position",
    data: {
      x: player.position.x,
      y: player.position.y,
      z: player.position.z
    }
  }))
}

function requestActionE() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "action_e",
    data: {
      x: player.position.x,
      y: player.position.y,
      z: player.position.z
    }
  }))

  const now = Date.now()
  if (now - lastPlayerInteractSent > 700) {
    lastPlayerInteractSent = now
    requestPlayerInteract()
  }
}

function requestChopTree(objectId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "action_chop_tree",
    data: {
      object_id: objectId,
      x: player.position.x,
      y: player.position.y,
      z: player.position.z
    }
  }))
}

function requestMineRock(objectId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "action_mine_rock",
    data: {
      object_id: objectId,
      x: player.position.x,
      y: player.position.y,
      z: player.position.z
    }
  }))
}

function requestPlaceWorkbench() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "action_place_workbench",
    data: {
      x: player.position.x,
      y: player.position.y,
      z: player.position.z,
      facing: player.rotation.y
    }
  }))
}

function requestInventoryAction(action) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "inventory_action",
    action
  }))
}

function requestInventoryTransform(consumes, produces) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "inventory_transform",
    consumes,
    produces
  }))
}

function saveGame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return

  ws.send(JSON.stringify({
    type: "save",
    data: {
      x: player.position.x,
      y: player.position.y,
      z: player.position.z,
      inventory: inventory
    }
  }))

  alert("Game saved")
}

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault()
    const text = chatInput.value.trim()
    if (text.length > 0) {
      sendChatMessage(text)
    }
    chatInput.value = ""
    closeChatInput()
  }

  if (event.key === "Escape") {
    event.preventDefault()
    chatInput.value = ""
    closeChatInput()
  }
})

chatInput.addEventListener("blur", () => {
  if (chatInputActive) {
    closeChatInput()
  }
})
