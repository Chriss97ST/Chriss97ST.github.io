let player
let velocityY = 0
let gravity = -0.02
let onGround = false
let playerIsMoving = false
let jumpQueued = false

let inventory = []
let workbenchCraftOpen = false
let workbenchIgnoreOutsideClickUntil = 0
let openChestId = 0
let openChestItems = []
let chestIgnoreOutsideClickUntil = 0
let draggedInventoryItem = ""
const craftSlots2x2 = Array(4).fill(null)
const craftSlots3x3 = Array(9).fill(null)
let equippedItem = ""

const itemMeta = {
  fruit: { key: "frucht", label: "Frucht", icon: "🍎" },
  frucht: { key: "frucht", label: "Frucht", icon: "🍎" },
  log: { key: "holzstamm", label: "Holzstamm", icon: "🪵" },
  holzstamm: { key: "holzstamm", label: "Holzstamm", icon: "🪵" },
  holzlatte: { key: "holzlatte", label: "Holzlatte", icon: "🪚" },
  stein: { key: "stein", label: "Stein", icon: "🪨" },
  werkbank: { key: "werkbank", label: "Werkbank", icon: "🧰" },
  kiste: { key: "kiste", label: "Kiste", icon: "🧺" },
  stock: { key: "stock", label: "Stock", icon: "🪵" },
  holzspitzhacke: { key: "holzspitzhacke", label: "Holzspitzhacke", icon: "⛏️" },
  steinspitzhacke: { key: "steinspitzhacke", label: "Steinspitzhacke", icon: "⛏️" },
  holzaxt: { key: "holzaxt", label: "Holzaxt", icon: "🪓" },
  eimer_leer: { key: "eimer_leer", label: "Eimer (leer)", icon: "🪣" },
  eimer_voll: { key: "eimer_voll", label: "Eimer (voll)", icon: "🪣" },
  tree_seed: { key: "baumsamen", label: "Baumsamen", icon: "🌱" },
  treeseed: { key: "baumsamen", label: "Baumsamen", icon: "🌱" },
  baumsamen: { key: "baumsamen", label: "Baumsamen", icon: "🌱" }
}

const workbenchRecipes = [
  {
    id: "craft_sticks",
    label: "4x Stock",
    pattern: [null, "holzlatte", null, null, "holzlatte", null, null, null, null],
    consumes: [{ item: "holzlatte", amount: 2 }],
    produces: [{ item: "stock", amount: 4 }]
  },
  {
    id: "craft_wood_pickaxe",
    label: "Holzspitzhacke",
    pattern: ["holzlatte", "holzlatte", "holzlatte", null, "stock", null, null, "stock", null],
    consumes: [{ item: "holzlatte", amount: 3 }, { item: "stock", amount: 2 }],
    produces: [{ item: "holzspitzhacke", amount: 1 }]
  },
  {
    id: "craft_stone_pickaxe",
    label: "Steinspitzhacke",
    pattern: ["stein", "stein", "stein", null, "stock", null, null, "stock", null],
    consumes: [{ item: "stein", amount: 3 }, { item: "stock", amount: 2 }],
    produces: [{ item: "steinspitzhacke", amount: 1 }]
  },
  {
    id: "craft_wood_axe",
    label: "Holzaxt",
    pattern: ["holzlatte", "holzlatte", null, "holzlatte", "stock", null, null, "stock", null],
    consumes: [{ item: "holzlatte", amount: 3 }, { item: "stock", amount: 2 }],
    produces: [{ item: "holzaxt", amount: 1 }]
  },
  {
    id: "craft_wood_axe_mirror",
    label: "Holzaxt",
    pattern: [null, "holzlatte", "holzlatte", null, "stock", "holzlatte", null, "stock", null],
    consumes: [{ item: "holzlatte", amount: 3 }, { item: "stock", amount: 2 }],
    produces: [{ item: "holzaxt", amount: 1 }]
  },
  {
    id: "craft_bucket",
    label: "Eimer (leer)",
    pattern: ["holzlatte", null, "holzlatte", null, "stein", null, null, null, null],
    consumes: [{ item: "holzlatte", amount: 2 }, { item: "stein", amount: 1 }],
    produces: [{ item: "eimer_leer", amount: 1 }]
  },
  {
    id: "craft_chest",
    label: "Kiste",
    pattern: ["holzlatte", "holzlatte", "holzlatte", "holzlatte", null, "holzlatte", "holzlatte", "holzlatte", "holzlatte"],
    consumes: [{ item: "holzlatte", amount: 8 }],
    produces: [{ item: "kiste", amount: 1 }]
  }
]

const equippableItems = new Set(["holzspitzhacke", "steinspitzhacke", "holzaxt", "eimer_leer", "eimer_voll"])

function getEquippedItem() {
  return equippedItem
}

function createHeldItemMesh(item) {
  if (item === "holzspitzhacke" || item === "steinspitzhacke") {
    const group = new THREE.Group()
    const handle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.04, 0.55, 8),
      new THREE.MeshStandardMaterial({ color: 0x8b5a2b, flatShading: true })
    )
    handle.rotation.z = Math.PI * 0.35
    const head = new THREE.Mesh(
      new THREE.BoxGeometry(0.34, 0.11, 0.09),
      new THREE.MeshStandardMaterial({ color: item === "steinspitzhacke" ? 0x8a949e : 0xc9a67b, flatShading: true })
    )
    head.position.set(0.17, 0.16, 0)
    head.rotation.z = Math.PI * 0.35
    group.add(handle)
    group.add(head)
    return group
  }

  if (item === "holzaxt") {
    const group = new THREE.Group()
    const handle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.04, 0.55, 8),
      new THREE.MeshStandardMaterial({ color: 0x8b5a2b, flatShading: true })
    )
    handle.rotation.z = Math.PI * 0.22
    const blade = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.16, 0.08),
      new THREE.MeshStandardMaterial({ color: 0xbec7d2, flatShading: true })
    )
    blade.position.set(0.12, 0.2, 0)
    blade.rotation.z = Math.PI * 0.35
    group.add(handle)
    group.add(blade)
    return group
  }

  if (item === "eimer_leer" || item === "eimer_voll") {
    const group = new THREE.Group()
    const bucket = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.16, 0.2, 10, 1, true),
      new THREE.MeshStandardMaterial({ color: 0x949ba5, flatShading: true, side: THREE.DoubleSide })
    )
    bucket.rotation.x = Math.PI * 0.15
    group.add(bucket)

    if (item === "eimer_voll") {
      const water = new THREE.Mesh(
        new THREE.CylinderGeometry(0.118, 0.118, 0.03, 10),
        new THREE.MeshStandardMaterial({ color: 0x2f84c7, flatShading: true })
      )
      water.position.y = 0.04
      group.add(water)
    }

    return group
  }

  return null
}

function setModelHeldItem(model, item) {
  if (!model || !model.userData || !model.userData.handAnchor) return

  if (model.userData.heldItemMesh) {
    model.userData.handAnchor.remove(model.userData.heldItemMesh)
    model.userData.heldItemMesh = null
  }

  const normalized = normalizeItem(item)
  if (!normalized || !equippableItems.has(normalized)) {
    model.userData.heldItem = ""
    return
  }

  const mesh = createHeldItemMesh(normalized)
  if (!mesh) {
    model.userData.heldItem = ""
    return
  }

  if (normalized === "holzspitzhacke" || normalized === "steinspitzhacke" || normalized === "holzaxt") {
    // Flip tool orientation so the hand grips the handle end, not the tool head.
    mesh.position.set(0.02, -0.3, 0.03)
    mesh.rotation.set(0.2, -0.15, Math.PI + 0.5)
  } else {
    mesh.position.set(0.08, -0.34, 0.03)
    mesh.rotation.set(0.2, -0.15, 0.5)
  }
  model.userData.handAnchor.add(mesh)
  model.userData.heldItemMesh = mesh
  model.userData.heldItem = normalized
}

function setEquippedItem(item, syncInventory = true) {
  const normalized = normalizeItem(item)
  equippedItem = equippableItems.has(normalized) ? normalized : ""
  setModelHeldItem(player, equippedItem)
  if (syncInventory) {
    updateInventory()
  }
}

function normalizeItem(item) {
  return itemMeta[item]?.key || item
}

function prettyItemName(item) {
  return itemMeta[item]?.label || item
}

function itemIcon(item) {
  return itemMeta[item]?.icon || "📦"
}

const allowedEmotes = new Set(["smile", "happy", "angry", "sad", "wink", "surprised"])

function normalizeEmote(emote) {
  const key = String(emote || "smile").trim().toLowerCase()
  return allowedEmotes.has(key) ? key : "smile"
}

function drawFaceEmote(ctx, emote) {
  const e = normalizeEmote(emote)
  const w = ctx.canvas.width
  const h = ctx.canvas.height

  ctx.clearRect(0, 0, w, h)

  ctx.fillStyle = "#ffd54f"
  ctx.beginPath()
  ctx.arc(64, 64, 58, 0, Math.PI * 2)
  ctx.fill()

  ctx.fillStyle = "#2b2b2b"

  if (e === "wink") {
    ctx.beginPath()
    ctx.arc(46, 52, 9, 0, Math.PI * 2)
    ctx.fill()

    ctx.strokeStyle = "#2b2b2b"
    ctx.lineWidth = 8
    ctx.lineCap = "round"
    ctx.beginPath()
    ctx.moveTo(74, 52)
    ctx.lineTo(89, 52)
    ctx.stroke()
  } else if (e === "surprised") {
    ctx.beginPath()
    ctx.arc(46, 50, 9, 0, Math.PI * 2)
    ctx.arc(82, 50, 9, 0, Math.PI * 2)
    ctx.fill()
  } else if (e === "sad") {
    ctx.beginPath()
    ctx.arc(46, 50, 8, 0, Math.PI * 2)
    ctx.arc(82, 50, 8, 0, Math.PI * 2)
    ctx.fill()
  } else {
    ctx.beginPath()
    ctx.arc(46, 52, 9, 0, Math.PI * 2)
    ctx.arc(82, 52, 9, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.strokeStyle = e === "angry" ? "#7a1414" : "#2b2b2b"
  ctx.lineWidth = 10
  ctx.lineCap = "round"

  if (e === "surprised") {
    ctx.beginPath()
    ctx.arc(64, 76, 11, 0, Math.PI * 2)
    ctx.stroke()
    return
  }

  if (e === "sad") {
    ctx.beginPath()
    ctx.arc(64, 90, 20, 1.1 * Math.PI, 1.9 * Math.PI)
    ctx.stroke()
    return
  }

  if (e === "happy") {
    ctx.beginPath()
    ctx.arc(64, 66, 30, 0.1 * Math.PI, 0.9 * Math.PI)
    ctx.stroke()
    return
  }

  if (e === "angry") {
    ctx.lineWidth = 7
    ctx.beginPath()
    ctx.moveTo(34, 38)
    ctx.lineTo(49, 45)
    ctx.moveTo(94, 38)
    ctx.lineTo(79, 45)
    ctx.stroke()
    ctx.lineWidth = 10
    ctx.beginPath()
    ctx.arc(64, 85, 19, 1.15 * Math.PI, 1.85 * Math.PI)
    ctx.stroke()
    return
  }

  ctx.beginPath()
  ctx.arc(64, 70, 26, 0.15 * Math.PI, 0.85 * Math.PI)
  ctx.stroke()
}

function createFaceTexture(emote = "smile") {
  const canvas = document.createElement("canvas")
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext("2d")
  drawFaceEmote(ctx, emote)

  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return { texture, canvas, ctx }
}

function setModelFaceEmote(model, emote) {
  if (!model || !model.userData || !model.userData.faceCtx || !model.userData.faceTexture) return
  const normalized = normalizeEmote(emote)
  if (model.userData.faceEmote === normalized) return
  drawFaceEmote(model.userData.faceCtx, normalized)
  model.userData.faceTexture.needsUpdate = true
  model.userData.faceEmote = normalized
}

function setPlayerEmote(emote) {
  if (!player) return
  setModelFaceEmote(player, emote)
}

function normalizeGender(gender) {
  return String(gender || "male").toLowerCase() === "female" ? "female" : "male"
}

function createPlayerModel(gender = "male") {
  const safeGender = normalizeGender(gender)
  const group = new THREE.Group()
  const skin = new THREE.MeshStandardMaterial({ color: 0xffcc99, flatShading: true })
  const cloth = new THREE.MeshStandardMaterial({ color: safeGender === "female" ? 0xffb6d9 : 0xb8d7ff, flatShading: true })
  const dark = new THREE.MeshStandardMaterial({ color: 0x23324a, flatShading: true })
  const shoulderOffset = safeGender === "female" ? 0.58 : 0.66
  const hipOffset = safeGender === "female" ? 0.28 : 0.23

  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1.25, 0.55), cloth)
  torso.position.y = 0.2
  if (safeGender === "female") {
    torso.scale.x = 0.9
  }

  const head = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.62, 0.62), skin)
  head.position.y = 1.18

  const faceData = createFaceTexture("smile")
  const faceMaterial = new THREE.MeshBasicMaterial({
    map: faceData.texture,
    transparent: true
  })
  const face = new THREE.Mesh(new THREE.PlaneGeometry(0.58, 0.58), faceMaterial)
  face.position.z = 0.33
  head.add(face)

  const armL = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.9, 0.24), skin)
  armL.position.set(-shoulderOffset, 0.14, 0)

  const armR = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.9, 0.24), skin)
  armR.position.set(shoulderOffset, 0.14, 0)

  const handAnchor = new THREE.Group()
  handAnchor.position.set(0.02, -0.38, 0.08)
  armR.add(handAnchor)

  const legL = new THREE.Mesh(new THREE.BoxGeometry(0.28, 1.0, 0.3), dark)
  legL.position.set(-hipOffset, -0.86, 0)

  const legR = new THREE.Mesh(new THREE.BoxGeometry(0.28, 1.0, 0.3), dark)
  legR.position.set(hipOffset, -0.86, 0)

  if (safeGender === "female") {
    const hips = new THREE.Mesh(new THREE.BoxGeometry(1.03, 0.34, 0.58), cloth)
    hips.position.set(0, -0.36, 0)

    const breastMat = new THREE.MeshStandardMaterial({ color: 0xffb6d9, flatShading: true })
    const breastL = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 12), breastMat)
    const breastR = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 12), breastMat)
    breastL.scale.set(1.0, 0.9, 1.35)
    breastR.scale.set(1.0, 0.9, 1.35)
    breastL.position.set(-0.19, 0.43, 0.41)
    breastR.position.set(0.19, 0.43, 0.41)
    group.add(hips)
    group.add(breastL)
    group.add(breastR)
  }

  group.add(torso)
  group.add(head)
  group.add(armL)
  group.add(armR)
  group.add(legL)
  group.add(legR)

  group.userData.anim = {
    cycle: 0,
    armL,
    armR,
    legL,
    legR,
    torso,
    head
  }

  group.userData.faceTexture = faceData.texture
  group.userData.faceCanvas = faceData.canvas
  group.userData.faceCtx = faceData.ctx
  group.userData.faceEmote = "smile"
  group.userData.handAnchor = handAnchor
  group.userData.heldItem = ""
  group.userData.heldItemMesh = null
  group.userData.gender = safeGender

  group.userData.collisionRadius = 0.65
  return group
}

function setPlayerMovementState(isMoving) {
  playerIsMoving = isMoving
}

function updatePlayerAnimation(delta) {
  if (!player || !player.userData.anim) return

  const anim = player.userData.anim
  const speed = playerIsMoving ? 12 : 3
  anim.cycle += delta * speed

  const walk = playerIsMoving ? Math.sin(anim.cycle) : Math.sin(anim.cycle) * 0.12
  const swing = playerIsMoving ? 0.9 : 0.08

  anim.legL.rotation.x = walk * swing
  anim.legR.rotation.x = -walk * swing
  anim.armL.rotation.x = -walk * swing * 0.8
  anim.armR.rotation.x = walk * swing * 0.8

  anim.torso.rotation.z = Math.sin(anim.cycle * 0.5) * (playerIsMoving ? 0.06 : 0.02)
  anim.head.rotation.y = Math.sin(anim.cycle * 0.45) * 0.08
}

function physics() {
  if ((keys[" "] || jumpQueued) && onGround) {
    velocityY = 0.35
    onGround = false
    jumpQueued = false
  }

  velocityY += gravity
  player.position.y += velocityY

  if (player.position.y < 1) {
    player.position.y = 1
    velocityY = 0
    onGround = true
  }
}

function queueJump() {
  if (typeof isRidingHare === "function" && isRidingHare()) {
    if (typeof queueHareJump === "function") {
      queueHareJump()
      return
    }
  }
  jumpQueued = true
}

function resolvePlayerCollision(nextX, nextZ) {
  const playerRadius = player.userData.collisionRadius || 0.65
  let x = nextX
  let z = nextZ

  const blockers = [...objects]

  for (let pass = 0; pass < 2; pass++) {
    for (const blocker of blockers) {
      if (!blocker || blocker === player) continue

      const obstacleRadius = blocker.userData?.collisionRadius || 1
      const dx = x - blocker.position.x
      const dz = z - blocker.position.z
      const dist = Math.hypot(dx, dz) || 0.0001
      const minDist = playerRadius + obstacleRadius

      if (dist < minDist) {
        const push = minDist - dist
        x += (dx / dist) * push
        z += (dz / dist) * push
      }
    }
  }

  return { x, z }
}

function move() {
  let speed = 0.15
  let inputX = 0
  let inputZ = 0

  if (keys["w"]) inputZ -= 1
  if (keys["s"]) inputZ += 1
  if (keys["a"]) inputX -= 1
  if (keys["d"]) inputX += 1

  if (window.touchMove && window.touchMove.active) {
    inputX += window.touchMove.x
    inputZ += window.touchMove.y
  }

  if (keys["ShiftLeft"] || window.touchSprintActive) {
    speed *= 1.8
  }

  const inputLen = Math.hypot(inputX, inputZ)
  if (inputLen > 1) {
    inputX /= inputLen
    inputZ /= inputLen
  }

  const dx = inputX * speed
  const dz = inputZ * speed

  const intendedX = player.position.x + dx
  const intendedZ = player.position.z + dz
  const resolved = resolvePlayerCollision(intendedX, intendedZ)

  player.position.x = resolved.x
  player.position.z = resolved.z

  const isMoving = Math.abs(dx) + Math.abs(dz) > 0
  setPlayerMovementState(isMoving)

  if (isMoving) {
    player.rotation.y = Math.atan2(dx, dz)
  }
}

function getMovementInputState() {
  let inputX = 0
  let inputZ = 0

  if (keys["w"]) inputZ -= 1
  if (keys["s"]) inputZ += 1
  if (keys["a"]) inputX -= 1
  if (keys["d"]) inputX += 1

  if (window.touchMove && window.touchMove.active) {
    inputX += window.touchMove.x
    inputZ += window.touchMove.y
  }

  const inputLen = Math.hypot(inputX, inputZ)
  if (inputLen > 1) {
    inputX /= inputLen
    inputZ /= inputLen
  }

  return {
    x: inputX,
    z: inputZ,
    sprint: Boolean(keys["ShiftLeft"] || window.touchSprintActive)
  }
}

window.getMovementInputState = getMovementInputState

function interact() {
  requestActionE()
}

function getInventoryCounts() {
  const counts = new Map()

  for (const raw of inventory) {
    const item = normalizeItem(raw)
    counts.set(item, (counts.get(item) || 0) + 1)
  }

  return counts
}

function countItemInSlots(slots, item) {
  let count = 0
  for (const slotItem of slots) {
    if (slotItem === item) count += 1
  }
  return count
}

function countItemReserved(item) {
  return countItemInSlots(craftSlots2x2, item) + countItemInSlots(craftSlots3x3, item)
}

function canAssignItemToSlot(item, currentSlotItem) {
  const counts = getInventoryCounts()
  const available = counts.get(item) || 0
  const reserved = countItemReserved(item) - (currentSlotItem === item ? 1 : 0)
  return available > reserved
}

function closeItemMenu() {
  const open = inventoryUI.querySelector(".inventory-item-menu")
  if (open) open.remove()
}

function closeWorkbenchCrafting() {
  workbenchCraftOpen = false
  updateInventory()
}

function closeWorkbenchAndInventory() {
  workbenchCraftOpen = false
  workbenchIgnoreOutsideClickUntil = 0
  openChestId = 0
  openChestItems = []
  chestIgnoreOutsideClickUntil = 0
  draggedInventoryItem = ""
  closeItemMenu()
  inventoryUI.style.display = "none"
  updateInventory()
}

function openWorkbenchCrafting() {
  workbenchCraftOpen = true
  openChestId = 0
  openChestItems = []
  workbenchIgnoreOutsideClickUntil = Date.now() + 220
  inventoryUI.style.display = "block"
  updateInventory()
}

function setOpenChestState(chestId, items) {
  const nextId = Number(chestId || 0)
  if (!nextId) {
    openChestId = 0
    openChestItems = []
    updateInventory()
    return
  }

  openChestId = nextId
  openChestItems = Array.isArray(items) ? items.map((raw) => normalizeItem(raw)) : []
  workbenchCraftOpen = false
  chestIgnoreOutsideClickUntil = Date.now() + 320
  inventoryUI.style.display = "block"
  updateInventory()
}

function closeChestStorage() {
  if (!openChestId && openChestItems.length === 0) return
  openChestId = 0
  openChestItems = []
  chestIgnoreOutsideClickUntil = 0
  updateInventory()
}

function buildItemOptions(item, amount = 1) {
  const options = []

  if (item === "frucht") {
    options.push({
      label: "Essen",
      action: () => requestInventoryAction("eat_fruit")
    })
  }

  if (item === "holzstamm") {
    options.push({
      label: "Zu Holzlatten verarbeiten",
      action: () => requestInventoryAction("craft_planks")
    })

    if (amount > 1) {
      options.push({
        label: `Stack verarbeiten (x${amount})`,
        action: () => requestInventoryAction("craft_planks_all")
      })
    }
  }

  if (item === "werkbank") {
    options.push({
      label: "Aufstellen",
      action: () => requestPlaceWorkbench()
    })
  }

  if (item === "kiste") {
    options.push({
      label: "Aufstellen",
      action: () => requestPlaceChest()
    })
  }

  if (item === "baumsamen") {
    options.push({
      label: "Einpflanzen",
      action: () => requestPlantTreeSeed()
    })
  }

  if (openChestId) {
    options.push({
      label: "In Kiste legen",
      action: () => requestChestAction("store_one", openChestId, item, 1)
    })

    if (amount > 1) {
      options.push({
        label: `Stack einlagern (x${amount})`,
        action: () => requestChestAction("store_all", openChestId, item, amount)
      })
    }
  }

  if (equippableItems.has(item)) {
    if (equippedItem === item) {
      options.push({
        label: "Ablegen",
        action: () => requestInventoryAction("unequip_item")
      })
    } else {
      options.push({
        label: "Ausrüsten",
        action: () => requestInventoryAction("equip_item", { item })
      })
    }
  }

  return options
}

function openItemMenu(anchor, item, amount = 1) {
  closeItemMenu()

  const options = buildItemOptions(item, amount)
  if (options.length === 0) return

  const menu = document.createElement("div")
  menu.className = "inventory-item-menu"

  for (const option of options) {
    const button = document.createElement("button")
    button.type = "button"
    button.className = "inventory-item-menu-btn"
    button.textContent = option.label
    button.addEventListener("click", () => {
      option.action()
      closeItemMenu()
    })
    menu.appendChild(button)
  }

  anchor.appendChild(menu)
}

function createCraftSlot(slotArray, slotIndex) {
  const slot = document.createElement("div")
  slot.className = "craft-slot"

  const value = slotArray[slotIndex]
  if (value) {
    slot.textContent = itemIcon(value)
    slot.title = prettyItemName(value)
    slot.classList.add("filled")
  }

  slot.addEventListener("dragover", (event) => {
    event.preventDefault()
  })

  slot.addEventListener("drop", (event) => {
    event.preventDefault()
    const droppedRaw = event.dataTransfer.getData("text/plain")
    const dropped = normalizeItem(droppedRaw)
    const current = slotArray[slotIndex]

    if (!dropped || !canAssignItemToSlot(dropped, current)) return

    slotArray[slotIndex] = dropped
    updateInventory()
  })

  slot.addEventListener("click", () => {
    if (!slotArray[slotIndex]) return
    slotArray[slotIndex] = null
    updateInventory()
  })

  return slot
}

function renderCraft2x2Section(parent) {
  const section = document.createElement("section")
  section.className = "craft-section"

  const title = document.createElement("h4")
  title.className = "craft-title"
  title.textContent = "2x2 Crafting"

  const grid = document.createElement("div")
  grid.className = "craft-grid craft-grid-2"

  for (let i = 0; i < craftSlots2x2.length; i++) {
    grid.appendChild(createCraftSlot(craftSlots2x2, i))
  }

  const allPlanks = craftSlots2x2.every((item) => item === "holzlatte")

  const footer = document.createElement("div")
  footer.className = "craft-footer"

  const result = document.createElement("span")
  result.className = "craft-result"
  result.textContent = allPlanks ? "Ergebnis: Werkbank" : "Rezept: 4x Holzlatte"

  const btn = document.createElement("button")
  btn.type = "button"
  btn.className = "craft-btn"
  btn.textContent = "Craften"
  btn.disabled = !allPlanks
  btn.addEventListener("click", () => {
    requestInventoryTransform(
      [{ item: "holzlatte", amount: 4 }],
      [{ item: "werkbank", amount: 1 }]
    )
    craftSlots2x2.fill(null)
    updateInventory()
  })

  footer.appendChild(result)
  footer.appendChild(btn)

  section.appendChild(title)
  section.appendChild(grid)
  section.appendChild(footer)
  parent.appendChild(section)
}

function findWorkbenchRecipe() {
  for (const recipe of workbenchRecipes) {
    let match = true

    for (let i = 0; i < 9; i++) {
      const expected = recipe.pattern[i]
      const actual = craftSlots3x3[i]
      if ((expected || null) !== (actual || null)) {
        match = false
        break
      }
    }

    if (match) return recipe
  }

  return null
}

function renderWorkbenchSection(parent) {
  if (!workbenchCraftOpen) return

  const section = document.createElement("section")
  section.className = "craft-section workbench-section"

  const head = document.createElement("div")
  head.className = "workbench-head"

  const title = document.createElement("h4")
  title.className = "craft-title"
  title.textContent = "Werkbank 3x3"

  const closeBtn = document.createElement("button")
  closeBtn.type = "button"
  closeBtn.className = "craft-close-btn"
  closeBtn.textContent = "Schließen"
  closeBtn.addEventListener("click", closeWorkbenchAndInventory)

  head.appendChild(title)
  head.appendChild(closeBtn)

  const grid = document.createElement("div")
  grid.className = "craft-grid craft-grid-3"

  for (let i = 0; i < craftSlots3x3.length; i++) {
    grid.appendChild(createCraftSlot(craftSlots3x3, i))
  }

  const recipe = findWorkbenchRecipe()

  const footer = document.createElement("div")
  footer.className = "craft-footer"

  const result = document.createElement("span")
  result.className = "craft-result"
  result.textContent = recipe ? `Ergebnis: ${recipe.label}` : "Kein gültiges Rezept"

  const btn = document.createElement("button")
  btn.type = "button"
  btn.className = "craft-btn"
  btn.textContent = "Craften"
  btn.disabled = !recipe
  btn.addEventListener("click", () => {
    if (!recipe) return
    requestInventoryTransform(recipe.consumes, recipe.produces)
    craftSlots3x3.fill(null)
    updateInventory()
  })

  footer.appendChild(result)
  footer.appendChild(btn)

  section.appendChild(head)
  section.appendChild(grid)
  section.appendChild(footer)
  parent.appendChild(section)
}

function getChestCounts() {
  const counts = new Map()
  for (const raw of openChestItems) {
    const item = normalizeItem(raw)
    counts.set(item, (counts.get(item) || 0) + 1)
  }
  return counts
}

function renderChestSection(parent) {
  if (!openChestId) return

  const section = document.createElement("section")
  section.className = "craft-section chest-section"

  const head = document.createElement("div")
  head.className = "workbench-head"

  const title = document.createElement("h4")
  title.className = "craft-title"
  title.textContent = `Kiste #${openChestId}`

  const closeBtn = document.createElement("button")
  closeBtn.type = "button"
  closeBtn.className = "craft-close-btn"
  closeBtn.textContent = "Schließen"
  closeBtn.addEventListener("click", closeWorkbenchAndInventory)

  head.appendChild(title)
  head.appendChild(closeBtn)
  section.appendChild(head)

  const chestList = document.createElement("div")
  chestList.className = "inventory-grid"

  const grouped = getChestCounts()
  if (grouped.size === 0) {
    const empty = document.createElement("div")
    empty.className = "inventory-empty"
    empty.textContent = "Kiste ist leer"
    chestList.appendChild(empty)
  } else {
    for (const [item, amount] of grouped.entries()) {
      const row = document.createElement("div")
      row.className = "inventory-item"

      const left = document.createElement("div")
      left.className = "inventory-item-left"

      const icon = document.createElement("span")
      icon.className = "inventory-item-icon"
      icon.textContent = itemIcon(item)

      const name = document.createElement("span")
      name.className = "inventory-item-name"
      name.textContent = prettyItemName(item)

      const count = document.createElement("span")
      count.className = "inventory-item-count"
      count.textContent = `x${amount}`

      const actions = document.createElement("div")
      actions.className = "chest-actions"

      const takeOne = document.createElement("button")
      takeOne.type = "button"
      takeOne.className = "inventory-item-menu-btn"
      takeOne.textContent = "Nehmen"
      takeOne.addEventListener("click", () => {
        requestChestAction("take_one", openChestId, item, 1)
      })

      const takeAll = document.createElement("button")
      takeAll.type = "button"
      takeAll.className = "inventory-item-menu-btn"
      takeAll.textContent = "Alles"
      takeAll.addEventListener("click", () => {
        requestChestAction("take_all", openChestId, item, amount)
      })

      actions.appendChild(takeOne)
      actions.appendChild(takeAll)
      left.appendChild(icon)
      left.appendChild(name)
      row.appendChild(left)
      row.appendChild(count)
      row.appendChild(actions)
      chestList.appendChild(row)
    }
  }

  section.appendChild(chestList)
  parent.appendChild(section)
}

function updateInventory() {
  inventoryUI.innerHTML = ""

  const panel = document.createElement("div")
  panel.className = "inventory-panel"

  const title = document.createElement("h3")
  title.className = "inventory-title"
  title.textContent = "Inventar"

  const subtitle = document.createElement("p")
  subtitle.className = "inventory-subtitle"
  subtitle.textContent = `${inventory.length} Item${inventory.length === 1 ? "" : "s"}${equippedItem ? ` | Ausgerüstet: ${prettyItemName(equippedItem)}` : ""}`

  const list = document.createElement("div")
  list.className = "inventory-grid"

  const grouped = getInventoryCounts()

  if (grouped.size === 0) {
    const empty = document.createElement("div")
    empty.className = "inventory-empty"
    empty.textContent = "Keine Items vorhanden"
    list.appendChild(empty)
  } else {
    for (const [item, amount] of grouped.entries()) {
      const card = document.createElement("div")
      card.className = "inventory-item"
      card.draggable = true
      if (item === equippedItem) {
        card.classList.add("equipped")
      }

      const left = document.createElement("div")
      left.className = "inventory-item-left"

      const icon = document.createElement("span")
      icon.className = "inventory-item-icon"
      icon.textContent = itemIcon(item)

      const name = document.createElement("span")
      name.className = "inventory-item-name"
      name.textContent = prettyItemName(item)

      const count = document.createElement("span")
      count.className = "inventory-item-count"
      count.textContent = `x${amount}`

      card.addEventListener("dragstart", (event) => {
        draggedInventoryItem = item
        event.dataTransfer.setData("text/plain", item)
        event.dataTransfer.effectAllowed = "move"
      })

      card.addEventListener("dragend", (event) => {
        const dropTarget = document.elementFromPoint(event.clientX, event.clientY)
        const droppedOutsideInventory = !dropTarget || !inventoryUI.contains(dropTarget)
        if (droppedOutsideInventory && draggedInventoryItem) {
          requestInventoryAction("drop_item", { item: draggedInventoryItem })
        }
        draggedInventoryItem = ""
      })

      card.addEventListener("click", () => {
        openItemMenu(card, item, amount)
      })

      if (openChestId) {
        card.addEventListener("dblclick", () => {
          requestChestAction("store_one", openChestId, item, 1)
        })
      }

      left.appendChild(icon)
      left.appendChild(name)
      card.appendChild(left)
      card.appendChild(count)
      list.appendChild(card)
    }
  }

  panel.appendChild(title)
  panel.appendChild(subtitle)
  panel.appendChild(list)

  if (!workbenchCraftOpen && !openChestId) {
    renderCraft2x2Section(panel)
  }
  renderWorkbenchSection(panel)
  renderChestSection(panel)

  inventoryUI.appendChild(panel)
}

document.addEventListener("click", (event) => {
  if (inventoryUI.style.display !== "block") return

  if (!inventoryUI.contains(event.target)) {
    if (workbenchCraftOpen) {
      if (Date.now() < workbenchIgnoreOutsideClickUntil) {
        return
      }
      closeWorkbenchAndInventory()
      return
    }
    if (openChestId) {
      if (Date.now() < chestIgnoreOutsideClickUntil) {
        return
      }
      closeWorkbenchAndInventory()
      return
    }
    closeWorkbenchAndInventory()
  }
})


