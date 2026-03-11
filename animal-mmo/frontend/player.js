let player
let velocityY = 0
let gravity = -0.02
let onGround = false
let playerIsMoving = false
let jumpQueued = false

let inventory = []
let workbenchCraftOpen = false
const craftSlots2x2 = Array(4).fill(null)
const craftSlots3x3 = Array(9).fill(null)

const itemMeta = {
  fruit: { key: "frucht", label: "Frucht", icon: "🍎" },
  frucht: { key: "frucht", label: "Frucht", icon: "🍎" },
  log: { key: "holzstamm", label: "Holzstamm", icon: "🪵" },
  holzstamm: { key: "holzstamm", label: "Holzstamm", icon: "🪵" },
  holzlatte: { key: "holzlatte", label: "Holzlatte", icon: "🪚" },
  stein: { key: "stein", label: "Stein", icon: "🪨" },
  werkbank: { key: "werkbank", label: "Werkbank", icon: "🧰" },
  stock: { key: "stock", label: "Stock", icon: "🪵" },
  holzspitzhacke: { key: "holzspitzhacke", label: "Holzspitzhacke", icon: "⛏️" },
  steinspitzhacke: { key: "steinspitzhacke", label: "Steinspitzhacke", icon: "⛏️" }
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
  }
]

function normalizeItem(item) {
  return itemMeta[item]?.key || item
}

function prettyItemName(item) {
  return itemMeta[item]?.label || item
}

function itemIcon(item) {
  return itemMeta[item]?.icon || "📦"
}

function createSmileyTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext("2d")

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  ctx.fillStyle = "#ffd54f"
  ctx.beginPath()
  ctx.arc(64, 64, 58, 0, Math.PI * 2)
  ctx.fill()

  ctx.fillStyle = "#2b2b2b"
  ctx.beginPath()
  ctx.arc(46, 52, 9, 0, Math.PI * 2)
  ctx.arc(82, 52, 9, 0, Math.PI * 2)
  ctx.fill()

  ctx.strokeStyle = "#2b2b2b"
  ctx.lineWidth = 10
  ctx.lineCap = "round"
  ctx.beginPath()
  ctx.arc(64, 70, 26, 0.15 * Math.PI, 0.85 * Math.PI)
  ctx.stroke()

  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

function createPlayerModel() {
  const group = new THREE.Group()
  const skin = new THREE.MeshStandardMaterial({ color: 0xffcc99, flatShading: true })
  const cloth = new THREE.MeshStandardMaterial({ color: 0x3a86ff, flatShading: true })
  const dark = new THREE.MeshStandardMaterial({ color: 0x23324a, flatShading: true })

  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1.25, 0.55), cloth)
  torso.position.y = 0.2

  const head = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.62, 0.62), skin)
  head.position.y = 1.18

  const faceTexture = createSmileyTexture()
  const faceMaterial = new THREE.MeshBasicMaterial({
    map: faceTexture,
    transparent: true
  })
  const face = new THREE.Mesh(new THREE.PlaneGeometry(0.58, 0.58), faceMaterial)
  face.position.z = 0.33
  head.add(face)

  const armL = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.9, 0.24), skin)
  armL.position.set(-0.66, 0.14, 0)

  const armR = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.9, 0.24), skin)
  armR.position.set(0.66, 0.14, 0)

  const legL = new THREE.Mesh(new THREE.BoxGeometry(0.28, 1.0, 0.3), dark)
  legL.position.set(-0.23, -0.86, 0)

  const legR = new THREE.Mesh(new THREE.BoxGeometry(0.28, 1.0, 0.3), dark)
  legR.position.set(0.23, -0.86, 0)

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

function openWorkbenchCrafting() {
  workbenchCraftOpen = true
  inventoryUI.style.display = "block"
  updateInventory()
}

function buildItemOptions(item) {
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
  }

  if (item === "werkbank") {
    options.push({
      label: "Aufstellen",
      action: () => requestPlaceWorkbench()
    })
  }

  return options
}

function openItemMenu(anchor, item) {
  closeItemMenu()

  const options = buildItemOptions(item)
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
  closeBtn.addEventListener("click", closeWorkbenchCrafting)

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

function updateInventory() {
  inventoryUI.innerHTML = ""

  const panel = document.createElement("div")
  panel.className = "inventory-panel"

  const title = document.createElement("h3")
  title.className = "inventory-title"
  title.textContent = "Inventar"

  const subtitle = document.createElement("p")
  subtitle.className = "inventory-subtitle"
  subtitle.textContent = `${inventory.length} Item${inventory.length === 1 ? "" : "s"}`

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
        event.dataTransfer.setData("text/plain", item)
      })

      card.addEventListener("click", () => {
        openItemMenu(card, item)
      })

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

  if (!workbenchCraftOpen) {
    renderCraft2x2Section(panel)
  }
  renderWorkbenchSection(panel)

  inventoryUI.appendChild(panel)
}

document.addEventListener("click", (event) => {
  if (!inventoryUI.contains(event.target)) {
    closeItemMenu()
  }
})


