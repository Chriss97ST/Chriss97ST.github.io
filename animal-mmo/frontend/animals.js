const serverAnimals = new Map()
const animalRaycaster = new THREE.Raycaster()
const animalPointer = new THREE.Vector2()
let mountedHareId = 0
let rideTransition = null
let rideControlActive = false
const HARE_RIDE_SEAT_HEIGHT = 1.46
const HARE_JUMP_VELOCITY = 0.34
const HARE_SUPER_JUMP_VELOCITY = 0.74
const HARE_JUMP_GRAVITY = -0.022
const HARE_JUMP_CHARGE_TIME = 1.05
let hareJumpOffset = 0
let hareJumpVelocity = 0
let hareJumpAirborne = false
let hareJumpCharging = false
let hareJumpCharge = 0
let hareJumpHeld = false

function setHareJumpChargeVisible(visible) {
  const bar = document.getElementById("rideJumpCharge")
  if (!bar) return
  bar.classList.toggle("visible", Boolean(visible))
}

function setHareJumpChargeAmount(amount) {
  const fill = document.getElementById("rideJumpChargeFill")
  if (!fill) return
  const ratio = Math.max(0, Math.min(1, Number(amount || 0)))
  fill.style.width = `${Math.round(ratio * 100)}%`
}

function clearHareJumpCharge(cancelOnly = false) {
  hareJumpCharging = false
  hareJumpCharge = 0
  hareJumpHeld = false
  setHareJumpChargeAmount(0)
  setHareJumpChargeVisible(false)

  if (!cancelOnly) {
    hareJumpOffset = 0
    hareJumpVelocity = 0
    hareJumpAirborne = false
  }
}

function setRideHudLabel(text = "", visible = false) {
  const label = document.getElementById("rideHudLabel")
  if (!label) return
  label.textContent = text
  label.classList.toggle("visible", Boolean(visible))
}

function setRideHudControlsVisible(visible) {
  const controls = document.getElementById("rideHudControls")
  if (!controls) return
  controls.classList.toggle("visible", Boolean(visible))
}

function updateRideControlButtonState() {
  const btn = document.getElementById("rideControlBtn")
  if (!btn) return
  btn.classList.toggle("active", Boolean(rideControlActive))
  btn.textContent = rideControlActive ? "Auto" : "Kontrolle"
}

function ensureRideHudControlsHandlers() {
  const controlBtn = document.getElementById("rideControlBtn")
  const dismountBtn = document.getElementById("rideDismountBtn")
  if (!controlBtn || !dismountBtn) return
  if (!controlBtn.dataset.boundRideControl) {
    controlBtn.dataset.boundRideControl = "1"
    controlBtn.addEventListener("click", (event) => {
      if (!mountedHareId || rideTransition) return
      rideControlActive = !rideControlActive
      if (!rideControlActive) {
        // Switching to auto should cancel any held jump charge and block jumping.
        clearHareJumpCharge(true)
        const mount = serverAnimals.get(mountedHareId)
        if (mount) {
          const toTargetX = mount.targetPos.x - mount.mesh.position.x
          const toTargetZ = mount.targetPos.z - mount.mesh.position.z
          const toTargetLen = Math.hypot(toTargetX, toTargetZ)
          if (toTargetLen > 0.02) {
            mount.mesh.rotation.y = Math.atan2(toTargetX, toTargetZ) + Math.PI
          } else if (Math.abs(mount.dirX) + Math.abs(mount.dirZ) > 0.001) {
            mount.mesh.rotation.y = Math.atan2(mount.dirX, mount.dirZ) + Math.PI
          }
        }
      }
      updateRideControlButtonState()
      setRideHudLabel(rideControlActive ? "Reitest: Hase (Kontrolle)" : "Reitest: Hase", true)
      if (event.currentTarget && typeof event.currentTarget.blur === "function") {
        event.currentTarget.blur()
      }
    })
  }
  if (!dismountBtn.dataset.boundRideDismount) {
    dismountBtn.dataset.boundRideDismount = "1"
    dismountBtn.addEventListener("click", (event) => {
      dismountHare()
      if (event.currentTarget && typeof event.currentTarget.blur === "function") {
        event.currentTarget.blur()
      }
    })
  }
}

function easeOutCubic(t) {
  const x = Math.max(0, Math.min(1, t))
  return 1 - Math.pow(1 - x, 3)
}

function lerpAngle(a, b, t) {
  let delta = (b - a) % (Math.PI * 2)
  if (delta > Math.PI) delta -= Math.PI * 2
  if (delta < -Math.PI) delta += Math.PI * 2
  return a + delta * t
}

function beginRideTransition(mode, startPos, endPos, startYaw, endYaw, targetAnimalId = 0, duration = 0.32) {
  rideTransition = {
    mode,
    elapsed: 0,
    duration: Math.max(0.08, duration),
    startPos: startPos.clone(),
    endPos: endPos.clone(),
    startYaw,
    endYaw,
    targetAnimalId: Number(targetAnimalId || 0)
  }
}

function setDismountButtonVisible(visible) {
  const btn = document.getElementById("touchDismount")
  if (!btn) return
  btn.classList.toggle("active", Boolean(visible))
}

function setTouchMountButtonVisible(visible) {
  const btn = document.getElementById("touchMount")
  if (!btn) return
  btn.classList.toggle("active", Boolean(visible))
}

function nearestRideableHare(maxDist2 = 16) {
  if (typeof player === "undefined" || !player) return null
  let best = null
  let bestDist2 = Number(maxDist2)

  for (const entry of serverAnimals.values()) {
    if (entry.type !== "hare") continue
    const dx = entry.mesh.position.x - player.position.x
    const dz = entry.mesh.position.z - player.position.z
    const dist2 = dx * dx + dz * dz
    if (dist2 > bestDist2) continue
    best = entry
    bestDist2 = dist2
  }

  return best
}

function tryMountNearestHare() {
  if (isRidingHare() || isRideTransitioning()) return false
  const nearest = nearestRideableHare(16)
  if (!nearest) return false
  return mountHare(nearest.id)
}

function isRidingHare() {
  return Boolean(mountedHareId)
}

function isRideTransitioning() {
  return Boolean(rideTransition)
}

function queueHareJump() {
  if (!mountedHareId || isRideTransitioning() || !rideControlActive || hareJumpAirborne) return false
  hareJumpVelocity = HARE_JUMP_VELOCITY
  hareJumpAirborne = true
  hareJumpCharging = false
  hareJumpCharge = 0
  setHareJumpChargeAmount(0)
  setHareJumpChargeVisible(false)
  return true
}

function setHareJumpHold(active) {
  if (!mountedHareId || isRideTransitioning() || !rideControlActive) {
    if (!active) clearHareJumpCharge(true)
    return false
  }

  if (active) {
    if (hareJumpAirborne || hareJumpCharging) return false
    hareJumpCharging = true
    hareJumpCharge = 0
    setHareJumpChargeAmount(0)
    setHareJumpChargeVisible(true)
    return true
  }

  if (!hareJumpCharging || hareJumpAirborne) {
    setHareJumpChargeAmount(0)
    setHareJumpChargeVisible(false)
    return false
  }

  const ratio = Math.max(0, Math.min(1, hareJumpCharge))
  hareJumpVelocity = HARE_JUMP_VELOCITY + (HARE_SUPER_JUMP_VELOCITY - HARE_JUMP_VELOCITY) * ratio
  hareJumpAirborne = true
  hareJumpCharging = false
  hareJumpCharge = 0
  setHareJumpChargeAmount(0)
  setHareJumpChargeVisible(false)
  return true
}

function dismountHare() {
  if (!mountedHareId || isRideTransitioning()) return
  const mount = serverAnimals.get(mountedHareId)
  if (!mount || typeof player === "undefined" || !player) {
    mountedHareId = 0
    setDismountButtonVisible(false)
    setRideHudLabel("", false)
    setRideHudControlsVisible(false)
    setTouchMountButtonVisible(false)
    rideControlActive = false
    updateRideControlButtonState()
    return
  }

  const startPos = player.position.clone()
  const endPos = new THREE.Vector3(
    mount.mesh.position.x + Math.cos(mount.mesh.rotation.y) * 1.25,
    1,
    mount.mesh.position.z - Math.sin(mount.mesh.rotation.y) * 1.25
  )
  const startYaw = player.rotation.y
  const endYaw = mount.mesh.rotation.y + Math.PI

  mountedHareId = 0
  rideControlActive = false
  clearHareJumpCharge(false)
  updateRideControlButtonState()
  setDismountButtonVisible(false)
  setTouchMountButtonVisible(false)
  setRideHudLabel("Steige ab...", true)
  setRideHudControlsVisible(true)
  beginRideTransition("dismount", startPos, endPos, startYaw, endYaw, 0, 0.28)
}

function mountHare(animalId) {
  if (isRideTransitioning() || mountedHareId) return false
  const entry = serverAnimals.get(Number(animalId))
  if (!entry || entry.type !== "hare" || typeof player === "undefined" || !player) return false

  const startPos = player.position.clone()
  const endPos = new THREE.Vector3(
    entry.mesh.position.x,
    entry.mesh.position.y + HARE_RIDE_SEAT_HEIGHT,
    entry.mesh.position.z
  )
  const startYaw = player.rotation.y
  const endYaw = entry.mesh.rotation.y + Math.PI

  setDismountButtonVisible(false)
  rideControlActive = false
  clearHareJumpCharge(false)
  updateRideControlButtonState()
  setRideHudLabel("Steige auf...", true)
  setRideHudControlsVisible(true)
  beginRideTransition("mount", startPos, endPos, startYaw, endYaw, Number(animalId), 0.34)
  return true
}

function tryRideHareByClick(event, camera, canvas) {
  if (!camera || !canvas || !serverAnimals.size || isRidingHare() || isRideTransitioning()) return false
  if (typeof player === "undefined" || !player) return false

  const rect = canvas.getBoundingClientRect()
  animalPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  animalPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  animalRaycaster.setFromCamera(animalPointer, camera)
  const hareMeshes = []
  for (const entry of serverAnimals.values()) {
    if (entry.type === "hare") hareMeshes.push(entry.mesh)
  }
  if (!hareMeshes.length) return false

  const hits = animalRaycaster.intersectObjects(hareMeshes, true)
  if (!hits.length) return false

  const hit = hits[0]
  let node = hit.object
  while (node && !node.userData?.animalId) {
    node = node.parent
  }
  const animalId = Number(node?.userData?.animalId || 0)
  if (!animalId) return false

  const entry = serverAnimals.get(animalId)
  if (!entry || entry.type !== "hare") return false
  const dx = entry.mesh.position.x - player.position.x
  const dz = entry.mesh.position.z - player.position.z
  if ((dx * dx + dz * dz) > 16) return false

  return mountHare(animalId)
}

function resolveMountedHareCollision(hareMesh, nextX, nextZ) {
  const riderRadius = Number(hareMesh?.userData?.collisionRadius || 0.52)
  let x = nextX
  let z = nextZ

  const blockers = (typeof objects !== "undefined" && Array.isArray(objects)) ? objects : []
  const currentPlayer = (typeof player !== "undefined") ? player : null
  for (let pass = 0; pass < 2; pass++) {
    for (const blocker of blockers) {
      if (!blocker || blocker === hareMesh || blocker === currentPlayer) continue
      const kind = blocker.userData?.worldObjectKind
      if (kind !== "tree" && kind !== "rock") continue

      const obstacleRadius = Number(blocker.userData?.collisionRadius || 0)
      if (obstacleRadius <= 0) continue

      const dx = x - blocker.position.x
      const dz = z - blocker.position.z
      const dist = Math.hypot(dx, dz) || 0.0001
      const minDist = riderRadius + obstacleRadius

      if (dist < minDist) {
        const push = minDist - dist
        x += (dx / dist) * push
        z += (dz / dist) * push
      }
    }
  }

  return {
    x: Math.max(-95, Math.min(95, x)),
    z: Math.max(-95, Math.min(95, z))
  }
}

function animalColors(type) {
  if (type === "wolf") return { body: 0x8f9aa6, detail: 0xdfe6ed, tail: 0x6d7782 }
  if (type === "hare") return { body: 0xc9a075, detail: 0xf3dfca, tail: 0x9f805f }
  return { body: 0xd96c35, detail: 0xf8e6d8, tail: 0x78422f }
}

function createAnimalModel(type) {
  const colors = animalColors(type)
  const bodyMat = new THREE.MeshStandardMaterial({ color: colors.body, flatShading: true })
  const detailMat = new THREE.MeshStandardMaterial({ color: colors.detail, flatShading: true })
  const tailMat = new THREE.MeshStandardMaterial({ color: colors.tail, flatShading: true })

  const group = new THREE.Group()

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.65, 0.5), bodyMat)
  body.position.y = 0.66
  group.add(body)

  const head = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.42, 0.42), detailMat)
  head.position.set(0, 0.92, -0.56)
  group.add(head)

  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.2, 0.55), tailMat)
  tail.position.set(0, 0.82, 0.72)
  tail.rotation.x = 0.7
  group.add(tail)

  const legFL = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.52, 0.18), detailMat)
  const legFR = legFL.clone()
  const legBL = legFL.clone()
  const legBR = legFL.clone()

  legFL.position.set(-0.35, 0.28, -0.2)
  legFR.position.set(0.35, 0.28, -0.2)
  legBL.position.set(-0.35, 0.28, 0.24)
  legBR.position.set(0.35, 0.28, 0.24)

  group.add(legFL)
  group.add(legFR)
  group.add(legBL)
  group.add(legBR)

  if (type === "hare") {
    const earL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.45, 0.1), detailMat)
    const earR = earL.clone()
    earL.position.set(-0.12, 1.3, -0.58)
    earR.position.set(0.12, 1.3, -0.58)
    group.add(earL)
    group.add(earR)
  }

  group.userData.anim = {
    cycle: 0,
    legs: [legFL, legFR, legBL, legBR],
    body,
    head,
    tail
  }

  group.userData.collisionRadius = type === "hare" ? 0.52 : 0.65
  return group
}

function spawnAnimals() {
  // Tiere kommen serverseitig über WebSocket-Snapshots.
}

function clearServerAnimals() {
  for (const entry of serverAnimals.values()) {
    scene.remove(entry.mesh)
  }
  serverAnimals.clear()
}

function applyAnimalsSnapshot(animalStates) {
  const seen = new Set()

  for (const state of animalStates || []) {
    const id = Number(state.id)
    seen.add(id)

    let entry = serverAnimals.get(id)

    if (!entry) {
      const mesh = createAnimalModel(state.type)
      mesh.position.set(state.x, state.y || 0, state.z)
      mesh.userData.animalId = id
      mesh.userData.animalType = state.type
      scene.add(mesh)

      entry = {
        id,
        type: state.type,
        mesh,
        targetPos: new THREE.Vector3(state.x, state.y || 0, state.z),
        dirX: state.dir_x || 0,
        dirZ: state.dir_z || 1,
        speed: state.speed || 0
      }

      serverAnimals.set(id, entry)
    }

    entry.targetPos.set(state.x, state.y || 0, state.z)
    entry.dirX = state.dir_x || 0
    entry.dirZ = state.dir_z || 1
    entry.speed = state.speed || 0
  }

  for (const [id, entry] of serverAnimals.entries()) {
    if (seen.has(id)) continue
    if (mountedHareId === id) {
      mountedHareId = 0
      setDismountButtonVisible(false)
      setRideHudLabel("", false)
    }
    if (rideTransition && rideTransition.targetAnimalId === id) {
      rideTransition = null
      setDismountButtonVisible(false)
      setRideHudLabel("", false)
    }
    scene.remove(entry.mesh)
    serverAnimals.delete(id)
  }
}

function updateAnimals(delta) {
  ensureRideHudControlsHandlers()

  const sharedInput = typeof window.getMovementInputState === "function"
    ? window.getMovementInputState()
    : { x: 0, z: 0, sprint: false }
  const sharedMoveLen = Math.hypot(sharedInput.x || 0, sharedInput.z || 0)
  const jumpPressed = typeof keys !== "undefined" && Boolean(keys[" "] || keys["Space"] || keys["Spacebar"])

  for (const entry of serverAnimals.values()) {
    const controlledMountedHare = Boolean(rideControlActive && mountedHareId === entry.id)
    const mesh = entry.mesh
    let faceX = entry.dirX || 0
    let faceZ = entry.dirZ || 0
    if (!controlledMountedHare) {
      const toTarget = new THREE.Vector3(
        entry.targetPos.x - mesh.position.x,
        entry.targetPos.y - mesh.position.y,
        entry.targetPos.z - mesh.position.z
      )
      const distance = toTarget.length()
      if (distance > 0.02) {
        faceX = toTarget.x
        faceZ = toTarget.z
      }
      if (distance > 0.0001) {
        const maxStep = delta * 5.2
        if (distance <= maxStep) {
          mesh.position.copy(entry.targetPos)
        } else {
          toTarget.multiplyScalar(1 / distance)
          mesh.position.addScaledVector(toTarget, maxStep)
        }
      }
    }

    if (!controlledMountedHare && Math.abs(faceX) + Math.abs(faceZ) > 0.001) {
      mesh.rotation.y = Math.atan2(faceX, faceZ) + Math.PI
    }

    const anim = mesh.userData.anim
    if (!anim) continue

    const moving = controlledMountedHare ? (sharedMoveLen > 0.02) : (entry.speed > 0.005)
    const animSpeed = moving ? Math.max(8, entry.speed * 150) : 4
    anim.cycle += delta * animSpeed

    const swing = Math.sin(anim.cycle) * (moving ? 0.9 : 0.12)

    anim.legs[0].rotation.x = swing
    anim.legs[1].rotation.x = -swing
    anim.legs[2].rotation.x = -swing
    anim.legs[3].rotation.x = swing

    anim.body.position.y = 0.66 + Math.abs(Math.sin(anim.cycle * 1.8)) * (moving ? 0.04 : 0.01)
    anim.head.rotation.x = Math.sin(anim.cycle * 0.6) * 0.07
    anim.tail.rotation.y = Math.sin(anim.cycle * 2.2) * 0.35
  }

  if (mountedHareId) {
    const mount = serverAnimals.get(mountedHareId)
    if (!mount || typeof player === "undefined" || !player) {
      mountedHareId = 0
      rideControlActive = false
      clearHareJumpCharge(false)
      updateRideControlButtonState()
      setDismountButtonVisible(false)
      setRideHudLabel("", false)
      setRideHudControlsVisible(false)
      setTouchMountButtonVisible(false)
      return
    }

    if (rideControlActive && jumpPressed && !hareJumpHeld) {
      setHareJumpHold(true)
    }
    if (rideControlActive && !jumpPressed && hareJumpHeld) {
      setHareJumpHold(false)
    }
    if (!rideControlActive && hareJumpCharging) {
      clearHareJumpCharge(true)
    }

    hareJumpHeld = jumpPressed

    if (hareJumpCharging && !hareJumpAirborne) {
      hareJumpCharge = Math.min(1, hareJumpCharge + (delta / HARE_JUMP_CHARGE_TIME))
      setHareJumpChargeAmount(hareJumpCharge)
    }

    if (hareJumpAirborne || hareJumpOffset > 0) {
      hareJumpVelocity += HARE_JUMP_GRAVITY
      hareJumpOffset += hareJumpVelocity
      if (hareJumpOffset <= 0) {
        hareJumpOffset = 0
        hareJumpVelocity = 0
        hareJumpAirborne = false
      }
    }

    if (rideControlActive && !rideTransition) {
      const input = sharedInput

      const len = Math.hypot(input.x, input.z)
      if (len > 0.001) {
        const speed = (input.sprint ? 0.21 : 0.145) * delta * 60
        const intendedX = mount.mesh.position.x + input.x * speed
        const intendedZ = mount.mesh.position.z + input.z * speed
        const resolved = resolveMountedHareCollision(mount.mesh, intendedX, intendedZ)
        mount.mesh.position.x = resolved.x
        mount.mesh.position.z = resolved.z
        mount.targetPos.set(mount.mesh.position.x, mount.targetPos.y, mount.mesh.position.z)
        mount.mesh.rotation.y = Math.atan2(input.x, input.z) + Math.PI
      }
    }

    const mountedResolved = resolveMountedHareCollision(mount.mesh, mount.mesh.position.x, mount.mesh.position.z)
    if (Math.abs(mountedResolved.x - mount.mesh.position.x) > 0.0001 || Math.abs(mountedResolved.z - mount.mesh.position.z) > 0.0001) {
      mount.mesh.position.x = mountedResolved.x
      mount.mesh.position.z = mountedResolved.z
      mount.targetPos.set(mount.mesh.position.x, mount.targetPos.y, mount.mesh.position.z)
    }

    mount.mesh.position.y = mount.targetPos.y + hareJumpOffset

    player.position.x = mount.mesh.position.x
    player.position.z = mount.mesh.position.z
    player.position.y = mount.mesh.position.y + HARE_RIDE_SEAT_HEIGHT
    player.rotation.y = mount.mesh.rotation.y + Math.PI
    setDismountButtonVisible(true)
    setRideHudLabel(rideControlActive ? "Reitest: Hase (Kontrolle)" : "Reitest: Hase", true)
    setRideHudControlsVisible(true)
    setTouchMountButtonVisible(false)
  } else if (!rideTransition) {
    rideControlActive = false
    clearHareJumpCharge(false)
    updateRideControlButtonState()
    setRideHudLabel("", false)
    setRideHudControlsVisible(false)
    const mobileClient = window.matchMedia("(hover: none) and (pointer: coarse)").matches
    setTouchMountButtonVisible(mobileClient && Boolean(nearestRideableHare(16)))
  }

  if (rideTransition && typeof player !== "undefined" && player) {
    rideTransition.elapsed += delta
    const t = Math.min(1, rideTransition.elapsed / rideTransition.duration)
    const eased = easeOutCubic(t)

    if (rideTransition.mode === "mount" && rideTransition.targetAnimalId) {
      const target = serverAnimals.get(rideTransition.targetAnimalId)
      if (!target) {
        rideTransition = null
        setRideHudLabel("", false)
        return
      }
      rideTransition.endPos.set(target.mesh.position.x, target.mesh.position.y + HARE_RIDE_SEAT_HEIGHT, target.mesh.position.z)
      rideTransition.endYaw = target.mesh.rotation.y + Math.PI
    }

    player.position.lerpVectors(rideTransition.startPos, rideTransition.endPos, eased)
    player.rotation.y = lerpAngle(rideTransition.startYaw, rideTransition.endYaw, eased)

    if (t >= 1) {
      if (rideTransition.mode === "mount") {
        mountedHareId = rideTransition.targetAnimalId
        setDismountButtonVisible(true)
        setRideHudControlsVisible(true)
        setRideHudLabel("Reitest: Hase", true)
        setTouchMountButtonVisible(false)
      } else {
        setDismountButtonVisible(false)
        setRideHudControlsVisible(false)
        setRideHudLabel("", false)
        const mobileClient = window.matchMedia("(hover: none) and (pointer: coarse)").matches
        setTouchMountButtonVisible(mobileClient && Boolean(nearestRideableHare(16)))
      }
      rideTransition = null
    }
  }
}

window.queueHareJump = queueHareJump
window.setHareJumpHold = setHareJumpHold
