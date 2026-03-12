let scene
let objects = []

const worldObjectsById = new Map()
const pickupMeshesById = new Map()
const worldRaycaster = new THREE.Raycaster()
const worldPointer = new THREE.Vector2()
const skyClouds = []

function createCloudTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 256
  canvas.height = 128
  const ctx = canvas.getContext("2d")

  ctx.clearRect(0, 0, 256, 128)
  ctx.fillStyle = "rgba(255,255,255,0.92)"
  ctx.beginPath()
  ctx.ellipse(80, 70, 54, 28, 0, 0, Math.PI * 2)
  ctx.ellipse(124, 62, 52, 30, 0, 0, Math.PI * 2)
  ctx.ellipse(162, 74, 40, 24, 0, 0, Math.PI * 2)
  ctx.fill()

  const tex = new THREE.CanvasTexture(canvas)
  tex.needsUpdate = true
  return tex
}

function createSkyEnvironment() {
  const sky = new THREE.Mesh(
    new THREE.BoxGeometry(900, 900, 900),
    new THREE.MeshBasicMaterial({ color: 0x9dd8ff, side: THREE.BackSide })
  )
  scene.add(sky)

  const cloudTexture = createCloudTexture()
  for (let i = 0; i < 20; i++) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(18 + Math.random() * 22, 8 + Math.random() * 12),
      new THREE.MeshBasicMaterial({
        map: cloudTexture,
        transparent: true,
        opacity: 0.58,
        depthWrite: false
      })
    )

    mesh.position.set(
      -180 + Math.random() * 360,
      42 + Math.random() * 24,
      -180 + Math.random() * 360
    )
    mesh.rotation.x = -Math.PI * 0.5
    mesh.rotation.z = Math.random() * Math.PI * 2

    const speed = 1.4 + Math.random() * 2.6
    skyClouds.push({ mesh, speed })
    scene.add(mesh)
  }
}

function updateSkyEnvironment(delta) {
  if (!skyClouds.length) return
  for (const entry of skyClouds) {
    entry.mesh.position.x += entry.speed * delta
    if (entry.mesh.position.x > 220) {
      entry.mesh.position.x = -220
      entry.mesh.position.z = -180 + Math.random() * 360
    }
  }
}

function createGround() {
  const ground = new THREE.Mesh(
    new THREE.BoxGeometry(200, 1, 200),
    new THREE.MeshStandardMaterial({ color: 0x2e7d32, flatShading: true })
  )

  ground.position.y = -0.5
  scene.add(ground)
}

function clearWorldVisuals() {
  for (const entry of worldObjectsById.values()) {
    scene.remove(entry.mesh)
  }

  for (const mesh of pickupMeshesById.values()) {
    scene.remove(mesh)
  }

  worldObjectsById.clear()
  pickupMeshesById.clear()
  objects = []
}

function createTreeMesh(obj) {
  const group = new THREE.Group()

  const species = obj.tree_species || "oak"
  const variant = Number(obj.tree_variant || 1)
  const growthStage = Number(obj.growth_stage ?? 2)
  const isFresh = Number(obj.planted_fresh || 0) === 1

  const stageScale = growthStage <= 0 ? 0.46 : growthStage === 1 ? 0.78 : 1
  const trunkScale = Number(obj.trunk_scale || 1) * stageScale
  const heightScale = Number(obj.height_scale || 1) * stageScale

  const trunkRadiusTop = 0.16 * obj.scale * trunkScale
  const trunkRadiusBottom = 0.24 * obj.scale * trunkScale
  const trunkHeight = 2.1 * obj.scale * heightScale

  let trunkColor = 0x8b5a2b
  let leavesColor = 0x2fb463
  let crownKind = "cone"

  if (species === "pine") {
    trunkColor = 0x6d4424
    leavesColor = variant === 1 ? 0x2f8f43 : variant === 2 ? 0x3aa655 : 0x267f3a
    crownKind = "tiered"
  } else if (species === "birch") {
    trunkColor = 0xdbd3c3
    leavesColor = variant === 1 ? 0xa8d85f : variant === 2 ? 0x9fcd57 : 0xb4de6b
    crownKind = "round"
  } else {
    leavesColor = variant === 1 ? 0x2fb463 : variant === 2 ? 0x37c26a : 0x2ea65e
    crownKind = variant === 3 ? "round" : "cone"
  }

  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(trunkRadiusTop, trunkRadiusBottom, trunkHeight, 7),
    new THREE.MeshStandardMaterial({ color: trunkColor, flatShading: true })
  )

  trunk.position.y = trunkHeight * 0.5

  const leafMat = new THREE.MeshStandardMaterial({ color: leavesColor, flatShading: true })

  if (growthStage <= 0) {
    const sproutStem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03 * obj.scale, 0.035 * obj.scale, 0.4 * obj.scale, 6),
      new THREE.MeshStandardMaterial({ color: 0x4a7a2f, flatShading: true })
    )
    sproutStem.position.y = 0.25 * obj.scale

    const leaf1 = new THREE.Mesh(new THREE.SphereGeometry(0.16 * obj.scale, 8, 8), leafMat)
    const leaf2 = new THREE.Mesh(new THREE.SphereGeometry(0.13 * obj.scale, 8, 8), leafMat)
    leaf1.position.set(-0.12 * obj.scale, 0.48 * obj.scale, 0)
    leaf2.position.set(0.14 * obj.scale, 0.55 * obj.scale, 0.04 * obj.scale)

    group.add(sproutStem)
    group.add(leaf1)
    group.add(leaf2)
  } else {
    group.add(trunk)

    if (crownKind === "round") {
      const crown = new THREE.Mesh(new THREE.SphereGeometry(1.1 * obj.scale * heightScale, 8, 7), leafMat)
      crown.position.y = trunkHeight + 0.95 * obj.scale * heightScale
      group.add(crown)

      if (variant >= 2) {
        const crown2 = new THREE.Mesh(new THREE.SphereGeometry(0.85 * obj.scale * heightScale, 8, 7), leafMat)
        crown2.position.set(0.45 * obj.scale, trunkHeight + 0.7 * obj.scale * heightScale, -0.2 * obj.scale)
        group.add(crown2)
      }
    } else if (crownKind === "tiered") {
      const cone1 = new THREE.Mesh(new THREE.ConeGeometry(1.05 * obj.scale, 1.65 * obj.scale * heightScale, 7), leafMat)
      const cone2 = new THREE.Mesh(new THREE.ConeGeometry(0.8 * obj.scale, 1.35 * obj.scale * heightScale, 7), leafMat)
      const cone3 = new THREE.Mesh(new THREE.ConeGeometry(0.55 * obj.scale, 1.0 * obj.scale * heightScale, 7), leafMat)
      cone1.position.y = trunkHeight + 0.55 * obj.scale * heightScale
      cone2.position.y = trunkHeight + 1.2 * obj.scale * heightScale
      cone3.position.y = trunkHeight + 1.78 * obj.scale * heightScale
      group.add(cone1)
      group.add(cone2)
      group.add(cone3)
    } else {
      const leaves = new THREE.Mesh(
        new THREE.ConeGeometry(1.45 * obj.scale, 2.4 * obj.scale * heightScale, 7),
        leafMat
      )
      leaves.position.y = trunkHeight + 0.7 * obj.scale * heightScale
      group.add(leaves)
    }
  }

  if (isFresh) {
    const freshMarker = new THREE.Mesh(
      new THREE.TorusGeometry(0.45 * obj.scale, 0.05 * obj.scale, 8, 16),
      new THREE.MeshStandardMaterial({ color: 0xefff54, emissive: 0x3d3f08, flatShading: true })
    )
    freshMarker.rotation.x = Math.PI * 0.5
    freshMarker.position.y = 0.08
    group.add(freshMarker)
  }

  group.position.set(obj.x, obj.y - 1, obj.z)
  group.rotation.y = obj.rotation
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "tree"
  group.userData.collisionRadius = growthStage <= 0 ? 0.45 * obj.scale : growthStage === 1 ? 1.0 * obj.scale : 1.45 * obj.scale

  const fruitNodes = []
  const fruitCount = growthStage <= 0 ? 0 : Number(obj.fruit_count || 0)
  for (let i = 0; i < fruitCount; i++) {
    const fruit = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 8, 8),
      new THREE.MeshStandardMaterial({ color: 0xff6644, flatShading: true })
    )

    const ring = (i % 3) + 1
    const angle = (i * 1.8) + obj.id * 0.37
    const radius = 0.45 + ring * 0.22
    fruit.position.set(
      Math.cos(angle) * radius,
      (trunkHeight * 0.72) + (ring * 0.22 * obj.scale),
      Math.sin(angle) * radius
    )

    group.add(fruit)
    fruitNodes.push(fruit)
  }

  scene.add(group)
  objects.push(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes,
    scale: obj.scale,
    trunkHeight,
    growthStage,
    treeSpecies: species,
    treeVariant: variant,
    trunkScale: Number(obj.trunk_scale || 1),
    heightScale: Number(obj.height_scale || 1),
    plantedFresh: isFresh
  })
}

function createRockMesh(obj) {
  const rock = new THREE.Mesh(
    new THREE.DodecahedronGeometry(0.7 * obj.scale, 0),
    new THREE.MeshStandardMaterial({ color: 0x727a84, flatShading: true })
  )

  rock.position.set(obj.x, obj.y, obj.z)
  rock.rotation.y = obj.rotation
  rock.userData.collisionRadius = 0.85 * obj.scale
  rock.userData.worldObjectId = obj.id
  rock.userData.worldObjectKind = "rock"

  scene.add(rock)
  objects.push(rock)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: rock,
    fruitNodes: []
  })
}

function createWorkbenchMesh(obj) {
  const group = new THREE.Group()

  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 0.22, 1.2),
    new THREE.MeshStandardMaterial({ color: 0xa26a3c, flatShading: true })
  )

  const legGeo = new THREE.BoxGeometry(0.14, 0.8, 0.14)
  const legMat = new THREE.MeshStandardMaterial({ color: 0x6c4626, flatShading: true })
  const leg1 = new THREE.Mesh(legGeo, legMat)
  const leg2 = leg1.clone()
  const leg3 = leg1.clone()
  const leg4 = leg1.clone()

  leg1.position.set(-0.45, -0.4, -0.45)
  leg2.position.set(0.45, -0.4, -0.45)
  leg3.position.set(-0.45, -0.4, 0.45)
  leg4.position.set(0.45, -0.4, 0.45)

  group.add(top)
  group.add(leg1)
  group.add(leg2)
  group.add(leg3)
  group.add(leg4)

  group.position.set(obj.x, obj.y + 0.5, obj.z)
  group.rotation.y = obj.rotation
  group.userData.collisionRadius = 0.85
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "workbench"

  scene.add(group)
  objects.push(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes: []
  })
}

function createChestMesh(obj) {
  const group = new THREE.Group()

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.62, 0.72),
    new THREE.MeshStandardMaterial({ color: 0x8c5a31, flatShading: true })
  )
  body.position.y = 0.26

  const lid = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.34, 0.72),
    new THREE.MeshStandardMaterial({ color: 0xa16a3c, flatShading: true })
  )
  lid.position.y = 0.74

  const latch = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.16, 0.08),
    new THREE.MeshStandardMaterial({ color: 0xcda24e, flatShading: true })
  )
  latch.position.set(0, 0.52, 0.4)

  group.add(body)
  group.add(lid)
  group.add(latch)

  group.position.set(obj.x, obj.y, obj.z)
  group.rotation.y = obj.rotation
  group.userData.collisionRadius = 0.8
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "chest"

  scene.add(group)
  objects.push(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes: []
  })
}

function createMeadowMesh(obj) {
  const group = new THREE.Group()

  const patch1 = new THREE.Mesh(
    new THREE.CircleGeometry(2.2 * obj.scale, 18),
    new THREE.MeshStandardMaterial({ color: 0x5fbf59, flatShading: true })
  )
  patch1.rotation.x = -Math.PI * 0.5
  patch1.position.y = 0.04

  const patch2 = new THREE.Mesh(
    new THREE.CircleGeometry(1.55 * obj.scale, 14),
    new THREE.MeshStandardMaterial({ color: 0x72d067, flatShading: true })
  )
  patch2.rotation.x = -Math.PI * 0.5
  patch2.position.set(0.95 * obj.scale, 0.041, -0.45 * obj.scale)

  group.add(patch1)
  group.add(patch2)

  group.position.set(obj.x, obj.y, obj.z)
  group.rotation.y = obj.rotation
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "meadow"
  group.userData.collisionRadius = 0

  scene.add(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes: []
  })
}

function createPondMesh(obj) {
  const group = new THREE.Group()

  const shore = new THREE.Mesh(
    new THREE.CylinderGeometry(2.05 * obj.scale, 2.2 * obj.scale, 0.08, 20),
    new THREE.MeshStandardMaterial({ color: 0x7d6247, flatShading: true })
  )
  shore.position.y = 0.03

  const water = new THREE.Mesh(
    new THREE.CylinderGeometry(1.85 * obj.scale, 1.95 * obj.scale, 0.045, 20),
    new THREE.MeshStandardMaterial({ color: 0x3d96d8, transparent: true, opacity: 0.83, flatShading: true })
  )
  water.position.y = 0.062

  group.add(shore)
  group.add(water)

  group.position.set(obj.x, obj.y, obj.z)
  group.rotation.y = obj.rotation
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "pond"
  group.userData.collisionRadius = 0

  scene.add(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes: []
  })
}

function createRiverMesh(obj) {
  const group = new THREE.Group()

  const lengthScale = Number(obj.height_scale || 3.5)
  const width = 1.55 * obj.scale
  const length = 6.4 * lengthScale

  const bed = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.55, 0.08, length + 0.45),
    new THREE.MeshStandardMaterial({ color: 0x6f5b45, flatShading: true })
  )
  bed.position.y = 0.03

  const water = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.05, length),
    new THREE.MeshStandardMaterial({ color: 0x2f84c7, transparent: true, opacity: 0.78, flatShading: true })
  )
  water.position.y = 0.07

  group.add(bed)
  group.add(water)

  group.position.set(obj.x, obj.y, obj.z)
  group.rotation.y = obj.rotation
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "river"
  group.userData.collisionRadius = 0

  scene.add(group)

  worldObjectsById.set(obj.id, {
    id: obj.id,
    kind: obj.kind,
    mesh: group,
    fruitNodes: []
  })
}

function addWorldObject(obj) {
  if (obj.kind === "tree") {
    createTreeMesh(obj)
  } else if (obj.kind === "rock") {
    createRockMesh(obj)
  } else if (obj.kind === "workbench") {
    createWorkbenchMesh(obj)
  } else if (obj.kind === "chest") {
    createChestMesh(obj)
  } else if (obj.kind === "meadow") {
    createMeadowMesh(obj)
  } else if (obj.kind === "pond") {
    createPondMesh(obj)
  } else if (obj.kind === "river") {
    createRiverMesh(obj)
  }
}

function setTreeFruitCount(treeId, fruitCount) {
  const entry = worldObjectsById.get(treeId)
  if (!entry || entry.kind !== "tree") return
  if (entry.growthStage <= 0) return

  for (const node of entry.fruitNodes) {
    entry.mesh.remove(node)
  }

  entry.fruitNodes = []

  for (let i = 0; i < fruitCount; i++) {
    const fruit = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 8, 8),
      new THREE.MeshStandardMaterial({ color: 0xff6644, flatShading: true })
    )

    const ring = (i % 3) + 1
    const angle = (i * 1.8) + treeId * 0.37
    const radius = 0.45 + ring * 0.22

    fruit.position.set(
      Math.cos(angle) * radius,
      (entry.trunkHeight || (2.0 * (entry.scale || 1))) * 0.72 + ring * 0.22,
      Math.sin(angle) * radius
    )

    entry.mesh.add(fruit)
    entry.fruitNodes.push(fruit)
  }
}

function addPickupMesh(pickup) {
  if (pickupMeshesById.has(pickup.id)) return

  const kind = pickup.kind
  const isLog = kind === "log"
  const isStone = kind === "stone"
  const isTreeSeed = kind === "tree_seed"
  const isFruit = kind === "fruit"
  const isKnown = isLog || isStone || isTreeSeed || isFruit

  const geometry = isLog
    ? new THREE.CylinderGeometry(0.22, 0.22, 0.9, 8)
    : isTreeSeed
      ? new THREE.ConeGeometry(0.2, 0.42, 8)
      : new THREE.SphereGeometry(0.23, 10, 10)

  const material = new THREE.MeshStandardMaterial({
    color: isLog ? 0x8b5a2b : isStone ? 0x8a949e : isTreeSeed ? 0x9ccc52 : isFruit ? 0xff2f2f : 0x36a7ff,
    emissive: isKnown ? 0x000000 : 0x0f4d8f,
    emissiveIntensity: isKnown ? 0 : 1.1,
    flatShading: true
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(pickup.x, pickup.y, pickup.z)
  mesh.userData.pickupId = pickup.id
  mesh.userData.pickupKind = pickup.kind

  if (isLog) mesh.rotation.z = 1.4
  if (isTreeSeed) mesh.rotation.x = 0.15

  scene.add(mesh)
  pickupMeshesById.set(pickup.id, mesh)
}

function removeWorldObject(worldObjectId) {
  const entry = worldObjectsById.get(worldObjectId)
  if (!entry) return

  scene.remove(entry.mesh)
  worldObjectsById.delete(worldObjectId)

  const index = objects.indexOf(entry.mesh)
  if (index >= 0) {
    objects.splice(index, 1)
  }
}

function removePickupMesh(pickupId) {
  const mesh = pickupMeshesById.get(pickupId)
  if (!mesh) return

  scene.remove(mesh)
  pickupMeshesById.delete(pickupId)
}

function upsertWorldObject(obj) {
  if (!obj || !obj.id) return
  removeWorldObject(obj.id)
  addWorldObject(obj)
}

function resolveWorldObjectFromIntersection(intersection) {
  let node = intersection.object

  while (node) {
    if (node.userData && node.userData.worldObjectId) {
      return {
        id: node.userData.worldObjectId,
        kind: node.userData.worldObjectKind
      }
    }

    node = node.parent
  }

  return null
}

function handleWorldLeftClick(event, camera, canvas) {
  if (!camera || !canvas) return

  const rect = canvas.getBoundingClientRect()
  worldPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  worldPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  worldRaycaster.setFromCamera(worldPointer, camera)
  const intersections = worldRaycaster.intersectObjects(objects, true)

  for (const hit of intersections) {
    const worldObject = resolveWorldObjectFromIntersection(hit)
    if (!worldObject) continue

    if (worldObject.kind === "tree") {
      requestChopTree(worldObject.id)
      break
    }

    if (worldObject.kind === "rock") {
      requestMineRock(worldObject.id)
      break
    }

    if (worldObject.kind === "workbench") {
      openWorkbenchCrafting()
      break
    }

    if (worldObject.kind === "chest") {
      requestOpenChest(worldObject.id)
      break
    }
  }
}

function handleWorldRightClick(event, camera, canvas) {
  if (!camera || !canvas) return

  const rect = canvas.getBoundingClientRect()
  worldPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  worldPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  worldRaycaster.setFromCamera(worldPointer, camera)
  const intersections = worldRaycaster.intersectObjects(objects, true)

  for (const hit of intersections) {
    const worldObject = resolveWorldObjectFromIntersection(hit)
    if (!worldObject) continue
    if (worldObject.kind !== "tree") continue

    requestActionE()
    break
  }
}

function tryPickupByTouch(event, camera, canvas) {
  if (!camera || !canvas || pickupMeshesById.size === 0) return false

  const rect = canvas.getBoundingClientRect()
  worldPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  worldPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  worldRaycaster.setFromCamera(worldPointer, camera)
  const pickupMeshes = Array.from(pickupMeshesById.values())
  const pickupHits = worldRaycaster.intersectObjects(pickupMeshes, false)
  if (!pickupHits.length) return false

  requestActionE()
  return true
}

function applyWorldSnapshot(snapshot) {
  clearWorldVisuals()

  for (const obj of snapshot.objects || []) {
    addWorldObject(obj)
  }

  for (const pickup of snapshot.pickups || []) {
    addPickupMesh(pickup)
  }
}

function applyWorldPatch(patch) {
  if (!patch) return

  if (patch.event === "pickup_collected") {
    removePickupMesh(patch.pickup_id)
  }

  if (patch.event === "tree_shaken") {
    setTreeFruitCount(patch.tree_id, patch.fruit_count)

    for (const pickup of patch.spawned_pickups || []) {
      addPickupMesh(pickup)
    }
  }

  if (patch.event === "object_removed") {
    removeWorldObject(patch.object_id)
    for (const pickup of patch.spawned_pickups || []) {
      addPickupMesh(pickup)
    }
  }

  if (patch.event === "workbench_added" && patch.object) {
    addWorldObject(patch.object)
  }

  if (patch.event === "chest_added" && patch.object) {
    addWorldObject(patch.object)
  }

  if (patch.event === "tree_added" && patch.object) {
    addWorldObject(patch.object)
  }

  if (patch.event === "tree_updated" && patch.object) {
    upsertWorldObject(patch.object)
  }

  if (patch.event === "admin_drop_added" && patch.pickup) {
    addPickupMesh(patch.pickup)
  }
}
