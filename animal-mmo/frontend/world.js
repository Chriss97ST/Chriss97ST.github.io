let scene
let objects = []

const worldObjectsById = new Map()
const pickupMeshesById = new Map()
const worldRaycaster = new THREE.Raycaster()
const worldPointer = new THREE.Vector2()

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

  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22 * obj.scale, 0.26 * obj.scale, 2.2 * obj.scale, 6),
    new THREE.MeshStandardMaterial({ color: 0x8b5a2b, flatShading: true })
  )

  const leaves = new THREE.Mesh(
    new THREE.ConeGeometry(1.45 * obj.scale, 2.4 * obj.scale, 7),
    new THREE.MeshStandardMaterial({ color: 0x2fb463, flatShading: true })
  )

  trunk.position.y = 1.1 * obj.scale
  leaves.position.y = 2.6 * obj.scale

  group.add(trunk)
  group.add(leaves)

  group.position.set(obj.x, obj.y - 1, obj.z)
  group.rotation.y = obj.rotation
  group.userData.worldObjectId = obj.id
  group.userData.worldObjectKind = "tree"
  group.userData.collisionRadius = 1.45 * obj.scale

  const fruitNodes = []
  for (let i = 0; i < obj.fruit_count; i++) {
    const fruit = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 8, 8),
      new THREE.MeshStandardMaterial({ color: 0xff6644, flatShading: true })
    )

    const ring = (i % 3) + 1
    const angle = (i * 1.8) + obj.id * 0.37
    const radius = 0.45 + ring * 0.22
    fruit.position.set(
      Math.cos(angle) * radius,
      (2.0 + ring * 0.24) * obj.scale,
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
    scale: obj.scale
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

function addWorldObject(obj) {
  if (obj.kind === "tree") {
    createTreeMesh(obj)
  } else if (obj.kind === "rock") {
    createRockMesh(obj)
  } else if (obj.kind === "workbench") {
    createWorkbenchMesh(obj)
  }
}

function setTreeFruitCount(treeId, fruitCount) {
  const entry = worldObjectsById.get(treeId)
  if (!entry || entry.kind !== "tree") return

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
      (2.0 + ring * 0.24) * (entry.scale || 1),
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

  const geometry = isLog
    ? new THREE.CylinderGeometry(0.22, 0.22, 0.9, 8)
    : new THREE.SphereGeometry(0.23, 10, 10)

  const material = new THREE.MeshStandardMaterial({
    color: isLog ? 0x8b5a2b : isStone ? 0x8a949e : 0xff2f2f,
    flatShading: true
  })

  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(pickup.x, pickup.y, pickup.z)
  mesh.userData.pickupId = pickup.id
  mesh.userData.pickupKind = pickup.kind

  if (isLog) mesh.rotation.z = 1.4

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
}
