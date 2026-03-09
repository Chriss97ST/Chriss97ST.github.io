const serverAnimals = new Map()

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
    scene.remove(entry.mesh)
    serverAnimals.delete(id)
  }
}

function updateAnimals(delta) {
  for (const entry of serverAnimals.values()) {
    const mesh = entry.mesh
    mesh.position.lerp(entry.targetPos, Math.min(1, delta * 10))

    if (Math.abs(entry.dirX) + Math.abs(entry.dirZ) > 0.001) {
      mesh.rotation.y = Math.atan2(entry.dirX, entry.dirZ) + Math.PI
    }

    const anim = mesh.userData.anim
    if (!anim) continue

    const moving = entry.speed > 0.005
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
}
