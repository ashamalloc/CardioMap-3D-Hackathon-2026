import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const vessels = {
  LAD: [
    [-0.05, 0.12, 0.25], [-0.02, 0.04, 0.30], [0.02, -0.06, 0.29], [0.03, -0.16, 0.24], [0.02, -0.25, 0.17],
  ],
  LCX: [
    [0.02, 0.20, 0.15], [0.12, 0.20, 0.16], [0.23, 0.16, 0.12], [0.31, 0.08, 0.08], [0.34, -0.01, 0.06],
  ],
  RCA: [
    [-0.02, 0.18, 0.10], [-0.13, 0.19, 0.08], [-0.23, 0.14, 0.05], [-0.29, 0.05, 0.00], [-0.27, -0.07, -0.03],
  ],
}

function probabilityColor(prob) {
  const p = Math.max(0, Math.min(1, prob ?? 0))
  const c = new THREE.Color()
  if (p < 0.5) {
    c.setHSL(0.47, 0.84, 0.54 - p * 0.10)
  } else {
    c.setHSL(0.10 - (p - 0.5) * 0.18, 0.90, 0.54)
  }
  return c
}

export default function HeartViewer({ probabilities = {}, selected, onSelect }) {
  const mount = useRef(null)

  useEffect(() => {
    const el = mount.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#07101d')

    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 100)
    camera.position.set(0.05, 0.15, 2.55)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8))
    renderer.setSize(el.clientWidth, el.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    el.appendChild(renderer.domElement)

    const ambient = new THREE.AmbientLight(0xffffff, 1.8)
    scene.add(ambient)
    const key = new THREE.DirectionalLight(0xaed4ff, 2.3)
    key.position.set(2, 3, 4)
    scene.add(key)
    const fill = new THREE.PointLight(0x64e7d8, 14, 7)
    fill.position.set(-2, -1, 2)
    scene.add(fill)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.minDistance = 1.7
    controls.maxDistance = 4.5
    controls.target.set(0, 0.02, 0.1)

    const anatomy = new THREE.Group()
    scene.add(anatomy)

    const torso = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.84, 1.22, 12, 32),
      new THREE.MeshPhysicalMaterial({ color: 0x172435, transparent: true, opacity: 0.35, roughness: 0.55, metalness: 0.05 })
    )
    torso.scale.set(0.88, 1.08, 0.52)
    torso.position.y = -0.08
    anatomy.add(torso)

    const shoulders = new THREE.Group()
    for (const x of [-0.72, 0.72]) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.32, 24, 24),
        new THREE.MeshStandardMaterial({ color: 0x172435, transparent: true, opacity: 0.28 })
      )
      s.position.set(x, 0.52, 0)
      s.scale.set(1.35, 0.76, 0.65)
      shoulders.add(s)
    }
    anatomy.add(shoulders)

    const heartGroup = new THREE.Group()
    heartGroup.rotation.z = -0.13
    heartGroup.position.set(0.03, 0.10, 0.23)
    anatomy.add(heartGroup)

    for (const [x, y, z, s] of [[-0.13, 0.05, 0, 0.28], [0.15, 0.03, 0.02, 0.33], [0.02, -0.19, 0.02, 0.36]]) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.5, 28, 28),
        new THREE.MeshPhysicalMaterial({ color: 0x7d273c, roughness: 0.72, metalness: 0.02 })
      )
      m.position.set(x, y, z)
      m.scale.set(s, s * 1.12, s * 0.74)
      heartGroup.add(m)
    }

    const aortaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.02, 0.32, 0.12), new THREE.Vector3(0.08, 0.48, 0.14), new THREE.Vector3(0.18, 0.60, 0.16),
      new THREE.Vector3(0.30, 0.56, 0.12), new THREE.Vector3(0.34, 0.44, 0.10),
    ])
    const aorta = new THREE.Mesh(
      new THREE.TubeGeometry(aortaCurve, 32, 0.085, 12, false),
      new THREE.MeshStandardMaterial({ color: 0x9e5365, transparent: true, opacity: 0.78 })
    )
    anatomy.add(aorta)

    const selectable = []
    const vesselGroups = {}
    Object.entries(vessels).forEach(([name, coords]) => {
      const group = new THREE.Group()
      group.name = name
      const points = coords.map(([x, y, z]) => new THREE.Vector3(x, y, z + 0.32))
      const curve = new THREE.CatmullRomCurve3(points)
      const prob = probabilities[name]
      const displayColor = prob == null ? new THREE.Color(0x66798c) : probabilityColor(prob)
      const material = new THREE.MeshStandardMaterial({ color: displayColor, emissive: displayColor, emissiveIntensity: selected === name ? 0.60 : 0.16, roughness: 0.40 })
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, selected === name ? 0.034 : 0.024, 10, false), material)
      tube.userData.vessel = name
      group.add(tube)
      selectable.push(tube)

      coords.forEach(([x, y, z], i) => {
        const node = new THREE.Mesh(
          new THREE.SphereGeometry(selected === name ? 0.052 : 0.042, 18, 18),
          new THREE.MeshStandardMaterial({ color: displayColor, emissive: displayColor, emissiveIntensity: 0.35 })
        )
        node.position.set(x, y, z + 0.32)
        node.userData.vessel = name
        group.add(node)
        selectable.push(node)
        if (i === coords.length - 1) {
          node.scale.setScalar(1.3)
        }
      })
      vesselGroups[name] = group
      anatomy.add(group)
    })

    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()
    const handlePointer = (event) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hit = raycaster.intersectObjects(selectable, false)[0]
      if (hit?.object?.userData?.vessel) onSelect?.(hit.object.userData.vessel)
    }
    renderer.domElement.addEventListener('pointerdown', handlePointer)

    const onResize = () => {
      camera.aspect = el.clientWidth / el.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(el.clientWidth, el.clientHeight)
    }
    window.addEventListener('resize', onResize)

    let frame = 0
    const animate = () => {
      controls.update()
      frame = requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('pointerdown', handlePointer)
      controls.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
      Object.values(vesselGroups).forEach(g => {
        g.traverse(o => {
          if (o.geometry) o.geometry.dispose()
          if (o.material) o.material.dispose()
        })
      })
    }
  }, [probabilities, selected, onSelect])

  return <div className="viewer" ref={mount}><div className="viewer-hint">Drag to rotate · scroll to zoom · click a vessel</div></div>
}
