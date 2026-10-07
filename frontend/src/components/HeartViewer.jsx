import React, { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

const VESSEL_FULL_NAMES = {
  LAD: 'Left Anterior Descending',
  LCX: 'Left Circumflex',
  RCA: 'Right Coronary Artery',
}

// Medical color scale for stenosis risk probability
function probabilityColor(prob) {
  if (prob == null) return new THREE.Color(0x66798c)
  const p = Math.max(0, Math.min(1, prob))
  const color = new THREE.Color()
  if (p < 0.33) {
    // Low risk: Teal / Emerald green
    color.setHSL(0.46, 0.85, 0.50)
  } else if (p < 0.66) {
    // Intermediate risk: Amber / Gold
    color.setHSL(0.12, 0.95, 0.52)
  } else {
    // High risk: Warning Crimson / Red
    color.setHSL(0.98, 0.88, 0.52)
  }
  return color
}

// Camera angle presets for clinical inspection
const CAMERA_PRESETS = {
  reset: {
    pos: [0.3, 0.2, 2.7],
    target: [0, 0, 0],
    label: 'Overview',
  },
  LAD: {
    pos: [0.35, 0.15, 2.65],
    target: [0.25, 0.05, 0.2],
    label: 'Anterior (LAD)',
  },
  LCX: {
    pos: [2.55, -0.25, -1.05],
    target: [0.15, -0.15, 0.0],
    label: 'Lateral (LCX)',
  },
  RCA: {
    pos: [-2.45, -0.65, 1.35],
    target: [-0.2, -0.2, 0.15],
    label: 'Right/Inferior (RCA)',
  },
}

export default function HeartViewer({ probabilities = {}, selected = 'LAD', onSelect }) {
  const mount = useRef(null)
  const [loading, setLoading] = useState(true)
  const [loadingText, setLoadingText] = useState('Loading 3D Anatomy...')
  const [heartMode, setHeartMode] = useState('translucent') // 'translucent', 'xray', 'solid', 'hidden'
  const [hoveredVessel, setHoveredVessel] = useState(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // References to keep Three.js entities accessible across renders
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const heartMeshRef = useRef(null)
  const vesselMeshesRef = useRef({})
  const targetCamPosRef = useRef(new THREE.Vector3(...CAMERA_PRESETS.reset.pos))
  const targetCamLookRef = useRef(new THREE.Vector3(...CAMERA_PRESETS.reset.target))
  const isTransitioningRef = useRef(false)

  // Camera preset trigger
  const moveToPreset = useCallback((presetKey) => {
    const preset = CAMERA_PRESETS[presetKey]
    if (!preset || !cameraRef.current || !controlsRef.current) return
    targetCamPosRef.current.set(...preset.pos)
    targetCamLookRef.current.set(...preset.target)
    isTransitioningRef.current = true
  }, [])

  // Update vessel colors and emission when probabilities or selected changes
  useEffect(() => {
    Object.entries(vesselMeshesRef.current).forEach(([name, mesh]) => {
      if (!mesh || !mesh.material) return
      const prob = probabilities[name]
      const col = probabilityColor(prob)
      const isSelected = selected === name

      mesh.material.color.copy(col)
      mesh.material.emissive.copy(col)
      mesh.material.emissiveIntensity = isSelected ? 0.75 : 0.22
    })
  }, [probabilities, selected])

  // Update heart opacity when heartMode changes
  useEffect(() => {
    if (!heartMeshRef.current) return
    const mat = heartMeshRef.current.material
    if (!mat) return

    if (heartMode === 'hidden') {
      heartMeshRef.current.visible = false
    } else {
      heartMeshRef.current.visible = true
      let opacity = 0.65
      if (heartMode === 'xray') opacity = 0.28
      if (heartMode === 'solid') opacity = 0.98

      mat.transparent = opacity < 0.95
      mat.opacity = opacity
      mat.depthWrite = opacity > 0.9
      mat.needsUpdate = true
    }
  }, [heartMode])


  // Mount Three.js canvas & load models
  useEffect(() => {
    const el = mount.current
    if (!el) return

    const scene = new THREE.Scene()
    sceneRef.current = scene
    scene.background = new THREE.Color('#07101d')

    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 100)
    camera.position.set(...CAMERA_PRESETS.reset.pos)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(el.clientWidth, el.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    el.appendChild(renderer.domElement)

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4)
    scene.add(ambientLight)

    const keyLight = new THREE.DirectionalLight(0xdbeafe, 2.6)
    keyLight.position.set(3, 4, 4)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0x7dd3fc, 1.6)
    fillLight.position.set(-3, -1, 3)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0xf472b6, 1.2)
    rimLight.position.set(0, 4, -4)
    scene.add(rimLight)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 1.2
    controls.maxDistance = 5.0
    controls.target.set(...CAMERA_PRESETS.reset.target)
    controlsRef.current = controls

    // Anatomical root group
    const anatomyGroup = new THREE.Group()
    scene.add(anatomyGroup)


    // Heart + Vessels Group
    const heartAssembly = new THREE.Group()
    anatomyGroup.add(heartAssembly)

    // Raw models container (pre-centered and re-oriented)
    const orientedGroup = new THREE.Group()
    // Body coordinate to Three.js conversion:
    // Superior (Body Z) -> Up (Three.js +Y)
    // Anterior (Body -Y) -> Front (Three.js +Z)
    orientedGroup.rotation.x = -Math.PI / 2
    // Slight anatomical tilt around long cardiac axis
    orientedGroup.rotation.z = -0.12
    orientedGroup.scale.setScalar(0.019) // 118mm -> ~2.24 units in scene
    heartAssembly.add(orientedGroup)

    const rawModelGroup = new THREE.Group()
    // Center of mass offset calculated from GLB accessors:
    rawModelGroup.position.set(-22.8877, 123.9158, -1233.9350)
    orientedGroup.add(rawModelGroup)

    const loader = new GLTFLoader()
    const selectable = []
    let isDisposed = false

    const modelFiles = [
      { name: 'heart', url: '/models/heart.glb' },
      { name: 'LAD', url: '/models/LAD.glb' },
      { name: 'LCX', url: '/models/LCX.glb' },
      { name: 'RCA', url: '/models/RCA.glb' },
    ]

    setLoadingText('Loading anatomical meshes (Heart, LAD, LCX, RCA)...')

    Promise.all(
      modelFiles.map(
        (item) =>
          new Promise((resolve, reject) => {
            loader.load(
              item.url,
              (gltf) => resolve({ name: item.name, gltf }),
              undefined,
              (err) => reject(new Error(`Failed to load ${item.name}: ${err.message}`))
            )
          })
      )
    )
      .then((loadedResults) => {
        if (isDisposed) return

        loadedResults.forEach(({ name, gltf }) => {
          const root = gltf.scene

          root.traverse((child) => {
            if (child.isMesh) {
              // Ensure smooth normals for realistic PBR lighting
              child.geometry.computeVertexNormals()

              if (name === 'heart') {
                // Heart Myocardium Material
                const heartMat = new THREE.MeshPhysicalMaterial({
                  color: new THREE.Color('#781e30'),
                  roughness: 0.55,
                  metalness: 0.05,
                  clearcoat: 0.35,
                  clearcoatRoughness: 0.25,
                  transparent: true,
                  opacity: 0.65,
                  depthWrite: true,
                  polygonOffset: true,
                  polygonOffsetFactor: 1,
                  polygonOffsetUnits: 1,
                })
                child.material = heartMat
                heartMeshRef.current = child
              } else {
                // Coronary Artery Meshes (LAD, LCX, RCA)
                const prob = probabilities[name]
                const color = probabilityColor(prob)
                const isSelected = selected === name

                const vesselMat = new THREE.MeshStandardMaterial({
                  color: color,
                  emissive: color,
                  emissiveIntensity: isSelected ? 0.75 : 0.22,
                  roughness: 0.35,
                  metalness: 0.12,
                })

                child.material = vesselMat
                child.userData.vessel = name
                vesselMeshesRef.current[name] = child
                selectable.push(child)
              }
            }
          })

          rawModelGroup.add(root)
        })

        setLoading(false)
      })
      .catch((err) => {
        console.error('Error loading 3D heart models:', err)
        setLoadingText('Error loading 3D models: ' + err.message)
      })

    // Pointer events: Raycasting for hover and click
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    const onPointerMove = (e) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(selectable, false)

      if (hits.length > 0 && hits[0].object?.userData?.vessel) {
        const vessel = hits[0].object.userData.vessel
        setHoveredVessel(vessel)
        setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top + 15 })
        renderer.domElement.style.cursor = 'pointer'
      } else {
        setHoveredVessel(null)
        renderer.domElement.style.cursor = 'default'
      }
    }

    const onPointerDown = (e) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(selectable, false)

      if (hits.length > 0 && hits[0].object?.userData?.vessel) {
        const vessel = hits[0].object.userData.vessel
        onSelect?.(vessel)
      }
    }

    renderer.domElement.addEventListener('pointermove', onPointerMove)
    renderer.domElement.addEventListener('pointerdown', onPointerDown)

    const onResize = () => {
      if (!el) return
      camera.aspect = el.clientWidth / el.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(el.clientWidth, el.clientHeight)
    }
    window.addEventListener('resize', onResize)

    // Animation render loop
    let animId = 0
    let clock = new THREE.Clock()

    const animate = () => {
      animId = requestAnimationFrame(animate)
      const elapsed = clock.getElapsedTime()

      // Smooth camera interpolation when switching views
      if (isTransitioningRef.current) {
        camera.position.lerp(targetCamPosRef.current, 0.08)
        controls.target.lerp(targetCamLookRef.current, 0.08)
        if (
          camera.position.distanceTo(targetCamPosRef.current) < 0.01 &&
          controls.target.distanceTo(targetCamLookRef.current) < 0.01
        ) {
          isTransitioningRef.current = false
        }
      }

      // Subtle emissive pulse on selected vessel
      if (selected && vesselMeshesRef.current[selected]) {
        const selectedMesh = vesselMeshesRef.current[selected]
        if (selectedMesh.material) {
          const pulse = 0.65 + Math.sin(elapsed * 4.5) * 0.25
          selectedMesh.material.emissiveIntensity = pulse
        }
      }

      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      isDisposed = true
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      controls.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement)
      }
    }
  }, []) // Mount once

  const hoveredProb = hoveredVessel ? probabilities[hoveredVessel] : null

  return (
    <div className="viewer" ref={mount}>
      {/* Loading Overlay */}
      {loading && (
        <div className="viewer-loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-title">Interactive 3D Anatomical Map</div>
          <div className="loading-text">{loadingText}</div>
        </div>
      )}

      {/* Top Left Navigation Hint */}
      <div className="viewer-hint">
        Click vessel or buttons to inspect · Drag to rotate · Scroll to zoom
      </div>

      {/* Top Right Anatomical View Presets */}
      <div className="viewer-controls-top">
        <div className="preset-group">
          <span className="preset-label">View:</span>
          {Object.entries(CAMERA_PRESETS).map(([key, item]) => (
            <button
              key={key}
              className={`preset-btn ${selected === key ? 'active' : ''}`}
              onClick={() => {
                moveToPreset(key)
                if (key !== 'reset') onSelect?.(key)
              }}
              title={`Switch camera to ${item.label}`}
            >
              {key === 'reset' ? 'Overview' : key}
            </button>
          ))}
        </div>
      </div>

      {/* Bottom Viewer Controls (Heart Translucency) */}
      <div className="viewer-controls-bottom">
        <div className="mode-group">
          <span className="control-label">Myocardium:</span>
          <button
            className={`mode-btn ${heartMode === 'translucent' ? 'active' : ''}`}
            onClick={() => setHeartMode('translucent')}
            title="Semi-transparent cardiac tissue"
          >
            Translucent
          </button>
          <button
            className={`mode-btn ${heartMode === 'xray' ? 'active' : ''}`}
            onClick={() => setHeartMode('xray')}
            title="X-Ray mode for deep vessel clarity"
          >
            X-Ray
          </button>
          <button
            className={`mode-btn ${heartMode === 'solid' ? 'active' : ''}`}
            onClick={() => setHeartMode('solid')}
            title="Solid anatomical heart"
          >
            Solid
          </button>
          <button
            className={`mode-btn ${heartMode === 'hidden' ? 'active' : ''}`}
            onClick={() => setHeartMode('hidden')}
            title="Hide heart muscle to display arteries only"
          >
            Arteries Only
          </button>
        </div>
      </div>

      {/* Hover Tooltip */}
      {hoveredVessel && (
        <div
          className="viewer-tooltip"
          style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}
        >
          <div className="tooltip-title">{hoveredVessel}</div>
          <div className="tooltip-sub">{VESSEL_FULL_NAMES[hoveredVessel]}</div>
          <div className="tooltip-risk">
            Risk:{' '}
            <b>{hoveredProb != null ? `${(hoveredProb * 100).toFixed(1)}%` : 'Not evaluated'}</b>
          </div>
        </div>
      )}
    </div>
  )
}
