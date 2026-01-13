
import React, { useRef, Suspense, Component } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Float, useTexture } from '@react-three/drei';

// ERROR BOUNDARY COMPONENT
class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("Scene3D Error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '20px', color: 'red', background: 'rgba(0,0,0,0.8)', position: 'absolute', top: '100px', zIndex: 999 }}>
                    <h2>3D Scene Crash:</h2>
                    <pre>{this.state.error && this.state.error.toString()}</pre>
                </div>
            );
        }
        return this.props.children;
    }
}

const HeroCard = () => {
    const meshRef = useRef();

    useFrame((state) => {
        const t = state.clock.getElapsedTime();
        if (meshRef.current) {
            meshRef.current.rotation.y = Math.sin(t * 0.1) * 0.2;
            meshRef.current.rotation.x = Math.cos(t * 0.1) * 0.1;
        }
    });

    return (
        <Float speed={1.5} rotationIntensity={0.5} floatIntensity={1}>
            <mesh ref={meshRef} scale={[3, 1.8, 0.1]}>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial
                    color="#0a0a0a"
                    roughness={0.2}
                    metalness={0.9}
                    emissive="#1a1a1a"
                    emissiveIntensity={0.2}
                />
            </mesh>
        </Float>
    );
};

const Scene3D = () => {
    return (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0, pointerEvents: 'none' }}>
            <ErrorBoundary>
                <Canvas
                    dpr={[1, 1.5]} // Clamp resolution for performance
                    gl={{ powerPreference: "high-performance", antialias: true, alpha: true }}
                    camera={{ position: [0, 0, 5], fov: 45 }}
                    performance={{ min: 0.5 }} // Allow downgrading quality on slow devices
                >
                    <Suspense fallback={null}>
                        <ambientLight intensity={0.5} />
                        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} color="#ffffff" />
                        <pointLight position={[-10, -10, -10]} intensity={1} color="#FFD700" />

                        {/* Reduced star count for better FPS */}
                        <Stars radius={100} depth={50} count={1000} factor={4} saturation={0} fade speed={1} />

                        <HeroCard />

                        <fog attach="fog" args={['#020202', 5, 20]} />
                    </Suspense>
                </Canvas>
            </ErrorBoundary>
        </div>
    );
};

export default Scene3D;
