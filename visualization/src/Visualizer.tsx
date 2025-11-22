import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";


const AttentionHead = ({ data, bufferRef }: { data: number[][], bufferRef: React.MutableRefObject<Float32Array | undefined> }) => {
	const size = data.length;
	const bufferSize = size * 3; // 3x larger buffer for scrolling effect
	
	const texture = useMemo(() => {
		const array = new Float32Array(bufferSize * bufferSize * 4);
		
		// If we have previous buffer data, shift it up and left by one cell
		if (bufferRef.current && bufferRef.current.length === array.length) {
			// Shift existing data up and left
			for (let i = 0; i < bufferSize - 1; i++) {
				for (let j = 0; j < bufferSize - 1; j++) {
					const srcIdx = ((i + 1) * bufferSize + (j + 1)) * 4;
					const dstIdx = (i * bufferSize + j) * 4;
					array[dstIdx] = bufferRef.current[srcIdx];
					array[dstIdx + 1] = bufferRef.current[srcIdx + 1];
					array[dstIdx + 2] = bufferRef.current[srcIdx + 2];
					array[dstIdx + 3] = bufferRef.current[srcIdx + 3];
				}
			}
		}
		
		// Normalize new data
		let maxVal = 0;
		for(let i=0; i<size; i++) for(let j=0; j<size; j++) maxVal = Math.max(maxVal, data[i][j]);
		
		// Write new data to bottom-right corner
		const offsetY = bufferSize - size;
		const offsetX = bufferSize - size;
		for (let i = 0; i < size; i++) {
			for (let j = 0; j < size; j++) {
				const val = data[i][j] / (maxVal || 1);
				const idx = ((offsetY + i) * bufferSize + (offsetX + j)) * 4;
				array[idx] = val;     // R
				array[idx + 1] = val; // G
				array[idx + 2] = val; // B
				array[idx + 3] = 1;   // A
			}
		}
		
		// Update the bufferRef with the new array
		bufferRef.current = array;
		
		const tex = new THREE.DataTexture(array, bufferSize, bufferSize, THREE.RGBAFormat, THREE.FloatType);
		tex.needsUpdate = true;
		tex.magFilter = THREE.NearestFilter;
		
		return tex;
	}, [data, bufferRef, size, bufferSize]);

	const material = useMemo(() => {
		return new THREE.ShaderMaterial({
			uniforms: {
				uTexture: { value: texture },
				uTime: { value: 0 }
			},
			vertexShader: `
				varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: `
				uniform sampler2D uTexture;
				uniform float uTime;
				varying vec2 vUv;
				void main() {
					float diag = vUv.x + (1.0 - vUv.y);
					if (diag > uTime) {
						gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
					} else {
						vec4 color = texture2D(uTexture, vec2(vUv.x, 1.0 - vUv.y));
						float val = color.r;
						val = pow(val, 0.9); 
						val *= 0.7;
						gl_FragColor = vec4(val, val, val, 1.0);
					}
				}
			`,
			transparent: true
		});
	}, [texture]);

	useFrame((state) => {
		const t = state.clock.getElapsedTime() * .5;
		material.uniforms.uTime.value = t;
	});

	return (
		<mesh position={[0, 0, 0]}>
			<planeGeometry args={[2, 2]} />
			<primitive object={material} attach="material" />
		</mesh>
	);
}

const HeadContainer = ({ data, bufferRef }: { data: number[][], bufferRef: React.MutableRefObject<Float32Array | undefined> }) => {
    return (
        <div className="head-wrapper" style={{ position: 'relative', width: '100%', aspectRatio: '1/1', boxSizing: 'border-box' }}>
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
                <Canvas>
                    <OrthographicCamera makeDefault manual left={-1} right={1} top={1} bottom={-1} position={[0, 0, 10]} />
                    <color attach="background" args={["#000"]} />
                    <AttentionHead data={data} bufferRef={bufferRef} />
                </Canvas>
            </div>
        </div>
    )
}

export const AttentionViz = ({ data }: { data: number[][][] | null }) => {
	// Create refs to store buffer data for each head
	const bufferRefs = useRef<React.MutableRefObject<Float32Array | undefined>[]>([]);
	
	if (!data) return <div className="placeholder"></div>;
	
	// Initialize buffer refs array if not already done
	if (bufferRefs.current.length !== data.length) {
		bufferRefs.current = new Array(data.length).fill(null).map(() => ({ current: undefined }));
	}

	return (
		<div className="viz-box" style={{ 
			width: '100vw', 
			height: '100vh', 
			overflow: 'hidden',
			position: 'fixed',
			top: 0,
			left: 0
		}}>
            <div style={{ 
                display: 'grid', 
                gridTemplateColumns: window.innerWidth > window.innerHeight ? 'repeat(4, 1fr)' : 'repeat(3, 1fr)',
                gap: '0',
                width: '100%',
                height: '100%'
            }}>
                {data.map((headData, i) => (
					<HeadContainer 
						key={`head-${i}`} 
						data={headData}
						bufferRef={bufferRefs.current[i]}
					/>
				))}
            </div>
		</div>
	);
};
