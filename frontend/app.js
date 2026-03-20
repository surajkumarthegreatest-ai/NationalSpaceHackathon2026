/** * Orbital Insight: WebGL Edition
 * Strategy: GPU-Accelerated Fluid Motion & Desaturated UI
 */

const {DeckGL, ScatterplotLayer, OrbitView} = deck;

// Initialization: DeckGL Container
const deckgl = new DeckGL({
    container: 'map-viewport',
    views: [new OrbitView({
        orbitAxis: 'Y',
        near: 0.1,
        far: 100000
    })],
    initialViewState: {
        target: [0, 0, 0],
        rotationX: 30,
        rotationOrbit: -45,
        zoom: 2 // Adjust based on your ECI coordinate scale
    },
    controller: true,
    // Set background to our UI/UX dark mode standard
    style: {backgroundColor: '#0F172A'}
});

let telemetry = [];

async function syncTelemetry() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        telemetry = data.states || [];
        
        // Remove skeleton loaders once data arrives [cite: 4]
        document.getElementById('loader').style.display = 'none';
        document.getElementById('stats').style.display = 'block';
        
        renderWebGL();
    } catch (e) {
        console.error("Telemetry Link Failure");
    }
}

function renderWebGL() {
    // Deck.gl ScatterplotLayer maps the array directly to the GPU
    const layers = [
        new ScatterplotLayer({
            id: 'orbital-objects',
            data: telemetry,
            // Accessors: map the [id, isDebris, rx, ry, rz] format to WebGL
            getPosition: d => [d[2], d[3], d[4]], 
            getFillColor: d => d[1] ? [239, 68, 68] : [16, 185, 129], // Ruby for Debris, Emerald for Sats
            getRadius: d => d[1] ? 100 : 300, // Scale based on your units (km)
            radiusMinPixels: 2,
            radiusMaxPixels: 10,
            updateTriggers: {
                getPosition: telemetry // Force GPU update when data changes
            }
        })
    ];

    deckgl.setProps({layers});
}

// Global Command Palette [cite: 29]
window.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        prompt("Enter Satellite ID to Track in 3D:");
    }
});

// Sync at 10Hz to match the new high-performance backend capabilities
setInterval(syncTelemetry, 100);