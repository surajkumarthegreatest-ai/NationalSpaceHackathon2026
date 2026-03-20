/** * Orbital Insight: Phosphor Edition
 * UI/UX Strategy: Desaturated Accents & Fluid Motion
 */

const canvas = document.getElementById('orbitCanvas');
const ctx = canvas.getContext('2d');

const COLORS = {
    background: 'rgba(15, 23, 42, 0.2)', // 20% opacity for the "Ghost Trace"
    satellite: 'rgba(16, 185, 129, 1)', 
    satGlow: 'rgba(16, 185, 129, 0.3)',
    debris: '#EF4444' // Desaturated red to prevent vibration [cite: 16, 17]
};

// --- PRE-RENDERED ASSETS (Off-screen Canvas) ---
const satCache = document.createElement('canvas');
satCache.width = 12; satCache.height = 12;
const sCtx = satCache.getContext('2d');
sCtx.fillStyle = COLORS.satGlow;
sCtx.beginPath(); sCtx.arc(6, 6, 6, 0, Math.PI * 2); sCtx.fill();
sCtx.fillStyle = COLORS.satellite;
sCtx.beginPath(); sCtx.arc(6, 6, 2, 0, Math.PI * 2); sCtx.fill();

let telemetry = [];

async function sync() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        telemetry = data.states || [];
    } catch (e) { console.error("Sync Lost"); }
}

function renderLoop() {
    // UI/UX FIX: Zero-Memory Ghost Trace
    // Instead of clearRect, we "fade" the previous frame 
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < telemetry.length; i++) {
        // [id, isDebris, rx, ry, rz, vx, vy, vz]
        const [id, isDebris, rx, ry] = telemetry[i];
        
        // Simple projection for 10,000 objects
        const x = (rx / 15000 + 0.5) * canvas.width;
        const y = (ry / 15000 + 0.5) * canvas.height;

        if (isDebris) {
            ctx.fillStyle = COLORS.debris;
            ctx.fillRect(x, y, 1.5, 1.5); // Rects are cheaper than arcs
        } else {
            ctx.drawImage(satCache, x - 6, y - 6); // Pre-rendered glow
        }
    }

    requestAnimationFrame(renderLoop);
}

// Initialization
setInterval(sync, 1000); 
renderLoop();