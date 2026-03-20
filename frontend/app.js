/** * Orbital Insight: Production Visualizer
 * Strategy: Fluid Motion & Efficient Navigation [cite: 21, 28]
 */
const canvas = document.getElementById('groundTrackCanvas');
const ctx = canvas.getContext('2d');
const viewport = document.getElementById('map-viewport');

// PRE-RENDERED ASSETS: UI/UX Asset Optimization [cite: 6]
const satCache = document.createElement('canvas');
satCache.width = 12; satCache.height = 12;
const sCtx = satCache.getContext('2d');
sCtx.fillStyle = 'rgba(16, 185, 129, 0.3)';
sCtx.beginPath(); sCtx.arc(6, 6, 6, 0, Math.PI * 2); sCtx.fill();
sCtx.fillStyle = '#10B981';
sCtx.beginPath(); sCtx.arc(6, 6, 2, 0, Math.PI * 2); sCtx.fill();

let telemetry = [];

// FIX 2: Layout Awareness - Size to parent, not window [cite: 32]
function syncSize() {
    canvas.width = viewport.clientWidth;
    canvas.height = viewport.clientHeight;
}

async function syncTelemetry() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        telemetry = data.states || [];
        // Remove skeleton loaders once data arrives [cite: 4]
        document.querySelectorAll('.skeleton').forEach(el => el.classList.remove('skeleton'));
    } catch (e) { console.error("Telemetry Link Failure"); }
}

function render() {
    // FIX 1: Ghost Trace that respects the background grid [cite: 23, 24]
    // We use clearRect with a low globalAlpha to "fade" but keep transparency
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'; // Fades existing pixels
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over';

    for (let i = 0; i < telemetry.length; i++) {
        const [id, isDebris, rx, ry] = telemetry[i];
        const x = (rx / 15000 + 0.5) * canvas.width;
        const y = (ry / 15000 + 0.5) * canvas.height;

        if (isDebris) {
            ctx.fillStyle = '#EF4444'; // Desaturated Ruby [cite: 16]
            ctx.fillRect(x, y, 1.5, 1.5);
        } else {
            ctx.drawImage(satCache, x - 6, y - 6); // Bit-Blt Glow [cite: 5]
        }
    }
    requestAnimationFrame(render);
}

// UI/UX: Command Palette (Ctrl+K) [cite: 29]
window.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        prompt("Enter Satellite ID:");
    }
});

window.addEventListener('resize', syncSize);
syncSize();
setInterval(syncTelemetry, 1000);
render();