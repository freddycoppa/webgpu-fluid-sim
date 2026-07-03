import { GPUFluid } from './gpu-fluid.js';
import { mouse } from './mouse.js';
import { randomNiceColor, closeEnoughToZero, randrange } from './utils.js';
import { setupControls } from './controls.js';

let error = null;

async function init() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) {
        alert("Need a browser that supports WebGPU");
        return;
    }

    const canvas = document.querySelector("canvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    device.lost.then(info => {
        alert(`WebGPU device was lost: ${info.message}`);
        if (info.reason !== 'destroyed') init();
    });

    device.addEventListener("uncapturederror", event => {
        error = event.error;
        console.error("Uncaptured WebGPU error:", error);
    });

    syncCanvasSize(canvas);

    const state = setupControls();

    main({ canvas, device, context, format, state });
}

async function main({ canvas, device, context, format, state }) {
    const fluid = new GPUFluid(
        device,
        canvas.width, canvas.height,
        state.dispScale, state.simScale,
        {
            WORKGROUP_SIZE_X: 8,
            WORKGROUP_SIZE_Y: 8,
        }
    );

    await fluid.loadShaders('./shader.wgsl');
    fluid.initPipelines(format);

    const observer = new ResizeObserver(() => {
        if (syncCanvasSize(canvas))
            fluid.resize(canvas.width, canvas.height);
    });

    observer.observe(canvas);

    state.onDispScaleChange(scale => fluid.resizeDisplay(scale));
    state.onSimScaleChange(scale => fluid.resizeSimulation(scale));

    document.addEventListener("keydown", event => {
        if (event.code === "Space")
            fluid.randomSplats(0.15, 2);
    });

    document
        .querySelector("#random-splats-button")
        .addEventListener("click", () => fluid.randomSplats(0.15, 2))
        ;

    fluid.randomSplats(0.15, 2);

    loop(context, fluid, state);
}

function syncCanvasSize(canvas) {
    const dpr = window.devicePixelRatio ?? 1;

    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        return true;
    }

    return false;
}

async function loop(context, fluid, state) {
    let prevMouseXY = null;
    let mouseVelocity = [0, 0];

    let prev_time = null;
    let dt = 1 / 60;

    while (error == null) {
        const timestamp = await new Promise(requestAnimationFrame);

        if (prev_time != null) dt = (timestamp - prev_time) / 1000;
        prev_time = timestamp;

        if (prevMouseXY != null) mouseVelocity = [
            (mouse.x - prevMouseXY[0]) / dt,
            (mouse.y - prevMouseXY[1]) / dt
        ];
        prevMouseXY = [mouse.x, mouse.y];

        if (
            mouse.clicked &&
            !mouseVelocity.every(closeEnoughToZero)
        ) fluid.addSplat(
            [ mouse.x, mouse.y ],
            state.splatRadius,
            randomNiceColor(),
            [ mouseVelocity[0] / 2, mouseVelocity[1] / 2 ],
        );

        fluid.step(context, {
            dt,
            vorticity: state.vorticity,
            densityDiffusion: state.densityDiffusion,
            velocityDiffusion: state.velocityDiffusion,
            contour: state.contour,
            sharpen: state.sharpen,
        });
    }
}

init();
