import { PingPongTexture, PingPongTexturePair } from './texture.js';
import { Uniforms } from './uniforms.js';
import { randomNiceColor, randrange } from './utils.js';

export class GPUFluid {
    constructor(device, width, height, dispScale, simScale, constants) {
        this.device = device;
        this.width = width;
        this.height = height;
        this.dispScale = dispScale;
        this.simScale = simScale;
        this.dispWidth = this.width / this.dispScale;
        this.dispHeight = this.height / this.dispScale;
        this.simWidth = this.width / this.simScale;
        this.simHeight = this.height / this.simScale;
        this.constants = constants;

        this.splatQueue = [];

        this.uniforms = new Uniforms(device);
        this.density  = new PingPongTexture(
            "density",
            device,
            this.dispWidth, this.dispHeight,
            GPUShaderStage.FRAGMENT,
            "rgba16float",
            "float"
        );
        this.velocity = new PingPongTexturePair(
            "velocity",
            device,
            this.simWidth + 1, this.simHeight,
            this.simWidth, this.simHeight + 1
        );
        this.curl = new PingPongTexture(
            "curl",
            device,
            this.simWidth + 1, this.simHeight + 1
        );

        this.pipelineLayout = device.createPipelineLayout({
            label: "pipeline layout",
            bindGroupLayouts: [
                this.uniforms.getLayout(),
                this.density .getLayout(),
                this.curl    .getLayout(),
                this.velocity.getLayout(),
            ]
        });
    }

    resizeDisplay(dispScale) {
        if (this.dispScale == dispScale) return;

        this.dispScale = dispScale ?? this.dispScale;
        this.dispWidth = this.width / this.dispScale;
        this.dispHeight = this.height / this.dispScale;
        this.density.resize(this.dispWidth, this.dispHeight);
    }

    resizeSimulation(simScale) {
        if (this.simScale == simScale) return;

        this.simScale = simScale ?? this.simScale;
        this.simWidth = this.width / this.simScale;
        this.simHeight = this.height / this.simScale;
        this.velocity.resize(
            this.simWidth + 1, this.simHeight,
            this.simWidth, this.simHeight + 1
        );
        this.curl.resize(this.simWidth + 1, this.simHeight + 1);
    }

    resize(width, height) {
        if (this.width == width && this.height == height) return;

        this.width = width;
        this.height = height;
        this.resizeDisplay();
        this.resizeSimulation();        
    }

    async loadShaders(path) {
        this.shaderModule = this.device.createShaderModule({
            code: await fetch(path).then(res => res.text())
        });

        const info = await this.shaderModule.getCompilationInfo();

        for (const message of info.messages) {
            console.log(
                `${message.type}: line ${message.lineNum}, col ${message.linePos}: ${message.message}`
            );
        }
    }

    createComputePipeline(entryPoint) {
        return this.device.createComputePipeline({
            label: entryPoint,
            layout: this.pipelineLayout,
            compute: {
                module: this.shaderModule,
                entryPoint,
                constants: this.constants
            }
        });
    }

    initPipelines(canvasFormat) {
        this.splatDensityPipeline    = this.createComputePipeline("splatDensity"   );
        this.splatUPipeline          = this.createComputePipeline("splatU"         );
        this.splatVPipeline          = this.createComputePipeline("splatV"         );
        this.advectDensityPipeline   = this.createComputePipeline("advectDensity"  );
        this.advectUPipeline         = this.createComputePipeline("advectU"        );
        this.advectVPipeline         = this.createComputePipeline("advectV"        );
        this.computeCurlPipeline     = this.createComputePipeline("computeCurl"    );
        this.applyVorticityUPipeline = this.createComputePipeline("applyVorticityU");
        this.applyVorticityVPipeline = this.createComputePipeline("applyVorticityV");
        this.projectRedPipeline      = this.createComputePipeline("projectRed"     );
        this.projectBlackPipeline    = this.createComputePipeline("projectBlack"   );

        this.renderPipeline = this.device.createRenderPipeline({
            label: "render",
            layout: this.pipelineLayout,
            vertex: {
                module: this.shaderModule,
                entryPoint: "vs",
            },
            fragment: {
                module: this.shaderModule,
                entryPoint: "fs",
                targets: [{ format: canvasFormat }],
            }
        });
    }

    setAllBindGroups(pass) {
        pass.setBindGroup(0, this.uniforms.getBindGroup());
        pass.setBindGroup(1, this.density .getBindGroup());
        pass.setBindGroup(2, this.curl.getBindGroup());
        pass.setBindGroup(3, this.velocity.getBindGroup());
    }

    splat(pass) {
        const {
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        } = this.constants;

        pass.setPipeline(this.splatDensityPipeline);
        pass.dispatchWorkgroups(
            Math.ceil( this.dispWidth / WORKGROUP_SIZE_X),
            Math.ceil(this.dispHeight / WORKGROUP_SIZE_Y),
        );
        this.density.swap();
        pass.setBindGroup(1, this.density.getBindGroup());

        pass.setPipeline(this.splatUPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth + 1) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight   ) / WORKGROUP_SIZE_Y),
        );
        pass.setPipeline(this.splatVPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth     ) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight + 1) / WORKGROUP_SIZE_Y),
        );
        this.velocity.swap();
        pass.setBindGroup(3, this.velocity.getBindGroup());
    }

    advectDensity(pass) {
        const {
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        } = this.constants;

        pass.setPipeline(this.advectDensityPipeline);
        pass.dispatchWorkgroups(
            Math.ceil( this.dispWidth / WORKGROUP_SIZE_X),
            Math.ceil(this.dispHeight / WORKGROUP_SIZE_Y),
        );
        this.density.swap();
        pass.setBindGroup(1, this.density.getBindGroup());
    }

    advectVelocity(pass) {
        const {
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        } = this.constants;

        pass.setPipeline(this.advectUPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth + 1) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight   ) / WORKGROUP_SIZE_Y),
        );
        pass.setPipeline(this.advectVPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth     ) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight + 1) / WORKGROUP_SIZE_Y),
        );
        this.velocity.swap();
        pass.setBindGroup(3, this.velocity.getBindGroup());
    }

    confineVorticity(pass) {
        const {
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        } = this.constants;

        pass.setPipeline(this.computeCurlPipeline);
        pass.dispatchWorkgroups(
            Math.ceil(( this.simWidth + 1) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight + 1) / WORKGROUP_SIZE_Y),
        );
        this.curl.swap();
        pass.setBindGroup(2, this.curl.getBindGroup());

        pass.setPipeline(this.applyVorticityUPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth + 1) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight   ) / WORKGROUP_SIZE_Y),
        );
        pass.setPipeline(this.applyVorticityVPipeline);
        pass.dispatchWorkgroups(
            Math.ceil((this.simWidth     ) / WORKGROUP_SIZE_X),
            Math.ceil((this.simHeight + 1) / WORKGROUP_SIZE_Y),
        );
        this.velocity.swap();
        pass.setBindGroup(3, this.velocity.getBindGroup());
    }

    project(pass, n = 40) {
        const {
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        } = this.constants;

        for (let i = 0; i < n; i++) {
            pass.setPipeline(this.projectRedPipeline);
            pass.dispatchWorkgroups(
                Math.ceil( this.simWidth / WORKGROUP_SIZE_X),
                Math.ceil(this.simHeight / WORKGROUP_SIZE_Y),
            );
            this.velocity.swap();
            pass.setBindGroup(3, this.velocity.getBindGroup());

            pass.setPipeline(this.projectBlackPipeline);
            pass.dispatchWorkgroups(
                Math.ceil(this.simWidth  / WORKGROUP_SIZE_X),
                Math.ceil(this.simHeight / WORKGROUP_SIZE_Y),
            );
            this.velocity.swap();
            pass.setBindGroup(3, this.velocity.getBindGroup());
        }
    }

    simulate(encoder) {
        const pass = encoder.beginComputePass();
        this.setAllBindGroups(pass);
        this.advectVelocity(pass);
        this.confineVorticity(pass);
        this.project(pass, 800 / this.simScale);
        this.advectDensity(pass);
        this.splat(pass);
        pass.end();
    }

    render(encoder, context) {
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: [0, 0, 0, 1],
                    loadOp: 'clear',
                    storeOp: 'store'
                }
            ]
        });
        pass.setPipeline(this.renderPipeline);
        this.setAllBindGroups(pass);
        pass.draw(3);
        pass.end();
    }

    addSplat(center, radius, density, velocity) {
        this.splatQueue.push({
            splatFlag: 1,
            splatCenter: center,
            splatRadius: radius,
            splatDensity: density,
            splatVelocity: velocity
        });
    }

    randomSplat(radius) {
        const pos = [randrange(0.1, 0.9), randrange(0.1, 0.9)];
    
        const a = Math.random() * Math.PI * 2;
        const dir = [Math.cos(a), Math.sin(a)];
        const speed = 3; // UV/sec
    
        this.addSplat(
            pos,
            radius,
            randomNiceColor(),
            [dir[0] * speed, dir[1] * speed]
        );
    }

    randomSplats(radius, n = 20) {
        for (let i = 0; i < n; i++)
            this.randomSplat(radius);
    }

    step(context, params) {
        this.uniforms.setParams({
            ...params,
            ...(this.splatQueue.shift() ?? {
                splatFlag: 0,
                splatCenter: [0, 0],
                splatRadius: 0,
                splatDensity: [0, 0, 0, 0],
                splatVelocity: [0, 0]
            })
        });
        const encoder = this.device.createCommandEncoder();
        this.simulate(encoder);
        this.render(encoder, context);
        this.device.queue.submit([ encoder.finish() ]);
    }
}
