export class Uniforms {
    constructor(device) {
        this.device = device;
        this.paramsSize = 64;
        this.paramsData = new ArrayBuffer(this.paramsSize);
        this.paramsF32 = new Float32Array(this.paramsData);
        this.paramsU32 = new Uint32Array(this.paramsData);

        this.paramsBuffer = this.device.createBuffer({
            label: "params",
            size: this.paramsSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.linearClampedSampler = this.device.createSampler({
            label: "linear clamped sampler",
            magFilter: "linear",
            minFilter: "linear",
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge"
        });

        this.nearestClampedSampler = this.device.createSampler({
            label: "nearest clamped sampler",
            magFilter: "nearest",
            minFilter: "nearest",
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
        });

        this.layout = this.device.createBindGroupLayout({
            label: "uniforms",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                    sampler: { type: "filtering" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: "filtering" }
                }
            ]
        });

        this.bindGroup = this.device.createBindGroup({
            label: "uniforms",
            layout: this.layout,
            entries: [
                { binding: 0, resource: this.paramsBuffer          },
                { binding: 1, resource: this.linearClampedSampler  },
                { binding: 2, resource: this.nearestClampedSampler },
            ]
        });
    }

    getLayout() {
        return this.layout;
    }

    getBindGroup() {
        return this.bindGroup;
    }

    setParams({
        dt,
        splatFlag,
        splatRadius,
        vorticity,
        splatCenter,
        splatVelocity,
        splatDensity,
        densityDiffusion,
        velocityDiffusion,
        contour,
        sharpen
    }) {
        this.paramsF32[0] = dt;
        this.paramsU32[1] = splatFlag;
        this.paramsF32[2] = splatRadius;
        this.paramsF32[3] = vorticity;

        this.paramsF32[4] = splatCenter[0];
        this.paramsF32[5] = splatCenter[1];
        this.paramsF32[6] = splatVelocity[0];
        this.paramsF32[7] = splatVelocity[1];

        this.paramsF32[ 8] = splatDensity[0];
        this.paramsF32[ 9] = splatDensity[1];
        this.paramsF32[10] = splatDensity[2];
        this.paramsF32[11] = splatDensity[3];

        this.paramsF32[12] = densityDiffusion;
        this.paramsF32[13] = velocityDiffusion;

        this.paramsU32[14] = contour;
        this.paramsU32[15] = sharpen;

        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    }
}
