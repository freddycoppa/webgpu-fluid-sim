export class PingPongTexture {
    constructor(
        label,
        device,
        width,
        height,
        additionalVisibility = 0,
        format = "r32float",
        sampleType = "unfilterable-float"
    ) {
        this.label  = label;
        this.device = device;
        this.width  = width ;
        this.height = height;
        this.format = format;

        this.layout = device.createBindGroupLayout({
            label: label + " bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | additionalVisibility
                        ,
                    texture: { sampleType }
                },
                {
                    binding: 1,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | additionalVisibility
                        ,
                    storageTexture: { format }
                }
            ]
        });

        this.createTextures(label);
        this.createBindGroups(label);
    }

    createTextures(label) {
        this.primaryTexture = this.device.createTexture({
            label: label + " primary texture",
            size: [this.width, this.height],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });

        this.secondaryTexture = this.device.createTexture({
            label: label + " secondary texture",
            size: [this.width, this.height],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });
    }

    createBindGroups(label) {
        this.primaryBindGroup = this.device.createBindGroup({
            label: label + " primary bind group",
            layout: this.layout,
            entries: [
                { binding: 0, resource: this.primaryTexture   },
                { binding: 1, resource: this.secondaryTexture },
            ]
        });

        this.secondaryBindGroup = this.device.createBindGroup({
            label: label + " secondary bind group",
            layout: this.layout,
            entries: [
                { binding: 0, resource: this.secondaryTexture },
                { binding: 1, resource: this.primaryTexture   },
            ]
        });
    }

    resize(width, height) {
        this.primaryTexture.destroy();
        this.secondaryTexture.destroy();

        this.width = width;
        this.height = height;

        this.createTextures(this.label + " resize");
        this.createBindGroups(this.label + " resize");
    }

    upload(values, bytesPerRow) {
        this.device.queue.writeTexture(
            { texture: this.primaryTexture },
            values,
            { bytesPerRow },
            { width: this.width, height: this.height }
        );
    }

    getLayout() {
        return this.layout;
    }

    swap() {
        [
            this.primaryBindGroup,
            this.secondaryBindGroup
        ] = [
            this.secondaryBindGroup,
            this.primaryBindGroup
        ];
        [
            this.primaryTexture,
            this.secondaryTexture
        ] = [
            this.secondaryTexture,
            this.primaryTexture
        ];
    }

    getBindGroup() {
        return this.primaryBindGroup;
    }
}

export class PingPongTexturePair {
    constructor(
        label,
        device,
        widthA,
        heightA,
        widthB,
        heightB,
        format = "r32float"
    ) {
        this.label = label;
        this.device = device;
        this.widthA = widthA;
        this.heightA = heightA;
        this.widthB = widthB;
        this.heightB = heightB;
        this.format = format;

        this.createLayout(label);
        this.createTextures(label);
        this.createBindGroups(label);
    }

    createLayout(label) {
        this.layout = this.device.createBindGroupLayout({
            label,
            entries: [
                {
                    binding: 0,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | GPUShaderStage.FRAGMENT
                        ,
                    texture: { sampleType: "unfilterable-float" }
                },
                {
                    binding: 1,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | GPUShaderStage.FRAGMENT
                        ,
                    storageTexture: { format: this.format }
                },
                {
                    binding: 2,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | GPUShaderStage.FRAGMENT
                        ,
                    texture: { sampleType: "unfilterable-float" }
                },
                {
                    binding: 3,
                    visibility
                        : GPUShaderStage.COMPUTE
                        | GPUShaderStage.FRAGMENT
                        ,
                    storageTexture: { format: this.format }
                }
            ]
        });
    }

    createTextures(label) {
        this.primaryTextureA = this.device.createTexture({
            label: label + " primary texture A",
            size: [ this.widthA, this.heightA ],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });

        this.secondaryTextureA = this.device.createTexture({
            label: label + " secondary texture A",
            size: [ this.widthA, this.heightA ],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });

        this.primaryTextureB = this.device.createTexture({
            label: label + " primary texture B",
            size: [ this.widthB, this.heightB ],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });

        this.secondaryTextureB = this.device.createTexture({
            label: label + " secondary texture B",
            size: [ this.widthB, this.heightB ],
            format: this.format,
            usage
                : GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.STORAGE_BINDING
                | GPUTextureUsage.COPY_DST
        });
    }

    createBindGroups(label) {
        this.primaryBindGroup = this.device.createBindGroup({
            label: label + " primary bind group",
            layout: this.layout,
            entries: [
                { binding: 0, resource: this.primaryTextureA   },
                { binding: 1, resource: this.secondaryTextureA },
                { binding: 2, resource: this.primaryTextureB   },
                { binding: 3, resource: this.secondaryTextureB },
            ]
        });

        this.secondaryBindGroup = this.device.createBindGroup({
            label: label + " secondary bind group",
            layout: this.layout,
            entries: [
                { binding: 0, resource: this.secondaryTextureA },
                { binding: 1, resource: this.primaryTextureA   },
                { binding: 2, resource: this.secondaryTextureB },
                { binding: 3, resource: this.primaryTextureB   },
            ]
        });
    }

    resize(widthA, heightA, widthB, heightB) {
        this.primaryTextureA.destroy();
        this.secondaryTextureA.destroy();
        this.primaryTextureB.destroy();
        this.secondaryTextureB.destroy();

        this.widthA = widthA;
        this.heightA = heightA;
        this.widthB = widthB;
        this.heightB = heightB;

        this.createTextures(this.label + " resize");
        this.createBindGroups(this.label + " resize");
    }

    getLayout() {
        return this.layout;
    }

    getBindGroup() {
        return this.primaryBindGroup;
    }

    uploadA(values, bytesPerRow) {
        this.device.queue.writeTexture(
            { texture: this.primaryTextureA },
            values,
            { bytesPerRow },
            { width: this.widthA, height: this.heightA }
        );
    }

    uploadB(values, bytesPerRow) {
        this.device.queue.writeTexture(
            { texture: this.primaryTextureB },
            values,
            { bytesPerRow },
            { width: this.widthB, height: this.heightB }
        );
    }

    swap() {
        [
            this.primaryBindGroup,
            this.secondaryBindGroup
        ] = [
            this.secondaryBindGroup,
            this.primaryBindGroup
        ];

        [
            this.primaryTextureA,
            this.secondaryTextureA
        ] = [
            this.secondaryTextureA,
            this.primaryTextureA
        ];

        [
            this.primaryTextureB,
            this.secondaryTextureB
        ] = [
            this.secondaryTextureB,
            this.primaryTextureB
        ]
    }
}
