export const mouse = new class {
    constructor() {
        this.x = 0;
        this.y = 0;
        this.clicked = false;

        const canvas = document.getElementById("fluid-canvas");

        canvas.addEventListener("pointermove", event => {
            const rect = canvas.getBoundingClientRect();
            this.x = (event.clientX - rect.left) / rect.width;
            this.y = (event.clientY - rect.top ) / rect.height;
        });

        canvas.addEventListener("pointerdown", event => {
            canvas.setPointerCapture(event.pointerId);
            this.clicked = true;
            const rect = canvas.getBoundingClientRect();
            this.x = (event.clientX - rect.left) / rect.width;
            this.y = (event.clientY - rect.top ) / rect.height;
        });

        canvas.addEventListener("pointerup", event => {
            canvas.releasePointerCapture(event.pointerId);
            this.clicked = false;
            const rect = canvas.getBoundingClientRect();
            this.x = (event.clientX - rect.left) / rect.width;
            this.y = (event.clientY - rect.top ) / rect.height;
        });

        canvas.addEventListener("pointercancel", () => {
            this.clicked = false;
        });

        canvas.addEventListener("lostpointercapture", () => {
            this.clicked = false;
        });
    }
}
