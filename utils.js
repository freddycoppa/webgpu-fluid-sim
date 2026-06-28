function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);

    let r, g, b;

    switch (i % 6) {
        case 0: [r, g, b] = [v, t, p]; break;
        case 1: [r, g, b] = [q, v, p]; break;
        case 2: [r, g, b] = [p, v, t]; break;
        case 3: [r, g, b] = [p, q, v]; break;
        case 4: [r, g, b] = [t, p, v]; break;
        case 5: [r, g, b] = [v, p, q]; break;
    }

    return [r, g, b];
}

export function randomNiceColor() {
    const h = Math.random();              // random hue
    const s = 0.75 + 0.25 * Math.random(); // high saturation
    const v = /* 0.8 + 0.2 *  */Math.random();   // bright

    return [...hsvToRgb(h, s, v), 1.0];
}

export function closeEnoughToZero(x) {
    return Math.abs(x) < 1e-6;
}

export function randrange(min, max) {
    return min + Math.random() * (max - min);
}
