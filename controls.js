export function setupControls() {
    const state = {
        dispScale: 0.0,
        simScale: 0.0,
        splatRadius: 0.0,
        vorticity: 0.0,
        densityDiffusion: 0.0,
        velocityDiffusion: 0.0,
        contour: false,
        sharpen: false,
    };

    const panel = document.querySelector("#controls-panel");
    const showButton = document.querySelector("#controls-show-button");
    const hideButton = document.querySelector("#controls-hide-button");

    showButton.classList.add("hidden");

    hideButton.addEventListener("click", () => {
        panel.classList.add("hidden");
        showButton.classList.remove("hidden");
    });

    showButton.addEventListener("click", () => {
        panel.classList.remove("hidden");
        showButton.classList.add("hidden");
    });

    // Prevent UI pointer events from leaking through to the canvas.
    panel.addEventListener("pointerdown", e => e.stopPropagation());
    panel.addEventListener("pointermove", e => e.stopPropagation());
    panel.addEventListener("pointerup", e => e.stopPropagation());
    showButton.addEventListener("pointerdown", e => e.stopPropagation());

    bindSlider(
        "#splat-radius-slider",
        "#splat-radius-value",
        value => value.toFixed(2),
        value => state.splatRadius = value,
    );

    bindSlider(
        "#vorticity-slider",
        "#vorticity-value",
        value => value.toFixed(1),
        value => state.vorticity = value,
    );

    bindSlider(
        "#density-diffusion-slider",
        "#density-diffusion-value",
        value => value.toFixed(2),
        value => state.densityDiffusion = value,
    );

    bindSlider(
        "#velocity-diffusion-slider",
        "#velocity-diffusion-value",
        value => value.toFixed(2),
        value => state.velocityDiffusion = value,
    );

    bindDropdown(
        "#display-scale-select",
        value => state.dispScale = value
    );

    bindDropdown(
        "#sim-scale-select",
        value => state.simScale = value
    );

    bindCheckbox(
        "#contour-checkbox",
        value => state.contour = value
    );

    bindCheckbox(
        "#sharpen-checkbox",
        value => state.sharpen = value
    );

    const dispScaleSelect = document.querySelector("#display-scale-select");
    const simScaleSelect = document.querySelector("#sim-scale-select");

    state.onDispScaleChange = function (callback) {
        dispScaleSelect.addEventListener(
            "change", () => callback(Number(dispScaleSelect.value))
        );
    }

    state.onSimScaleChange = function (callback) {
        simScaleSelect.addEventListener(
            "change", () => callback(Number(simScaleSelect.value))
        );
    }

    return state;
}

function bindSlider(
    sliderSelector,
    valueSelector,
    format,
    callback,
) {
    const slider = document.querySelector(sliderSelector);
    const valueLabel = document.querySelector(valueSelector);

    function updateLabel() {
        valueLabel.textContent = format(Number(slider.value));
    }

    slider.addEventListener("input", updateLabel);
    slider.addEventListener("input", () => callback(Number(slider.value)));
    
    updateLabel();
    callback(Number(slider.value))
}

function bindDropdown(
    dropdownSelector,
    callback
) {
    const dropdown = document.querySelector(dropdownSelector);
    dropdown.addEventListener("change", () => callback(Number(dropdown.value)));
    callback(Number(dropdown.value));
}

function bindCheckbox(
    checkboxSelector,
    callback
) {
    const checkbox = document.querySelector(checkboxSelector);
    checkbox.addEventListener("change", () => callback(checkbox.checked));
    callback(checkbox.checked);
}
