// SeedVR2 output size preview in the img2img tab.
// The size of the input image is read from the page, so the image itself is not sent to the server for the preview.
// The selector is the one Forge Neo's own "Resize by" preview uses (currentImg2imgSourceResolution in javascript/ui.js),
// without its rounding to the width/height slider step: the upscale works on the exact size.

function seedvr2Img2imgSourceSize() {
    const img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] :is(img, canvas)');
    if (!img) return [0, 0];
    return [img.naturalWidth || img.width || 0, img.naturalHeight || img.height || 0];
}

// Event `_js` of the preview: replaces the first two inputs (hidden numbers) with the input image size.
function seedvr2Img2imgPreview(...args) {
    return [...seedvr2Img2imgSourceSize(), ...args.slice(2)];
}

// Forge's DOM observer does not see an image replacing another (only childList changes are observed), so the size is
// checked once a second and the hidden refresh button is pressed when it changes (new image, cleared image, other tab).
onUiLoaded(function () {
    let lastSize = null;
    setInterval(function () {
        const button = gradioApp().getElementById("seedvr2_img2img_size_refresh");
        if (!button) return;
        const size = seedvr2Img2imgSourceSize().join("x");
        if (size !== lastSize) {
            lastSize = size;
            button.click();
        }
    }, 1000);
});
