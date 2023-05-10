function getLayerCenter(layer, docWidth, docHeight) {
    var bounds = layer.bounds;
    var centerX = ((bounds[0].as("px") + bounds[2].as("px")) / 2) / docWidth;
    var centerY = ((bounds[1].as("px") + bounds[3].as("px")) / 2) / docHeight;
    var width = (bounds[2].as("px") - bounds[0].as("px")) / docWidth;
    var height = (bounds[3].as("px") - bounds[1].as("px")) / docHeight;

    return {
        x: centerX,
        y: centerY,
        width: width,
        height: height
    };
}

function processLayer(layer, layerInfo, parentFolderName, docWidth, docHeight) {
    if (!layer.isBackgroundLayer && layer.visible) {
        var layerCenter = getLayerCenter(layer, docWidth, docHeight);
        layerInfo.push({
            name: layer.name,
            folder: parentFolderName,
            centerX: layerCenter.x,
            centerY: layerCenter.y,
            width: layerCenter.width,
            height: layerCenter.height
        });
    }
}

function processLayers(layers, layerInfo, parentFolderName, docWidth, docHeight) {
    for (var i = 0; i < layers.length; i++) {
        var layer = layers[i];
        if (layer.typename === "LayerSet") {
            processLayers(layer.layers, layerInfo, layer.name, docWidth, docHeight);
        } else {
            processLayer(layer, layerInfo, parentFolderName, docWidth, docHeight);
        }
    }
}

function writeBoudingBoxesToFile(layerInfo, doc) {
    var filePath = doc.path.fsName.replace(/\\/g, '/');
    var fileName = doc.name.replace(/\.[^\.]+$/, '');
    var file = new File(filePath + '/' + fileName + '.txt');
    file.encoding = "UTF-8";
    file.open("w");

    for (var i = 0; i < layerInfo.length; i++) {
        var info = layerInfo[i];
        var folder = info.folder || "Root";
        file.writeln(folder.replace(/\?/g, "") + " " + info.centerX.toFixed(4) + " " + info.centerY.toFixed(4) + " " + info.width.toFixed(4) + " " + info.height.toFixed(4));
    }

    file.close();
    alert("Bounding boxes have been saved to " + fileName + ".txt in the same folder as the open file.");
}

function writeClassesToFile(layerInfo, doc) {
    var uniqueFolders = {};
    for (var i = 0; i < layerInfo.length; i++) {
        var folder = layerInfo[i].folder || "Root";
        uniqueFolders[folder.replace(/\?/g, "")] = true;
    }

    var filePath = doc.path.fsName.replace(/\\/g, '/');
    var fileName = doc.name.replace(/\.[^\.]+$/, '');
    var file = new File(filePath + '/' + fileName + '_classes.txt');
    file.encoding = "UTF-8";
    file.open("w");

    for (var folder in uniqueFolders) {
        file.writeln(folder);
    }

    file.close();
    alert("Class names have been saved to " + fileName + "_classes.txt in the same folder as the open file.");
}

function main() {
    if (app.documents.length === 0) {
        alert("No active document found. Please open a document.");
        return;
    }

    var doc = app.activeDocument;
    var layerInfo = [];
    var docWidth = doc.width.as("px");
    var docHeight = doc.height.as("px");
    processLayers(doc.layers, layerInfo, null, docWidth, docHeight);

    writeBoudingBoxesToFile(layerInfo, doc);
    writeClassesToFile(layerInfo, doc);
}

main();