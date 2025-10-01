// ==============================================================================
// ALGORITMO DE DETECCIÓN - JavaScript con OpenCV.js
// ==============================================================================

// Configuración del algoritmo
const config = {
    threshold: 0.35,
    amarilloBajo: [15, 50, 50],
    amarilloAlto: [40, 255, 255],
    azulBajo: [95, 80, 80],
    azulAlto: [130, 255, 255]
};

/**
 * Filtro de color adaptativo
 * @param {cv.Mat} hsvRoi - Imagen en espacio HSV
 * @param {Array} initialLower - Límite inferior inicial [H, S, V]
 * @param {Array} initialUpper - Límite superior inicial [H, S, V]
 * @param {Number} hTolerance - Tolerancia de matiz
 * @param {Number} svTolerance - Tolerancia de saturación/valor
 * @returns {cv.Mat} Máscara binaria
 */
function adaptiveColorFilter(hsvRoi, initialLower, initialUpper, hTolerance = 7, svTolerance = 100) {
    const lower = new cv.Mat(1, 1, cv.CV_8UC3, initialLower);
    const upper = new cv.Mat(1, 1, cv.CV_8UC3, initialUpper);
    const initialMask = new cv.Mat();
    
    cv.inRange(hsvRoi, lower, upper, initialMask);
    
    if (cv.countNonZero(initialMask) === 0) {
        lower.delete();
        upper.delete();
        return initialMask;
    }
    
    const meanHsv = cv.mean(hsvRoi, initialMask);
    const dominantH = Math.round(meanHsv[0]);
    const dominantS = Math.round(meanHsv[1]);
    const dominantV = Math.round(meanHsv[2]);
    
    const newLowerH = Math.max(0, dominantH - hTolerance);
    const newUpperH = Math.min(179, dominantH + hTolerance);
    const newLowerS = Math.max(0, dominantS - svTolerance);
    const newUpperS = Math.min(255, dominantS + svTolerance);
    const newLowerV = Math.max(0, dominantV - svTolerance);
    const newUpperV = Math.min(255, dominantV + svTolerance);
    
    const adaptiveLower = new cv.Mat(1, 1, cv.CV_8UC3, [newLowerH, newLowerS, newLowerV]);
    const adaptiveUpper = new cv.Mat(1, 1, cv.CV_8UC3, [newUpperH, newUpperS, newUpperV]);
    const finalMask = new cv.Mat();
    
    cv.inRange(hsvRoi, adaptiveLower, adaptiveUpper, finalMask);
    
    lower.delete();
    upper.delete();
    initialMask.delete();
    adaptiveLower.delete();
    adaptiveUpper.delete();
    
    return finalMask;
}

/**
 * Encuentra centros de círculos (sólidos y con agujeros)
 * @param {cv.Mat} mask - Máscara binaria
 * @param {Number} minSolidArea - Área mínima para círculos sólidos
 * @param {Number} minHoleArea - Área mínima para agujeros
 * @returns {Array} Array de centros [{x, y}, ...]
 */
function findCenters(mask, minSolidArea = 20, minHoleArea = 100) {
    const centers = [];
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    
    cv.findContours(mask, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    
    if (hierarchy.rows === 0) {
        contours.delete();
        hierarchy.delete();
        return centers;
    }
    
    for (let i = 0; i < contours.size(); i++) {
        const parent = hierarchy.intAt(i, 3);
        const firstChild = hierarchy.intAt(i, 2);
        
        // Contorno sin padre (nivel superior)
        if (parent === -1) {
            const contour = contours.get(i);
            
            // Círculo sólido (sin hijos)
            if (firstChild === -1) {
                const area = cv.contourArea(contour);
                if (area > minSolidArea) {
                    const M = cv.moments(contour);
                    if (M.m00 !== 0) {
                        centers.push({
                            x: Math.round(M.m10 / M.m00),
                            y: Math.round(M.m01 / M.m00)
                        });
                    }
                }
            } else {
                // Círculo con agujeros
                let childIndex = firstChild;
                while (childIndex !== -1) {
                    const holeContour = contours.get(childIndex);
                    const area = cv.contourArea(holeContour);
                    if (area > minHoleArea) {
                        const M = cv.moments(holeContour);
                        if (M.m00 !== 0) {
                            centers.push({
                                x: Math.round(M.m10 / M.m00),
                                y: Math.round(M.m01 / M.m00)
                            });
                        }
                    }
                    childIndex = hierarchy.intAt(childIndex, 0);
                }
            }
        }
    }
    
    contours.delete();
    hierarchy.delete();
    return centers;
}

/**
 * Ordena 4 puntos en orden: TL, TR, BR, BL
 * @param {Array} pts - Array de 4 puntos [{x, y}, ...]
 * @returns {Array} Puntos ordenados
 */
function orderPoints(pts) {
    const rect = new Array(4);
    
    // Suma de coordenadas
    const sums = pts.map(p => p.x + p.y);
    const minSumIdx = sums.indexOf(Math.min(...sums));
    const maxSumIdx = sums.indexOf(Math.max(...sums));
    
    rect[0] = pts[minSumIdx]; // Top-left
    rect[2] = pts[maxSumIdx]; // Bottom-right
    
    // Diferencia de coordenadas
    const diffs = pts.map(p => p.y - p.x);
    const minDiffIdx = diffs.indexOf(Math.min(...diffs));
    const maxDiffIdx = diffs.indexOf(Math.max(...diffs));
    
    rect[1] = pts[minDiffIdx]; // Top-right
    rect[3] = pts[maxDiffIdx]; // Bottom-left
    
    return rect;
}

/**
 * K-Means clustering simple (para agrupar puntos en 4 clusters)
 * @param {Array} points - Array de puntos [{x, y}, ...]
 * @param {Number} k - Número de clusters
 * @returns {Array} Centroides de clusters
 */
function kMeansClustering(points, k = 4, maxIterations = 100) {
    if (points.length < k) return points;
    
    // Inicializar centroides aleatoriamente
    let centroids = [];
    const shuffled = [...points].sort(() => Math.random() - 0.5);
    for (let i = 0; i < k; i++) {
        centroids.push({...shuffled[i]});
    }
    
    for (let iter = 0; iter < maxIterations; iter++) {
        // Asignar puntos a clusters
        const clusters = Array.from({length: k}, () => []);
        
        for (const point of points) {
            let minDist = Infinity;
            let clusterIdx = 0;
            
            for (let i = 0; i < k; i++) {
                const dist = Math.hypot(point.x - centroids[i].x, point.y - centroids[i].y);
                if (dist < minDist) {
                    minDist = dist;
                    clusterIdx = i;
                }
            }
            
            clusters[clusterIdx].push(point);
        }
        
        // Actualizar centroides
        const newCentroids = [];
        for (const cluster of clusters) {
            if (cluster.length > 0) {
                const sumX = cluster.reduce((sum, p) => sum + p.x, 0);
                const sumY = cluster.reduce((sum, p) => sum + p.y, 0);
                newCentroids.push({
                    x: Math.round(sumX / cluster.length),
                    y: Math.round(sumY / cluster.length)
                });
            }
        }
        
        // Verificar convergencia
        let converged = true;
        for (let i = 0; i < k; i++) {
            if (newCentroids[i].x !== centroids[i].x || newCentroids[i].y !== centroids[i].y) {
                converged = false;
                break;
            }
        }
        
        centroids = newCentroids;
        if (converged) break;
    }
    
    return centroids;
}

/**
 * Procesa un frame para detectar la superficie
 * @param {cv.Mat} frame - Frame a procesar
 * @param {cv.Mat} template - Plantilla a buscar
 * @param {Function} progressCallback - Callback para reportar progreso
 * @returns {Promise<Array|null>} Array de 4 puntos o null si no se detecta
 */
async function processFrame(frame, template, progressCallback) {
    let bestVal = -1;
    let bestLoc = null;
    let bestDims = null;
    
    // Template matching multi-escala
    const scales = [];
    for (let i = 0; i < 20; i++) {
        scales.push(1.5 - (i * 0.05));
    }
    
    for (const scale of scales) {
        const width = Math.round(template.cols * scale);
        const height = Math.round(template.rows * scale);
        
        if (width > frame.cols || height > frame.rows) continue;
        
        const resized = new cv.Mat();
        const dsize = new cv.Size(width, height);
        cv.resize(template, resized, dsize, 0, 0, cv.INTER_LINEAR);
        
        const result = new cv.Mat();
        cv.matchTemplate(frame, resized, result, cv.TM_CCOEFF_NORMED);
        
        const minMax = cv.minMaxLoc(result);
        
        if (minMax.maxVal > bestVal) {
            bestVal = minMax.maxVal;
            bestLoc = minMax.maxLoc;
            bestDims = {width, height};
        }
        
        resized.delete();
        result.delete();
    }
    
    if (progressCallback) progressCallback(25);
    
    if (bestVal < config.threshold) {
        if (progressCallback) progressCallback(100);
        return null;
    }
    
    // Extraer ROI
    const roi = frame.roi(new cv.Rect(
        bestLoc.x,
        bestLoc.y,
        bestDims.width,
        bestDims.height
    ));
    
    if (progressCallback) progressCallback(50);
    
    // Convertir a HSV
    const hsvRoi = new cv.Mat();
    cv.cvtColor(roi, hsvRoi, cv.COLOR_BGR2HSV);
    
    // Detectar colores
    const maskAzul = adaptiveColorFilter(hsvRoi, config.azulBajo, config.azulAlto);
    const maskAmarillo = adaptiveColorFilter(hsvRoi, config.amarilloBajo, config.amarilloAlto);
    
    const blueCenters = findCenters(maskAzul, 60);
    const yellowCenters = findCenters(maskAmarillo, 20);
    
    if (progressCallback) progressCallback(75);
    
    const nBlue = blueCenters.length;
    const nYellow = yellowCenters.length;
    
    let absoluteMidpoints = null;
    
    // Verificar que tengamos la configuración correcta
    if ((nBlue === 4 && nYellow >= 1 && nYellow <= 4) || 
        (nYellow === 4 && nBlue >= 1 && nBlue <= 4)) {
        
        const allPoints = [...blueCenters, ...yellowCenters];
        const midpoints = kMeansClustering(allPoints, 4);
        
        // Convertir a coordenadas absolutas
        absoluteMidpoints = midpoints.map(p => ({
            x: p.x + bestLoc.x,
            y: p.y + bestLoc.y
        }));
    }
    
    // Limpiar
    roi.delete();
    hsvRoi.delete();
    maskAzul.delete();
    maskAmarillo.delete();
    
    if (progressCallback) progressCallback(100);
    
    return absoluteMidpoints;
}

/**
 * Dibuja la cuadrícula extendida sobre el frame
 * @param {cv.Mat} image - Imagen donde dibujar
 * @param {Array} midpoints - 4 puntos de la superficie
 * @param {Array} color - Color BGR [B, G, R]
 */
function drawExtendedGrid(image, midpoints, color = [255, 255, 0]) {
    const h = image.rows;
    const w = image.cols;
    
    const orderedPts = orderPoints(midpoints);
    
    // Puntos de perspectiva (4 esquinas detectadas)
    const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        orderedPts[0].x, orderedPts[0].y,
        orderedPts[1].x, orderedPts[1].y,
        orderedPts[2].x, orderedPts[2].y,
        orderedPts[3].x, orderedPts[3].y
    ]);
    
    // Puntos del plano plano (0,0 a 1,1)
    const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        1, 0,
        1, 1,
        0, 1
    ]);
    
    // Matriz de transformación inversa
    const matrixInv = cv.getPerspectiveTransform(dstPoints, srcPoints);
    
    const rangeExt = 5;
    const nLines = 21;
    const coords = [];
    for (let i = 0; i < nLines; i++) {
        coords.push(-rangeExt + (i * (2 * rangeExt) / (nLines - 1)));
    }
    
    // Dibujar líneas horizontales
    for (const y of coords) {
        const p1Flat = cv.matFromArray(1, 1, cv.CV_32FC2, [-rangeExt, y]);
        const p2Flat = cv.matFromArray(1, 1, cv.CV_32FC2, [rangeExt, y]);
        
        const p1Persp = new cv.Mat();
        const p2Persp = new cv.Mat();
        
        cv.perspectiveTransform(p1Flat, p1Persp, matrixInv);
        cv.perspectiveTransform(p2Flat, p2Persp, matrixInv);
        
        const pt1 = new cv.Point(
            Math.round(p1Persp.floatAt(0, 0)),
            Math.round(p1Persp.floatAt(0, 1))
        );
        const pt2 = new cv.Point(
            Math.round(p2Persp.floatAt(0, 0)),
            Math.round(p2Persp.floatAt(0, 1))
        );
        
        // Clip line a los límites de la imagen
        if (pt1.x >= 0 && pt1.x < w && pt1.y >= 0 && pt1.y < h &&
            pt2.x >= 0 && pt2.x < w && pt2.y >= 0 && pt2.y < h) {
            cv.line(image, pt1, pt2, color, 1);
        }
        
        p1Flat.delete();
        p2Flat.delete();
        p1Persp.delete();
        p2Persp.delete();
    }
    
    // Dibujar líneas verticales
    for (const x of coords) {
        const p1Flat = cv.matFromArray(1, 1, cv.CV_32FC2, [x, -rangeExt]);
        const p2Flat = cv.matFromArray(1, 1, cv.CV_32FC2, [x, rangeExt]);
        
        const p1Persp = new cv.Mat();
        const p2Persp = new cv.Mat();
        
        cv.perspectiveTransform(p1Flat, p1Persp, matrixInv);
        cv.perspectiveTransform(p2Flat, p2Persp, matrixInv);
        
        const pt1 = new cv.Point(
            Math.round(p1Persp.floatAt(0, 0)),
            Math.round(p1Persp.floatAt(0, 1))
        );
        const pt2 = new cv.Point(
            Math.round(p2Persp.floatAt(0, 0)),
            Math.round(p2Persp.floatAt(0, 1))
        );
        
        if (pt1.x >= 0 && pt1.x < w && pt1.y >= 0 && pt1.y < h &&
            pt2.x >= 0 && pt2.x < w && pt2.y >= 0 && pt2.y < h) {
            cv.line(image, pt1, pt2, color, 1);
        }
        
        p1Flat.delete();
        p2Flat.delete();
        p1Persp.delete();
        p2Persp.delete();
    }
    
    // Dibujar marcadores en los puntos medios
    for (const point of orderedPts) {
        cv.drawMarker(
            image,
            new cv.Point(point.x, point.y),
            [255, 0, 255],
            cv.MARKER_STAR,
            20,
            2
        );
    }
    
    srcPoints.delete();
    dstPoints.delete();
    matrixInv.delete();
}

/**
 * Calcula la distancia real entre dos puntos
 * @param {Object} p1 - Punto 1 {x, y}
 * @param {Object} p2 - Punto 2 {x, y}
 * @param {Array} surfacePoints - 4 puntos de calibración
 * @returns {Number} Distancia en cm
 */
function calculateRealDistance(p1, p2, surfacePoints) {
    const orderedPts = orderPoints(surfacePoints);
    
    const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        orderedPts[0].x, orderedPts[0].y,
        orderedPts[1].x, orderedPts[1].y,
        orderedPts[2].x, orderedPts[2].y,
        orderedPts[3].x, orderedPts[3].y
    ]);
    
    const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        1, 0,
        1, 1,
        0, 1
    ]);
    
    const matrix = cv.getPerspectiveTransform(srcPoints, dstPoints);
    
    const p1Mat = cv.matFromArray(1, 1, cv.CV_32FC2, [p1.x, p1.y]);
    const p2Mat = cv.matFromArray(1, 1, cv.CV_32FC2, [p2.x, p2.y]);
    
    const p1Flat = new cv.Mat();
    const p2Flat = new cv.Mat();
    
    cv.perspectiveTransform(p1Mat, p1Flat, matrix);
    cv.perspectiveTransform(p2Mat, p2Flat, matrix);
    
    const dx = p2Flat.floatAt(0, 0) - p1Flat.floatAt(0, 0);
    const dy = p2Flat.floatAt(0, 1) - p1Flat.floatAt(0, 1);
    const distFlat = Math.sqrt(dx * dx + dy * dy);
    
    const distCm = distFlat * 40.0; // 1 unidad = 40 cm
    
    srcPoints.delete();
    dstPoints.delete();
    matrix.delete();
    p1Mat.delete();
    p2Mat.delete();
    p1Flat.delete();
    p2Flat.delete();
    
    return distCm;
}

// Exportar funciones
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        config,
        adaptiveColorFilter,
        findCenters,
        orderPoints,
        kMeansClustering,
        processFrame,
        drawExtendedGrid,
        calculateRealDistance
    };
}