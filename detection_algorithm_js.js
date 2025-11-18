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

/**
 * Codifica una imagen a JPEG y la convierte a base64
 * @param {HTMLVideoElement|HTMLCanvasElement} source - Fuente de la imagen (video o canvas)
 * @returns {Promise<string>} Imagen codificada en base64 (sin prefijo)
 */
async function encodeImageToBase64(source) {
    // Crear canvas temporal
    const tempCanvas = document.createElement('canvas');
    
    if (source instanceof HTMLVideoElement) {
        tempCanvas.width = source.videoWidth;
        tempCanvas.height = source.videoHeight;
    } else if (source instanceof HTMLCanvasElement) {
        tempCanvas.width = source.width;
        tempCanvas.height = source.height;
    } else {
        throw new Error('Source debe ser un HTMLVideoElement o HTMLCanvasElement');
    }

    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(source, 0, 0, tempCanvas.width, tempCanvas.height);

    // Convertir a base64 usando el método nativo del canvas (JPEG)
    return new Promise((resolve, reject) => {
        tempCanvas.toBlob((blob) => {
            if (!blob) {
                reject(new Error('Error al codificar la imagen'));
                return;
            }
            
            const reader = new FileReader();
            reader.onloadend = () => {
                // Remover el prefijo data:image/jpeg;base64,
                const base64 = reader.result.replace(/^data:image\/jpeg;base64,/, '');
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.9); // Calidad JPEG 90%
    });
}

/**
 * Calcula el bounding box de un array de puntos
 * @param {Array<{x: number, y: number}>} points - Array de puntos
 * @returns {Array<number>} Bounding box [x1, y1, x2, y2]
 */
function calculateBoundingBox(points) {
    if (!points || points.length === 0) {
        throw new Error('No hay puntos para calcular el bounding box');
    }

    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const x1 = Math.min(...xs);
    const y1 = Math.min(...ys);
    const x2 = Math.max(...xs);
    const y2 = Math.max(...ys);

    return [x1, y1, x2, y2];
}

/**
 * Clase para proyectar mallas 3D sobre la superficie detectada
 */
class SurfaceProjector {
    /**
     * @param {Object} imageShape - {width: number, height: number}
     * @param {Array<{x: number, y: number}>} detected2DPoints - 4 puntos detectados en 2D
     * @param {number} realWorldSideCm - Lado del cuadrado en cm en el mundo real
     */
    constructor(imageShape, detected2DPoints, realWorldSideCm = 40.0) {
        console.log('Creando instancia de SurfaceProjector...');

        const h = imageShape.height;
        const w = imageShape.width;

        const halfSide = realWorldSideCm / 2.0;

        // Puntos 3D de las esquinas del mundo real (centrado en origen, z=0)
        this.worldPoints3DCorners = cv.matFromArray(4, 1, cv.CV_32FC3, [
            -halfSide, -halfSide, 0.0,
            halfSide, -halfSide, 0.0,
            halfSide, halfSide, 0.0,
            -halfSide, halfSide, 0.0
        ]);

        // Ordenar puntos 2D (TL, TR, BR, BL)
        const ordered2D = orderPoints(detected2DPoints);
        this.imagePoints2D = cv.matFromArray(4, 1, cv.CV_32FC2, [
            ordered2D[0].x, ordered2D[0].y,
            ordered2D[1].x, ordered2D[1].y,
            ordered2D[2].x, ordered2D[2].y,
            ordered2D[3].x, ordered2D[3].y
        ]);

        // Matriz de cámara
        const focalLength = w;
        const centerX = w / 2;
        const centerY = h / 2;
        this.cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [
            focalLength, 0, centerX,
            0, focalLength, centerY,
            0, 0, 1
        ]);

        // Coeficientes de distorsión (cero)
        this.distCoeffs = new cv.Mat(4, 1, cv.CV_64F);
        this.distCoeffs.setTo(new cv.Scalar(0, 0, 0, 0));

        // Vectores de rotación y traslación
        this.rvec = new cv.Mat(3, 1, cv.CV_64F);
        this.tvec = new cv.Mat(3, 1, cv.CV_64F);

        console.log('solvePnP con puntos 3D y puntos 2D');
        const success = cv.solvePnP(
            this.worldPoints3DCorners,
            this.imagePoints2D,
            this.cameraMatrix,
            this.distCoeffs,
            this.rvec,
            this.tvec
        );

        if (!success) {
            console.error('¡solvePnP falló!');
        } else {
            console.log('solvePnP exitoso.');
        }
    }

    /**
     * Proyecta puntos 3D a 2D manualmente (cv.projectPoints no está disponible)
     * @param {cv.Mat} points3D - Puntos 3D (N x 1, CV_32FC3)
     * @returns {Array<{x: number, y: number}>} Puntos 2D proyectados
     */
    projectPoints3DTo2D(points3D) {
        // Convertir rvec a matriz de rotación usando Rodrigues
        const rmat = new cv.Mat(3, 3, cv.CV_64F);
        cv.Rodrigues(this.rvec, rmat);
        
        // Obtener parámetros de la matriz de cámara
        const fx = this.cameraMatrix.doubleAt(0, 0);
        const fy = this.cameraMatrix.doubleAt(1, 1);
        const cx = this.cameraMatrix.doubleAt(0, 2);
        const cy = this.cameraMatrix.doubleAt(1, 2);
        
        // Obtener traslación
        const tx = this.tvec.doubleAt(0, 0);
        const ty = this.tvec.doubleAt(1, 0);
        const tz = this.tvec.doubleAt(2, 0);
        
        const points2D = [];
        const numPoints = points3D.rows;
        
        for (let i = 0; i < numPoints; i++) {
            // Obtener punto 3D
            const x3d = points3D.floatAt(i, 0);
            const y3d = points3D.floatAt(i, 1);
            const z3d = points3D.floatAt(i, 2);
            
            // Aplicar rotación: R * [x, y, z]^T
            const r11 = rmat.doubleAt(0, 0);
            const r12 = rmat.doubleAt(0, 1);
            const r13 = rmat.doubleAt(0, 2);
            const r21 = rmat.doubleAt(1, 0);
            const r22 = rmat.doubleAt(1, 1);
            const r23 = rmat.doubleAt(1, 2);
            const r31 = rmat.doubleAt(2, 0);
            const r32 = rmat.doubleAt(2, 1);
            const r33 = rmat.doubleAt(2, 2);
            
            const x_rot = r11 * x3d + r12 * y3d + r13 * z3d;
            const y_rot = r21 * x3d + r22 * y3d + r23 * z3d;
            const z_rot = r31 * x3d + r32 * y3d + r33 * z3d;
            
            // Aplicar traslación
            const x_trans = x_rot + tx;
            const y_trans = y_rot + ty;
            const z_trans = z_rot + tz;
            
            // Proyectar usando matriz de cámara
            if (z_trans > 0) {
                const x2d = (fx * x_trans / z_trans) + cx;
                const y2d = (fy * y_trans / z_trans) + cy;
                points2D.push({ x: x2d, y: y2d });
            } else {
                // Si está detrás de la cámara, usar un punto fuera de la imagen
                points2D.push({ x: -1, y: -1 });
            }
        }
        
        rmat.delete();
        return points2D;
    }

    /**
     * Dibuja la malla proyectada sobre la imagen
     * @param {cv.Mat} image - Imagen donde dibujar (BGR)
     * @param {number} heightCm - Altura en cm para proyectar la malla
     */
    drawProjectedGrids(image, heightCm = 0.0) {
        const h = image.rows;
        const w = image.cols;

        const rangeExt = 80.0;
        const nLines = 41;
        const coords = [];
        for (let i = 0; i < nLines; i++) {
            coords.push(-rangeExt + (i * (2 * rangeExt) / (nLines - 1)));
        }

        // Crear líneas de la malla en 3D
        const projGridLines3D = [];
        for (const coord of coords) {
            // Líneas horizontales
            projGridLines3D.push([-rangeExt, coord, heightCm]);
            projGridLines3D.push([rangeExt, coord, heightCm]);
            // Líneas verticales
            projGridLines3D.push([coord, -rangeExt, heightCm]);
            projGridLines3D.push([coord, rangeExt, heightCm]);
        }

        // Convertir a Mat
        const allProjPoints3D = cv.matFromArray(projGridLines3D.length, 1, cv.CV_32FC3, 
            projGridLines3D.flat()
        );

        // Proyectar puntos 3D a 2D manualmente
        const projPoints2D = this.projectPoints3DTo2D(allProjPoints3D);

        // Dibujar líneas de la malla (verde)
        const green = new cv.Scalar(0, 255, 0, 0);
        for (let i = 0; i < projGridLines3D.length; i += 2) {
            const pt1_2d = projPoints2D[i];
            const pt2_2d = projPoints2D[i + 1];
            
            // Saltar si el punto está detrás de la cámara
            if (pt1_2d.x < 0 || pt2_2d.x < 0) continue;
            
            let pt1 = new cv.Point(Math.round(pt1_2d.x), Math.round(pt1_2d.y));
            let pt2 = new cv.Point(Math.round(pt2_2d.x), Math.round(pt2_2d.y));

            // Clip line a los límites de la imagen (similar a cv2.clipLine)
            // Si la línea intersecta con los bordes, dibujarla
            const isInside1 = pt1.x >= 0 && pt1.x < w && pt1.y >= 0 && pt1.y < h;
            const isInside2 = pt2.x >= 0 && pt2.x < w && pt2.y >= 0 && pt2.y < h;
            
            // Si al menos uno de los puntos está dentro o la línea cruza los bordes, dibujarla
            if (isInside1 || isInside2 || 
                (pt1.x < 0 && pt2.x >= 0) || (pt1.x >= w && pt2.x < w) ||
                (pt1.y < 0 && pt2.y >= 0) || (pt1.y >= h && pt2.y < h)) {
                cv.line(image, pt1, pt2, green, 1);
            }
        }

        // Proyectar esquinas elevadas
        const elevatedCorners3D = new cv.Mat(4, 1, cv.CV_32FC3);
        for (let i = 0; i < 4; i++) {
            const x = this.worldPoints3DCorners.floatAt(i, 0);
            const y = this.worldPoints3DCorners.floatAt(i, 1);
            const z = this.worldPoints3DCorners.floatAt(i, 2) + heightCm;
            
            const idx = i * 3;
            elevatedCorners3D.data32F[idx] = x;
            elevatedCorners3D.data32F[idx + 1] = y;
            elevatedCorners3D.data32F[idx + 2] = z;
        }

        // Proyectar esquinas elevadas manualmente
        const projectedCorners2D = this.projectPoints3DTo2D(elevatedCorners3D);

        // Dibujar líneas desde esquinas base a esquinas elevadas (blanco)
        const white = new cv.Scalar(255, 255, 255, 0);
        for (let i = 0; i < 4; i++) {
            const baseX = this.imagePoints2D.floatAt(i, 0);
            const baseY = this.imagePoints2D.floatAt(i, 1);
            const projPt = projectedCorners2D[i];
            
            // Saltar si el punto está detrás de la cámara
            if (projPt.x < 0) continue;
            
            const p1 = new cv.Point(Math.round(baseX), Math.round(baseY));
            const p2 = new cv.Point(Math.round(projPt.x), Math.round(projPt.y));
            
            // Dibujar línea siempre (las esquinas base deberían estar dentro)
            cv.line(image, p1, p2, white, 1);
        }

        // Dibujar marcadores en los puntos base (morado) - asterisco
        const purple = new cv.Scalar(255, 0, 255, 0);
        for (let i = 0; i < 4; i++) {
            const x = this.imagePoints2D.floatAt(i, 0);
            const y = this.imagePoints2D.floatAt(i, 1);
            const center = new cv.Point(Math.round(x), Math.round(y));
            
            // Dibujar asterisco de 8 rayos
            const size = 20;
            const numRays = 8;
            
            for (let j = 0; j < numRays; j++) {
                const angle = (Math.PI * 2 * j) / numRays;
                const px = center.x + Math.cos(angle) * size;
                const py = center.y + Math.sin(angle) * size;
                const endPoint = new cv.Point(Math.round(px), Math.round(py));
                
                // Dibujar línea desde el centro hacia afuera
                cv.line(image, center, endPoint, purple, 2);
            }
            
            // Dibujar centro
            cv.circle(image, center, 2, purple, -1);
        }

        // Limpiar memoria
        allProjPoints3D.delete();
        elevatedCorners3D.delete();
    }

    /**
     * Obtiene las esquinas proyectadas a una altura específica
     * @param {number} heightCm - Altura en cm
     * @returns {Array<{x: number, y: number}>} Esquinas proyectadas en 2D
     */
    getProjectedCorners(heightCm) {
        // Crear puntos 3D elevados
        const elevatedCorners3D = new cv.Mat(4, 1, cv.CV_32FC3);
        for (let i = 0; i < 4; i++) {
            const x = this.worldPoints3DCorners.floatAt(i, 0);
            const y = this.worldPoints3DCorners.floatAt(i, 1);
            const z = this.worldPoints3DCorners.floatAt(i, 2) + heightCm;
            
            const idx = i * 3;
            elevatedCorners3D.data32F[idx] = x;
            elevatedCorners3D.data32F[idx + 1] = y;
            elevatedCorners3D.data32F[idx + 2] = z;
        }

        // Proyectar esquinas elevadas manualmente
        const projectedCorners2D = this.projectPoints3DTo2D(elevatedCorners3D);

        // Convertir a array de puntos
        const corners = projectedCorners2D.map(pt => ({
            x: Math.round(pt.x),
            y: Math.round(pt.y)
        }));

        elevatedCorners3D.delete();

        return corners;
    }

    /**
     * Limpia la memoria de OpenCV
     */
    delete() {
        this.worldPoints3DCorners.delete();
        this.imagePoints2D.delete();
        this.cameraMatrix.delete();
        this.distCoeffs.delete();
        this.rvec.delete();
        this.tvec.delete();
    }
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
        calculateRealDistance,
        encodeImageToBase64,
        calculateBoundingBox,
        SurfaceProjector
    };
}

// También exportar globalmente para uso en HTML
if (typeof window !== 'undefined') {
    window.DetectionAlgorithm = {
        config,
        adaptiveColorFilter,
        findCenters,
        orderPoints,
        kMeansClustering,
        processFrame,
        drawExtendedGrid,
        calculateRealDistance,
        encodeImageToBase64,
        calculateBoundingBox,
        SurfaceProjector
    };
}