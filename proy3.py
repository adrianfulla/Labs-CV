## Imports

import cv2
import numpy as np

# Read images

img1 = cv2.imread('input/panorama_4/1.jpg')
img2 = cv2.imread('input/panorama_4/2.jpg')
img3 = cv2.imread('input/panorama_4/3.jpg')

# Elegir una imagen base

base_img = img2

# Establecer un mapa de correspondencia entre las imágenes
# (1, 2) -> (2, 3) -> (3, 1)


# Crear el detector ORB
orb = cv2.ORB_create(nfeatures=2000)

# Detectar puntos clave y descriptores
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp3, des3 = orb.detectAndCompute(img3, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Matching img1 <-> img2
matches_12 = bf.match(des1, des2)
matches_12 = sorted(matches_12, key=lambda x: x.distance)

# Matching img3 <-> img2
matches_32 = bf.match(des3, des2)
matches_32 = sorted(matches_32, key=lambda x: x.distance)


# Calcular la homografía entre las imágenes


# Función para obtener homografía entre dos imágenes
def get_homography(kpA, kpB, matches):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
    return H, mask


H_12, mask_12 = get_homography(kp1, kp2, matches_12)
H_32, mask_32 = get_homography(kp3, kp2, matches_32)


# Tamaños de las imágenes
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]

# Coordenadas de las esquinas de cada imagen
corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
corners3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]]).reshape(-1, 1, 2)


# Transformar esquinas a la vista de la imagen base
warped_corners1 = cv2.perspectiveTransform(corners1, H_12)
warped_corners3 = cv2.perspectiveTransform(corners3, H_32)

# Concatenar todas las esquinas para calcular la extensión del canvas
all_corners = np.concatenate((warped_corners1, corners2,corners3, warped_corners3), axis=0)

# Obtener los extremos del canvas
[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Traslación para evitar coordenadas negativas
translate = [-xmin, -ymin]
T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])  # matriz de traslación

# Tamaño del canvas final
canvas_width = xmax - xmin
canvas_height = ymax - ymin

# Warp de cada imagen al canvas
result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Warp de img1
warp1 = cv2.warpPerspective(img1, T @ H_12, (canvas_width, canvas_height))

# Warp de img3
warp3 = cv2.warpPerspective(img3, T @ H_32, (canvas_width, canvas_height))

# Pegar img2 (base) directamente en el canvas
result[translate[1]:translate[1]+h2, translate[0]:translate[0]+w2] = img2

# Combinar los warps (img1 y img3) encima del resultado base
mask1 = (warp1 > 0)
result[mask1] = warp1[mask1]

mask3 = (warp3 > 0)
result[mask3] = warp3[mask3]

# Mostrar o guardar el resultado
# cv2.imshow('Panorama', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imwrite('output/panorama_4/panorama.jpg', result)


def warp_and_blend_feathering(img, H, canvas_size, T, weight_acc, image_acc):
    h, w = img.shape[:2]
    
    # Warp de imagen
    warped_img = cv2.warpPerspective(img, T @ H, canvas_size)
    
    # Crear una máscara blanca del tamaño de la imagen original
    mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, T @ H, canvas_size)
    
    # Crear una máscara suave (feathered) difuminando bordes
    kernel_size = 51  # cuanto más grande, más suave el borde
    soft_mask = cv2.GaussianBlur(warped_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    soft_mask = soft_mask / 255.0  # Normalizar entre 0 y 1
    soft_mask = np.expand_dims(soft_mask, axis=2)  # Convertir a shape (H, W, 1)
    
    # Acumular imagen ponderada y pesos
    image_acc += warped_img.astype(np.float32) * soft_mask
    weight_acc += soft_mask


def create_distance_weight_map(img):
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    
    # Compute distances from center
    center_y, center_x = h // 2, w // 2
    dist_from_center_x = np.abs(x - center_x) / (w / 2)
    dist_from_center_y = np.abs(y - center_y) / (h / 2)
    
    # Combine using radial distance (adjust power for faster/slower falloff)
    dist = np.sqrt(dist_from_center_x**2 + dist_from_center_y**2)
    weight = np.clip(1.0 - dist**1.5, 0, 1)
    
    # Apply smoothing 
    weight = cv2.GaussianBlur(weight, (71, 71), 0)
    
    return weight  # Return as 2D array

def warp_and_blend_center_weighted(img, H, canvas_size, T, weight_acc, image_acc):
    # Creacion de mapa de pesos que enfatiza las partes centrales de la imagen
    weight_map = create_distance_weight_map(img)
    
    # Mapa de pesos con 3 dimensiones y shape (h, w, 1)
    weight_map_3d = np.expand_dims(weight_map, axis=2)
    
    # Aplicacion del mapa de pesos a la imagen
    weighted_img = img.astype(np.float32) * weight_map_3d
    
    # Warp de la imagen
    warped_weighted_img = cv2.warpPerspective(weighted_img, T @ H, canvas_size)
    

    # Warp separado por imagen para mantener la tri-dimensionalidad
    warped_weight = cv2.warpPerspective(weight_map, T @ H, canvas_size)
    
    # Mantener la tri-dimensionalidad del mapa de pesos
    warped_weight_3d = np.expand_dims(warped_weight, axis=2)
    
    # Accumulate
    image_acc += warped_weighted_img  
    weight_acc += warped_weight_3d
    
canvas_size = (canvas_width, canvas_height)

image_acc = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
weight_acc = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)


warp_and_blend_center_weighted(img1, H_12, canvas_size, T, weight_acc, image_acc)
warp_and_blend_center_weighted(img2, np.eye(3), canvas_size, T, weight_acc, image_acc)
warp_and_blend_center_weighted(img3, H_32, canvas_size, T, weight_acc, image_acc)

# warp_and_blend_feathering(img1, H_12, canvas_size, T, weight_acc, image_acc)
# warp_and_blend_feathering(img2, np.eye(3), canvas_size, T, weight_acc, image_acc)
# warp_and_blend_feathering(img3, H_32, canvas_size, T, weight_acc, image_acc)

epsilon = 1e-10
weight_acc[weight_acc < epsilon] = epsilon


blended_result = (image_acc / weight_acc).astype(np.uint8)


# cv2.imwrite('output/panorama_4/blended_panorama_feathering.jpg', blended_result)
cv2.imwrite('output/panorama_4/blended_panorama_center_weighted.jpg', blended_result)
# Mostrar o guardar
# cv2.imshow('Blended Panorama', blended_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('output/blended_panorama.jpg', blended_result)



