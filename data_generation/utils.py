import os
import random

import cv2
import numpy as np
from PIL import Image
from itertools import permutations
from matplotlib.path import Path

def cat_lists(lists):
    o = []
    for l in lists:
        o += l
    return o


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0: 
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

# helper functions
from shapely.geometry import Polygon, Point
import numpy as np

def sample_position_inside_1(s1, s2, scale, n_candidates: int = 200):
    """
    Centros de s2 escalado que queden totalmente dentro de s1
    (usando disco de radio r_small para erosión).
    """
    c1 = s1.get_contour()
    outer = Polygon(c1)
    # radio aproximado de s2
    c2 = s2.get_contour() * scale
    cen2 = c2.mean(axis=0)
    r_small = np.max(np.linalg.norm(c2 - cen2, axis=1))
    # región factible tras erosión
    region = outer.buffer(-r_small)
    if region.is_empty:
        return np.empty((0,2))
    # bounds válidos
    minx, miny, maxx, maxy = region.bounds
    lo = np.array([minx, miny], dtype=float)
    hi = np.array([maxx, maxy], dtype=float)
    if np.any(hi <= lo):
        return np.empty((0,2))
    # muestreo vectorizado
    pts = np.random.rand(n_candidates, 2) * (hi - lo)[None, :] + lo[None, :]
    # filtro por contención en la región
    mask = np.array([region.contains(Point(x, y)) for x, y in pts])
    return pts[mask]


def sample_position_outside_1(s1, s2, scale, n_candidates: int = 200):
    """
    Centros de s2 escalado que queden totalmente fuera de s1,
    muestreados en bbox extendido.
    """
    c1 = s1.get_contour()
    outer = Polygon(c1)
    # radio aproximado de s2
    c2 = s2.get_contour() * scale
    cen2 = c2.mean(axis=0)
    r_small = np.max(np.linalg.norm(c2 - cen2, axis=1))
    # bbox extendido
    minx, miny = c1.min(0) - r_small
    maxx, maxy = c1.max(0) + r_small
    lo = np.array([minx, miny], dtype=float)
    hi = np.array([maxx, maxy], dtype=float)
    # muestreo vectorizado
    pts = np.random.rand(n_candidates, 2) * (hi - lo)[None, :] + lo[None, :]
    valid = []
    for x, y in pts:
        small = Polygon(c2 + np.array([x, y]))
        if small.disjoint(outer):
            valid.append((x, y))
    return np.array(valid)


def sample_position_inside_many(s1, shapes, scales, n_candidates: int = 500):
    """
    Sets de centros para shapes escalados que queden dentro de s1
    sin solaparse (aprox. con discos).
    """
    c1 = s1.get_contour()
    outer = Polygon(c1)
    # radios de cada shape
    radii = []
    for shp, sc in zip(shapes, scales):
        c2 = shp.get_contour() * sc
        cen2 = c2.mean(axis=0)
        r = np.max(np.linalg.norm(c2 - cen2, axis=1))
        radii.append(r)
    # erosión con el mayor radio
    R = max(radii)
    region = outer.buffer(-R)
    if region.is_empty:
        return np.zeros((0, len(shapes), 2))
    # bounds válidos
    minx, miny, maxx, maxy = region.bounds
    lo = np.array([minx, miny], dtype=float)
    hi = np.array([maxx, maxy], dtype=float)
    # muestreo vectorizado de candidatos
    pts = np.random.rand(n_candidates, len(shapes), 2) * (hi - lo)[None, None, :] + lo[None, None, :]
    valids = []
    for centers in pts:
        # 1) todos los centros en la región
        if not all(region.contains(Point(*centers[i])) for i in range(len(shapes))):
            continue
        # 2) no solapamiento circular
        ok = True
        for i in range(len(shapes)):
            for j in range(i+1, len(shapes)):
                if np.linalg.norm(centers[i] - centers[j]) <= radii[i] + radii[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valids.append(centers)
    return np.array(valids)



def sample_int_sum_n(n_numbers, s, min_v=0):
    samples = np.random.rand(n_numbers)
    samples = samples/samples.sum()*s
    samples = np.ceil(samples).astype(int)
    samples[samples<min_v] = min_v
    
    while samples.sum()>s:
        diff = samples.sum() - s    
        idx = np.where(samples>min_v)[0]
        if diff<len(idx):
            idx = np.random.choice(idx, size=diff, replace=False)
        samples[idx] -=1
    return samples    


# different n values that cover a range without overlapping with minimum distances between them
def sample_over_range(range_, min_dists):
    n_values = len(min_dists)
    
    dists = np.random.rand(n_values)
    dists = dists / dists.sum() * (range_[1] - range_[0] - min_dists.sum())
    dists[0] = dists[0] * np.random.rand()
    dists = dists + min_dists
    v = np.cumsum(dists)
    v = v - min_dists[0]/2 + range_[0]

    return v

def sample_over_range_t(n_samples, range_, min_dists):
    if len(range_.shape) == 1:
        range_ = range_[None,:]
    if len(min_dists.shape) == 1:
        min_dists = min_dists[None,:]

    n_values = min_dists.shape[1]
    
    dists = np.random.rand(n_samples, n_values)
    dists = dists / dists.sum(1)[:,None] * (range_[:,1] - range_[:,0] - min_dists.sum(1)[:,None])
    dists[:,0] = dists[:,0] * np.random.rand(n_samples)
    dists = dists + min_dists
    v = np.cumsum(dists, 1)
    v = v - min_dists/2 + range_[:,0:1]

    return v


def sample_positions(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]

    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_

def sample_positions_square(size):
    """
    Sample positions in [0,1]x[0,1] for 4 objects, such that they are placed in a square formation.
    size: (n,1) array of object sizes
    """

    # Generate square
    square = np.array([[-0.5, -0.5],
                      [ 0.5, -0.5],
                      [ 0.5,  0.5],
                      [-0.5,  0.5]])

    # Rotate randomly
    angle = np.random.rand() * 2 * np.pi
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle),  np.cos(angle)]])
    square_rotated = square @ rotation_matrix

    # Scale
    slack = 0.02
    max_size = size.max() * np.sqrt(2)     
    max_coord = (square_rotated + max_size / 2).max(axis=0)
    min_coord = (square_rotated - max_size / 2).min(axis=0)

    scale_max = (1 - max_size)/(max_coord[0] - min_coord[0])
    scale = np.random.uniform(max_size + slack, scale_max - slack)
    square_rotated *= scale 
    max_coord = (square_rotated + max_size / 2).max(axis=0)
    min_coord = (square_rotated - max_size / 2).min(axis=0)
    w = max_coord[0] - min_coord[0] 
    if w > 1:
        print("w > 1")
        print(w)

    # Locate square
    position = np.random.uniform(w*0.5 + slack, 1 - w*0.5 - slack, 2)
    square_rotated += position 

    xy = square_rotated
    # print("\n Checking \n")
    # print(check_square(xy))
    return xy

def squared_distance(p1, p2):
    return np.sum((p1 - p2) ** 2)

def check_square(xy):

    for perm in permutations(xy):
        a, b, c, d = perm
        d1 = squared_distance(a,b)
        d2 = squared_distance(b,c)
        d3 = squared_distance(c,d)  
        d4 = squared_distance(d,a)
        diag1 = squared_distance(a,c)
        diag2 = squared_distance(b,d)

        difs = [abs(d1 - d2), abs(d2 - d3), abs(d3 - d4), abs(d4 - d1)]

        if all(d < 1e-6 for d in difs) and abs(diag1 - diag2) < 1e-6:
            return True
    return False

def sample_positions_equidist(size, max_attempts=100, max_inner_attempts=50):

    """
    Sample positions in [0,1] x [0,1] for 4 objects, such that 
    distance between 1 and 2 is equal to distance between 3 and 4
    """
    n_objects = size.shape[0]
    if n_objects != 4:
        raise ValueError("This function only supports 4 objects.")

    p1 = np.random.uniform(size[0]/2, 1 - size[0]/2, 2)

    # Calculate distance from p1 to furthest corner
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # max_distance = np.sqrt(np.max([squared_distance(p1, corner) for corner in corners])) - size[1] * np.sqrt(2) * 0.5
    max_distance = np.linalg.norm(corners - p1, axis=1).max() - size[1] * np.sqrt(2) * 0.5
    min_distance = size[0] * np.sqrt(2) * 0.5 + size[1] * np.sqrt(2) * 0.5

    for _ in range(max_attempts):

        dist = np.random.uniform(min_distance, max_distance)

        p2_inner = False
        for _ in range(max_inner_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            p2 = p1 + dist * np.array([np.cos(angle), np.sin(angle)])           
            if (0 <= p2[0] - size[1]/2 and p2[1] + size[1]/2 <= 1 and
                0 <= p2[1] - size[1]/2 and p2[0] + size[1]/2 <= 1):
                p2_inner = True
                break
        
        # if not p2_inner:
        #     raise ValueError("Failed to find a valid position for p2 after multiple attempts.")
        
        # Calculate position for p3 and p4
        p3_inner = False
        for _ in range(max_inner_attempts):
            p3 = np.random.uniform(size[2]/2, 1 - size[2]/2, 2)
            if (p1[0] - size[0]/2 <= p3[0] + size[2]/2 and
                p1[0] + size[0]/2 >= p3[0] - size[2]/2 and
                p1[1] - size[0]/2 <= p3[1] + size[2]/2 and
                p1[1] + size[0]/2 >= p3[1] - size[2]/2) or (
                p2[0] - size[1]/2 <= p3[0] + size[2]/2 and
                p2[0] + size[1]/2 >= p3[0] - size[2]/2 and
                p2[1] - size[1]/2 <= p3[1] + size[2]/2 and
                p2[1] + size[1]/2 >= p3[1] - size[2]/2):
                continue
            else:
                p3_inner = True
                break
        # if not p3_inner:
        #     raise ValueError("Failed to find a valid position for p3 after multiple attempts.")
        
        p4_inner = False
        for _ in range(max_inner_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            p4 = p3 + dist * np.array([np.cos(angle), np.sin(angle)])
            if (0 <= p4[0] - size[3]/2 and p4[1] + size[3]/2 <= 1 and
                0 <= p4[1] - size[3]/2 and p4[0] + size[3]/2 <= 1):
                if (p1[0] - size[0]/2 <= p4[0] + size[3]/2 and
                    p1[0] + size[0]/2 >= p4[0] - size[3]/2 and
                    p1[1] - size[0]/2 <= p4[1] + size[3]/2 and
                    p1[1] + size[0]/2 >= p4[1] - size[3]/2) or (
                    p2[0] - size[1]/2 <= p4[0] + size[3]/2 and
                    p2[0] + size[1]/2 >= p4[0] - size[3]/2 and
                    p2[1] - size[1]/2 <= p4[1] + size[3]/2 and
                    p2[1] + size[1]/2 >= p4[1] - size[3]/2):
                    continue
                else:
                    p4_inner = True
                    break
            # if not p4_inner:
            #     raise ValueError("Failed to find a valid position for p4 after multiple attempts.")

        if p2_inner and p3_inner and p4_inner:
            break

    if not (p2_inner and p3_inner and p4_inner):
        raise ValueError("Failed to find valid positions for all objects after multiple attempts.")
        
    return np.array([p1, p2, p3, p4])



def sample_positions_bb(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]

    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_

# Suma de los tamaños debe ser menor que 1
def sample_positions_align(size):
    size = size[0]  
    n_objects = size.shape[0]
    widths = size.flatten()
    # Random line
    # Random angle
    theta = np.random.rand() * 2 * np.pi
    direction = np.array([np.cos(theta), np.sin(theta)])
    # Center of [0,1]x[0,1] square
    center = np.array([0.5,0.5])

    # Random gaps between objects
    min_gap = 0.08
    gap1 = random.uniform(min_gap, widths.sum())
    while widths.sum() - gap1 < min_gap:
        gap1 = random.uniform(min_gap, widths.sum())
    gap2 = random.uniform(min_gap, widths.sum() - gap1)
    gaps = np.array([gap1, gap2])
    total_gap = gaps.sum()

    total_length = widths.sum() + total_gap

    # Positions along the line
    positions = []
    pos = center
    for i, w in enumerate(widths):
        positions.append(pos + direction * (w / 2))
        if i < n_objects - 1:
            pos = pos + direction * (w + gaps[i])
    positions = np.stack(positions, axis=0)

    xy = positions[None, ...]

    return xy



def sample_positions_symmetric_pairs(
    size, 
    margin_x=0.08, 
    margin_y=0.08, 
    min_dist=0.01, 
    max_tries=100,
    max_pair_tries=50
):
    """
    Genera posiciones (x, y) para 3 pares de objetos simétricos respecto al eje x=0.5,
    muestreando cada par por separado y comprobando solapamientos contra todos los anteriores.
    size: array (6,1) con los diámetros de cada figura (3 pares).
    Return: xy de shape (6, 2)
    """
    assert size.shape[0] == 6, "size debe tener longitud 6 (3 pares)."
    sizes = size.flatten()
    # radios de cada figura
    radios = sizes / 2  

    for attempt in range(max_tries):
        xy = np.zeros((6, 2))
        placed_centers = []
        placed_radii  = []

        success = True
        # iterar sobre los 3 pares
        for pair in range(3):
            pair_placed = False
            for _ in range(max_pair_tries):
                s = radios[2*pair]  # radio de la figura del par
                # muestrea x izquierdo dentro de márgenes
                x_l = np.random.uniform(margin_x + s, 0.5 - min_dist - s)
                x_r = 1.0 - x_l
                # muestrea y común dentro de márgenes
                y = np.random.uniform(margin_y + s, 1 - margin_y - s)

                cand_l = np.array([x_l, y])
                cand_r = np.array([x_r, y])

                # chequeo solapamiento contra ya ubicados
                overlap = False
                for (ec, er) in zip(placed_centers, placed_radii):
                    if np.linalg.norm(cand_l - ec) < (s + er) + min_dist or \
                       np.linalg.norm(cand_r - ec) < (s + er) + min_dist:
                        overlap = True
                        break

                if overlap:
                    continue

                # si llega acá, ambos son válidos
                idx_l, idx_r = 2*pair, 2*pair+1
                xy[idx_l] = cand_l
                xy[idx_r] = cand_r
                placed_centers += [cand_l, cand_r]
                placed_radii  += [s, s]
                pair_placed = True
                break

            if not pair_placed:
                success = False
                break

        if success:
            return xy

    # si no encontró configuración tras max_tries:
    print(f"Advertencia: no halló disposición tras {max_tries} intentos.")
    return xy


def sample_points_in_circle(
    n,
    radius=0.2,
    center=(0.5, 0.5),
    on_edge=False
):
    """
    Devuelve una lista de n coordenadas (x, y) dentro de un círculo de radio `radius`
    centrado en `center`.
    - Si on_edge es True, todos los puntos estarán exactamente en el borde.
    - Si on_edge es False, todos los puntos se ubican aleatoriamente dentro del área.
    """
    center = np.asarray(center)
    angles = np.random.uniform(0, 2 * np.pi, size=n)
    if on_edge:
        rhos = np.full(n, radius)
    else:
        rhos = radius * np.sqrt(np.random.rand(n))
    coords = center + np.stack([rhos * np.cos(angles), rhos * np.sin(angles)], axis=1)
    return coords


def sample_positions_circle(
    sizes,
    min_circle_radius=0.10,
    max_circle_radius=1.0,
    max_tries=200,
    max_clone_tries=100
):
    """
    Coloca primero el odd aleatoriamente en [margin,1-margin]^2.
    Luego muestrea clones uno a uno:
      - Descarta si sale del margen.
      - Descarta si solapa con odd o con cualquier clone previo.
      - Guarda si es válido.
    Devuelve cuando hay 3 clones válidos o tras max_tries.
    """
    sizes = np.array(sizes).flatten()
    margin     = np.max(sizes) / 2
    rad_clone  = sizes[0]    # asumimos clones todos con el mismo tamaño
    rad_odd    = sizes[3]    # odd

    for attempt in range(max_tries):
        # 1) ubica el odd
        odd_center = np.random.uniform(margin, 1 - margin, size=2)

        # 2) elige radio del círculo
        circle_radius = np.random.uniform(min_circle_radius, max_circle_radius)

        clones = []
        clone_attempts = 0

        # 3) muestrea clones hasta tener 3 válidos
        while len(clones) < 3 and clone_attempts < max_clone_tries:
            theta     = np.random.rand() * 2 * np.pi
            candidate = odd_center + np.array([np.cos(theta), np.sin(theta)]) * circle_radius

            # Chequeo de márgenes
            if not np.all((candidate >= margin) & (candidate <= 1 - margin)):
                clone_attempts += 1
                continue

            # Preparo lista de centros y radios para chequeo genérico
            existing_centers = clones + [odd_center]
            existing_radii   = [rad_clone] * len(clones) + [rad_odd]

            # Chequeo genérico de solapamiento
            overlap = False
            for ec, er in zip(existing_centers, existing_radii):
                # distancia centro-a-centro vs semidiámetros
                if np.linalg.norm(candidate - ec) < (rad_clone + er) / 2:
                    overlap = True
                    break

            if overlap:
                clone_attempts += 1
                continue

            # posición válida: la guardo
            clones.append(candidate)

        # Si ya tengo 3 clones, retorno las 4 posiciones
        if len(clones) == 3:
            return np.vstack([clones, odd_center])

    # Si no encontró configuración tras max_tries
    print(f"Advertencia: no halló disposición tras {max_tries} intentos.")
    return np.vstack([clones, odd_center])


def compute_inscribed_circle(shape, resolution=100):
    """
    Dado un Shape, calcula el mayor círculo inscrito en su contorno.
    Parámetros:
      shape      : objeto Shape con método get_contour() en [0,1]²
      resolution : tamaño en píxeles del lienzo para rasterizar
    Retorna:
      center : np.array([cx, cy])  – centro normalizado en [0,1]²
      radius : float               – radio normalizado en [0,1]
    """
    # 1) Crear lienzo vacío
    img = np.zeros((resolution, resolution), dtype=np.uint8)

    # 2) Obtener contorno en coordenadas normalizadas y escalar a píxeles
    contour = shape.get_contour()                  # (N,2) en [0,1]
    pts = np.round(contour * (resolution - 1))     # escalar
    pts = pts.astype(np.int32).reshape(-1, 1, 2)   # formato OpenCV

    # 3) Rellenar polígono
    cv2.fillPoly(img, [pts], color=255)

    # 4) Distance transform
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    # 5) Buscar máximo: radio y posición
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)

    # 6) Normalizar de vuelta a [0,1]
    center = np.array([max_loc[0], max_loc[1]], dtype=float) / (resolution - 1)
    radius = float(max_val) / (resolution - 1)

    return center, radius


def sample_random_colors(n_samples):
    h = np.random.rand(n_samples)
    s = np.random.rand(n_samples) * 0.5 + 0.5
    v = np.random.rand(n_samples) * 1

    color = np.stack([h,s,v],1)
    return color


def sample_shuffle_unshuffle_indices(n):
    perm = np.random.permutation(n)
    indices_input = np.arange(n)
    indices_output = indices_input[perm]
    rev_perm = (indices_output[:, None] == indices_input).argmax(axis=0)
    return perm, rev_perm


def shuffle_t(t, perms):
    # t.reshape()
    for i in range(t.shape[0]):
        t[i] = t[i, perms[i]]


def sample_contact(s1, s2, scale, direction=0):
    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * scale
    
    if direction==0:
        p1 = np.argmax(c1[:,0]) 
        p2 = np.argmin(c2[:,0]) 
    elif direction==1:
        p1 = np.argmin(c1[:,0]) 
        p2 = np.argmax(c2[:,0]) 
    elif direction==2:
        p1 = np.argmax(c1[:,1]) 
        p2 = np.argmin(c2[:,1]) 
    elif direction==3:
        p1 = np.argmin(c1[:,1]) 
        p2 = np.argmax(c2[:,1]) 

    xy2 = (c2.max(0) + c2.min(0))/2 - c2[p2] + c1[p1]
    # xy1 = np.zeros(2)
    
    return xy2


def sample_contact_many(shapes, sizes, image_dim=128, a=None):
    n_objects = len(shapes)
    contours = [shapes[i].get_contour() * sizes[i] for i in range(n_objects)]

    # intialize clump as the first object
    clump = contours[0]
    positions = np.zeros([1,2])
    clump_size = np.ones(2) * sizes[0]
    for i in range(1, n_objects):
        # sample direction
        if a is None:
            angle = np.random.rand() * 2 * np.pi
        elif isinstance(a, float):
            angle = a
        else:
            angle = a[i]
            
        pos2 = (sizes[i]+clump_size) * np.array([np.cos(angle), np.sin(angle)])[None,:]
        
        idx_p_contact_clump = (clump * (np.cos(angle),np.sin(angle))).sum(-1) > 0
        idx_p_contact_object = (contours[i] * (np.cos(angle),np.sin(angle))).sum(-1) < 0
        
        # move object in direction
        c = contours[i] + pos2
        
        idx_min = np.linalg.norm(clump[idx_p_contact_clump][:,None,:] - c[idx_p_contact_object][None,:,:], axis=2).argmin()
        s_ = idx_p_contact_object.sum()
        idx_min_clump, idx_min_object = idx_min // s_, idx_min % s_
        p_clump = clump[idx_p_contact_clump][idx_min_clump]
        p_obj = contours[i][idx_p_contact_object][idx_min_object]
        new_pos = (p_clump - p_obj)*(1-4/image_dim)
        
        clump = np.concatenate([clump, contours[i]+new_pos[None,:]], 0)
        bb = clump.min(0), clump.max(0)
        
        clump = clump - (bb[1] + bb[0])/2
        clump_size = bb[1] - bb[0]

        positions = np.concatenate([positions,new_pos[None,:]], 0)
        positions = positions - (bb[1] + bb[0])/2

    return positions, clump_size

def flip_diag_scene(xys, shapes):

    for s in shapes:
        s.flip_diag()

    for i, xy in enumerate(xys):
        xys[i] = xy[::-1]

    return xys, shapes

def render_cv(xy, size, shapes, color=None, image_size=128):
        
    color = [hsv_to_rgb(c[0], c[1], c[2]) for c in color]

    image = (np.ones([image_size,image_size, 3]) * 255).astype(np.uint8)

    for i in range(len(shapes)):
        size_ = size[i]
        s_ = shapes[i]
        s_.scale(size_)
        xy_ = xy[i]

        c = s_.get_contour()
        
        c = (c*image_size).astype(int)

        c_ = np.concatenate([c,c[0:1]],0)
        dist = np.abs(c_[1:] - c_[:-1]) 
        c = c[(dist>0).any(1)]

        c = c + (xy_[None,:] * image_size).astype(int)

        col_ = (np.array(color[i])*255).tolist()
        cv2.drawContours(image, [c], -1, col_, 1)
        
    return image


def render_ooo(xy, size, shape, color, image_size=128):

    images = []
    for i in range(len(shape)):
        im = render_cv(xy[i], size[i], shape[i], color[i], image_size=128)
        im = np.pad(im, [[4,4], [4,4], [0,0]], constant_values=0)
        images.append(im)

    images = np.concatenate(images, axis=1)
    
    return images

def render_scene_safe(xy, size, shape, color, image_size=128):
    """
    Renderiza múltiples figuras en una sola imagen, asegurando el formato correcto.
    """
    # Normalizar colores
    clean_colors = []
    for c in color:
        c = np.array(c).flatten()
        if c.shape[0] != 3:
            raise ValueError(f"Color inválido: se esperaba un vector de 3 elementos, pero llegó {c.shape}")
        clean_colors.append(c)

    # Desenrollar shape: de [[s], [s], ...] → [s, s, ...]
    clean_shapes = [s[0] if isinstance(s, list) else s for s in shape]

    return render_cv(xy, size, clean_shapes, clean_colors, image_size=image_size)




def save_image_human_exp(images, meta, base_path):
    im_shape = images.shape
    
    dim_0 = im_shape[1]//4

    pad = (dim_0-128)//2

    images = images.reshape([im_shape[0]//dim_0, dim_0, 4, dim_0, 3]).transpose([0,2,1,3,4]).reshape([-1, dim_0, dim_0, 3])
    for i in range(len(images)):
        idx1, idx2 = i//4, i%4 
        save_path = os.path.join(base_path, '{:02d}_{}.png'.format(idx1, idx2))
        img = Image.fromarray(images[i,pad:dim_0-pad, pad:dim_0-pad]).convert('RGB')
        img.save(save_path)


def save_image_bin(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    if images.dtype != np.uint8:
        images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('1')
    img.save(save_path)



def save_image(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    # if images.dtype != np.uint8:
    #     images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('RGB')
    img.save(save_path)
