import copy
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

gap_max = 0.006

nb_max_pixels = 50*50

margin = 1


class Shape(object):

    def __init__(self, gap_max=0.007, radius = 0.5, hole_radius = 0.05, randomize=True, sym_flip=False, sym_rot=1):
        self.gap_max = gap_max
        self.radius = radius
        self.hole_radius = hole_radius
        self.nb_pixels = 0
        self.x_pixels = []
        self.y_pixels = []
        self.transformations = []

        self.sym_flip = sym_flip
        self.sym_rot = sym_rot
        
        if randomize:
            self.randomize()
        

    def generate_part_part(self, radius, hole_radius, x1, y1, x2, y2): 
    
        if abs(x1 - x2) > self.gap_max or abs(y1 - y2) > self.gap_max:

            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)/5
            
            dx = (2 * np.random.rand() - 1) * d
            dy = (2 * np.random.rand() - 1) * d
            while dx**2 + dy**2 > d**2:
                dx = (2 * np.random.rand() - 1) * d
                dy = (2 * np.random.rand() - 1) * d
            x3 = (x1 + x2) / 2 + dx
            y3 = (y1 + y2) / 2 + dy

            while x3**2 + y3**2 > radius**2:
                dx = (2 * np.random.rand() - 1) * d
                dy = (2 * np.random.rand() - 1) * d
                while dx**2 + dy**2 > d**2:
                    dx = (2 * np.random.rand() - 1) * d
                    dy = (2 * np.random.rand() - 1) * d
                x3 = (x1 + x2) / 2 + dx
                y3 = (y1 + y2) / 2 + dy
                

            if self.generate_part_part(radius, hole_radius, x1, y1, x3, y3):
                return True

            if self.generate_part_part(radius, hole_radius, x3, y3, x2, y2):
                return True
            
        else:

            if x1**2 + y1**2 >= radius**2 or x1**2 + y1**2 < hole_radius**2:
                return True
            
            self.x_pixels.append(x1)
            self.y_pixels.append(y1)

        return False

    def generate_part(self, radius, hole_radius):

        err1, err2, err3, err4 = True, True, True, True
        
        while err1 or err2 or err3 or err4:
            nb_pixels = 0
            
            self.x_pixels = []
            self.y_pixels = []

            x1 = np.random.rand() * radius
            y1 = np.random.rand() * radius
            while x1**2 + y1**2 > radius**2 or x1**2 + y1**2 < hole_radius**2:
                x1 = np.random.rand() * radius
                y1 = np.random.rand() * radius

            x2 = -np.random.rand() * radius
            y2 = np.random.rand() * radius
            while x2**2 + y2**2 > radius**2 or x2**2 + y2**2 < hole_radius**2:
                x2 = -np.random.rand() * radius
                y2 = np.random.rand() * radius

            x3 = -np.random.rand() * radius
            y3 = -np.random.rand() * radius
            while x3**2 + y3**2 > radius**2 or x3**2 + y3**2 < hole_radius**2:
                x3 = -np.random.rand() * radius
                y3 = -np.random.rand() * radius

            x4 = np.random.rand() * radius
            y4 = -np.random.rand() * radius
            while x4**2 + y4**2 > radius**2 or x4**2 + y4**2 < hole_radius**2:
                x4 = np.random.rand() * radius
                y4 = -np.random.rand() * radius

            self.n_pixels1 = len(self.x_pixels)
            err1 = self.generate_part_part(radius, hole_radius, x1, y1, x2, y2)
            self.n_pixels2 = len(self.x_pixels)
            err2 = self.generate_part_part(radius, hole_radius, x2, y2, x3, y3)
            self.n_pixels3 = len(self.x_pixels)
            err3 = self.generate_part_part(radius, hole_radius, x3, y3, x4, y4)
            self.n_pixels4 = len(self.x_pixels)
            err4 = self.generate_part_part(radius, hole_radius, x4, y4, x1, y1)

    def randomize(self):
        self.x_pixels = []
        self.y_pixels = []

        # self.nb_pixels = 0
        
        self.generate_part(self.radius, self.hole_radius)

        self.nb_pixels = len(self.x_pixels)

        self.x_pixels = np.array(self.x_pixels)
        self.y_pixels = np.array(self.y_pixels)

        ### random rotation
        self.rotate(np.random.rand() * np.pi * 2)

        # Este código no hace nada (funciones no implementadas)
        # if self.sym_rot > 1:
        #     self.rot_symmetrize(self.sym_rot)
        # if self.sym_flip:
        #     self.flip_symmetrize()
        
        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        ### resets to a squre
        self.x_pixels = self.x_pixels/(self.x_pixels.max() - self.x_pixels.min())
        self.y_pixels = self.y_pixels/(self.y_pixels.max() - self.y_pixels.min())

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        self.wh = (1, 1)

        self.transformations = []

    def smooth(self, n_sampled_points = 100, fourier_terms = 20):

        # Recuperar figura base
        base_contour = self.get_contour()
        length = len(base_contour)

        # Samplear puntos equiespaciados
        idx = np.linspace(0, length-1, n_sampled_points, dtype=int)
        t_sel = np.linspace(0, 1, n_sampled_points)
        x_sel = base_contour[idx, 0]
        y_sel = base_contour[idx, 1]

        # Ajuste de Fourier
        cx = self.__fit_fourier(t_sel, x_sel, fourier_terms)
        cy = self.__fit_fourier(t_sel, y_sel, fourier_terms)

        t_curve = np.linspace(0, 1, n_sampled_points)
        x_curve = self.__eval_fourier(cx, t_curve)
        y_curve = self.__eval_fourier(cy, t_curve)

        # Normalizar figura
        x_curve -= (x_curve.max() + x_curve.min()) / 2
        y_curve -= (y_curve.max() + y_curve.min()) / 2
        x_curve /= (x_curve.max() - x_curve.min())
        y_curve /= (y_curve.max() - y_curve.min())

        self.x_pixels = x_curve
        self.y_pixels = y_curve
        self.nb_pixels = len(x_curve)
        self.wh = (1, 1)
        self.transformations = []

    def __fit_fourier(self, t, f, N):
        A = [np.ones_like(t)]
        for k in range(1, N+1):
            A.append(np.cos(2 * np.pi * k * t))
            A.append(np.sin(2 * np.pi * k * t))
        A = np.stack(A, axis=1)
        coeffs = np.linalg.lstsq(A, f, rcond=None)[0]
        return coeffs
    
    def __eval_fourier(self, coeffs, t):
        N = (len(coeffs) - 1) // 2
        result = coeffs[0] * np.ones_like(t)
        for k in range(1, N+1):
            result += coeffs[2*k-1] * np.cos(2 * np.pi * k * t) + coeffs[2*k] * np.sin(2 * np.pi * k * t)
        return result

    def rigid_transform(self, type = 'polygon', points = 3, rotate = 0):

        # Alterar figura según tipo de figura rígida deseada
        if type == 'polygon':
            theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
            # Rota para que un vértice apunte hacia arriba
            theta += np.pi / 2
            self.x_pixels = np.cos(theta)
            self.y_pixels = np.sin(theta)

        elif type == 'irregular':
            # Generamos poliedro irregular a partir de figura base
            # Seleccionamos points puntos y los conectamos
            idx = np.linspace(0, len(self.x_pixels)-1, points + 1, dtype=int)
            self.x_pixels = self.x_pixels[idx]
            self.y_pixels = self.y_pixels[idx]

        elif type == 'arrow':
            # Forma de flecha "↣"
            pts = np.array([
                [-0.5,  0.2],
                [-0.5, -0.2],
                [ 0.2, -0.2],
                [ 0.2, -0.5],
                [ 0.5,  0.0],
                [ 0.2,  0.5],
                [ 0.2,  0.2],
            ])
            self.x_pixels, self.y_pixels = pts[:, 0], pts[:, 1]

        elif type == 'arrow_line':
            pts = np.array([
                [0.5, 0.0],
                [-0.5, 0.0],
                [-0.2, 0.3],
                [-0.5, 0.0],
                [-0.2, -0.3],
                [-0.5, 0.0],
            ])
            self.x_pixels, self.y_pixels = pts[:, 0], pts[:, 1]

        if rotate:
            # Aplicar rotación en ángulo aleatorio
            angle = np.random.rand() * 2 * np.pi
            self.rotate(angle)

        self.nb_pixels = len(self.x_pixels)
        self.x_pixels = np.array(self.x_pixels)
        self.y_pixels = np.array(self.y_pixels)

        # Centrar
        self.x_pixels -= (self.x_pixels.max() + self.x_pixels.min()) / 2
        self.y_pixels -= (self.y_pixels.max() + self.y_pixels.min()) / 2

        # Normalizar a [-0.5, 0.5] en la dimensión más grande
        w = self.x_pixels.max() - self.x_pixels.min()
        h = self.y_pixels.max() - self.y_pixels.min()
        scale = 1 / max(w, h)
        self.x_pixels *= scale
        self.y_pixels *= scale

        self.bb = (w, h)
        self.wh = (w, h)
        self.transformations = []

    def symmetrize(self, rotate = 0):

        if self.x_pixels[0] < 0:   
            # Tomamos un lado de la figura
            mask = self.x_pixels >= 0

            # Copiamos el lado original
            x_copy = self.x_pixels[mask]
            y_copy = self.y_pixels[mask]

            self.x_pixels = self.x_pixels[mask][::-1]
            self.y_pixels = self.y_pixels[mask][::-1]

            # Reflejamos
            self.flip()

            # Concatenamos nuevamente el lado original sin reflejar
            self.x_pixels = np.concatenate([self.x_pixels, x_copy])
            self.y_pixels = np.concatenate([self.y_pixels, y_copy])

        elif self.x_pixels[0] >= 0:

            mask = self.x_pixels < 0
            
            # Copiamos el lado original
            x_copy = self.x_pixels[mask]
            y_copy = self.y_pixels[mask]

            self.x_pixels = self.x_pixels[mask][::-1]
            self.y_pixels = self.y_pixels[mask][::-1]

            # Reflejamos
            self.flip()

            # Concatenamos nuevamente el lado original sin reflejar
            self.x_pixels = np.concatenate([self.x_pixels, x_copy])
            self.y_pixels = np.concatenate([self.y_pixels, y_copy])

        if rotate:
            # Aplicar rotación en ángulo aleatorio
            angle = np.random.rand() * 2 * np.pi
            self.rotate(angle)

        self.nb_pixels = len(self.x_pixels)
        # Centrar
        self.x_pixels -= (self.x_pixels.max() + self.x_pixels.min()) / 2
        self.y_pixels -= (self.y_pixels.max() + self.y_pixels.min()) / 2

        # Normalizar a [-0.5, 0.5] en la dimensión más grande
        w = self.x_pixels.max() - self.x_pixels.min()
        h = self.y_pixels.max() - self.y_pixels.min()
        scale = 1 / max(w, h)
        self.x_pixels *= scale
        self.y_pixels *= scale

        self.bb = (w, h)
        self.wh = (w, h)
        self.transformations = []

    def flip_diag(self):
        self.x_pixels, self.y_pixels = self.y_pixels, self.x_pixels

    def get_contour(self):
        p1 = np.stack([self.x_pixels, self.y_pixels], 1)
        p1 = np.concatenate([p1, p1[:1]], 0)
        return p1

    def rotate(self, alpha):
        ux, uy = np.cos(alpha), -np.sin(alpha)
        vx, vy = np.sin(alpha), np.cos(alpha)
        
        x = self.x_pixels * ux + self.y_pixels * uy
        y = self.x_pixels * vx + self.y_pixels * vy

        self.x_pixels = x
        self.y_pixels = y
        
        self.transformations.append(('r', alpha))
        
        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)
        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    def get_bb(self):
        w = self.x_pixels.max() - self.x_pixels.min()
        h = self.y_pixels.max() - self.y_pixels.min()
        return (w,h)

    def scale(self, s): 
        if isinstance(s, tuple):
            self.x_pixels = self.x_pixels*s[0]
            self.y_pixels = self.y_pixels*s[1]
        else:
            self.x_pixels = self.x_pixels*s
            self.y_pixels = self.y_pixels*s
        
        self.transformations.append(('s', s))

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)
        # temp_size = w * h
        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    def set_wh(self, wh):
        current_w = self.x_pixels.max() - self.x_pixels.min()
        current_h = self.y_pixels.max() - self.y_pixels.min()
        
        scale = (wh[0]/current_w, wh[1]/current_h) 
        self.scale(scale)

    def set_size(self, s):
        current_size = np.sqrt(self.bb[0]*self.bb[1])
        self.scale(s/current_size)

    def clone(self):
        s = self.__class__(randomize=False)

        s.nb_pixels = copy.deepcopy(self.nb_pixels)
        s.x_pixels = np.copy(self.x_pixels)
        s.y_pixels = np.copy(self.y_pixels)

        # Copiar atributos opcionales si existen
        for attr in ['n_pixels1', 'n_pixels2', 'n_pixels3', 'n_pixels4',
                    'bb', 'circle_center', 'max_radius']:
            if hasattr(self, attr):
                setattr(s, attr, copy.deepcopy(getattr(self, attr)))

        s.wh = copy.deepcopy(self.wh)
        s.transformations = copy.deepcopy(self.transformations)

        return s

    
    def flip(self):
        self.x_pixels = - self.x_pixels

        self.transformations.append(('f', None))
    
    def subsample(self):
        # sample many intermediate points
        N = max(500//len(self.x_pixels), 1)
        xy = np.stack([self.x_pixels, self.y_pixels], 1)

        if N>1:
            a = np.linspace(0,1,N)
            xy = xy[:-1,None,:] + (xy[1:,None,:]-xy[:-1,None,:])*a[None,:,None]
            xy = xy.reshape([-1, 2])

        for i in range(1,len(xy)-1):
            xy[i] = (xy[i+1] + xy[i-1])/2 + (np.random.rand()-0.5)*(xy[i+1] - xy[i-1])[::-1] * np.array([-1,1]) /np.linalg.norm(xy[i+1] - xy[i-1]) * 0.07

        self.x_pixels = xy[:,0]
        self.y_pixels = xy[:,1]
    
    def reset(self):
        
        for t, v in reversed(self.transformations):
            if t == 'r':
                self.rotate(-v)
            if t == 's':
                if isinstance(v, tuple):
                    v = (1/v[0], 1/v[1])
                else:
                    v = 1/v
                self.scale(v)
            if t == 'f':
                self.flip()
        
        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        self.transformations = []
        
        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)

        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    def get_hole_radius(self):
        
        img = (np.zeros((100,100)) * 255).astype(np.uint8)
        x_pixels = np.concatenate([self.x_pixels, self.x_pixels[0:1]]) 
        y_pixels = np.concatenate([self.y_pixels, self.y_pixels[0:1]]) 
        
        contours = np.stack([x_pixels,y_pixels], 1) * 70 + 50
        contours = contours.astype(np.int32)
        contours[:10,:10]
        cv2.fillPoly(img, pts =[contours], color=(255,255,255))

        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

        idx = dist.argmax()
        rad = dist[idx//100, idx%100]
        point = np.array([idx//100, idx%100])

        point1 = (point - 50)/70
        rad1 = rad/70

        self.circle_center = point1
        self.max_radius = rad1


class ShapeCurl(Shape):

    def randomize(self):
        self.x_pixels = []
        self.y_pixels = []

        self.generate_part(self.radius, self.hole_radius)

        self.nb_pixels = len(self.x_pixels)

        self.x_pixels = np.array(self.x_pixels)
        self.y_pixels = np.array(self.y_pixels)

        ### random rotation
        self.rotate(np.random.rand() * np.pi * 2)

        self.subsample()
        
        # Este código no hace nada
        # if self.sym_rot > 1:
        #     self.rot_symmetrize(self.sym_rot)
        # if self.sym_flip:
        #     self.flip_symmetrize()

        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        ### resets to a squre
        self.x_pixels = self.x_pixels/(self.x_pixels.max() - self.x_pixels.min())
        self.y_pixels = self.y_pixels/(self.y_pixels.max() - self.y_pixels.min())

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        self.wh = (1, 1)


        self.transformations = []


# test
if __name__ == "__main__":
    shape = Shape()
    shape.rigid_transform(type='arrow_line', points=5, rotate=True)
    contour = shape.get_contour()
    plt.plot(contour[:, 0], contour[:, 1])
    plt.axis('equal')
    plt.show()