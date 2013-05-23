#/usr/bin/env python

'''
Video

Sistema deteccion de flash

ESC    - exit

'''

import numpy as np
import cv2
from time import clock
from numpy import pi, sin, cos
import Image
import math


class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg = None, **params):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv2.imread(bg, 1)
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv2.resize(self.bg, self.frame_size)
        self.noise = float(noise)

    def render(self, dst):
        pass

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        self.render(buf)

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv2.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv2.add(buf, noise, dtype=cv2.CV_8UC3)
        return True, buf

    def isOpened(self):
        return True


def create_capture(source = 0):
    source = str(source).strip()
    chunks = source.split(':')
    # hanlde drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv2.VideoCapture(source)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print 'Warning: unable to open video source: ', source
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

if __name__ == '__main__':
    import sys
    import getopt

    print __doc__

    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ 0 ]

    caps = map(create_capture, sources)
    shot_idx = 0
    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            imgs.append(img)
            cv2.imshow('capture %d' % i, img)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord(' '):
            for i, img in enumerate(imgs):
                fn = '%s/flash_%03d.bmp' % (shotdir, shot_idx)
                cv2.imwrite(fn, img)
                #print fn, 'saved'

    	#Escala		
	    im = Image.open(fn)
	    pix = im.load()
	    ancho, largo =im.size
       	    for x in range(0,ancho): #Ancho
	    	for y in range(0,largo): #Largo
	    	    (r,g,b) = pix[x,y]
	            promedio=(r+g+b)/3
	    	    pix[x,y] =(promedio,promedio,promedio)
	    im.save('escala%03d.bmp' % shot_idx)
	    cont = 0

	    #Binarizacion
	    im = Image.open('escala%03d.bmp' % shot_idx)
	    pix = im.load()
	    for x in range(0,ancho-1): #Ancho
	    	for y in range(0,largo-1): #Largo
	    		(r,g,b) = pix[x,y]
	    		promedio=(r+g+b)/3
	    		if promedio<=250:
	    			pix[x,y] =(0,0,0)
				cont = cont + 1
	    		else:
	    			pix[x,y] =(255,255,255)	    
	    im.save('binarizada%03d.bmp' % shot_idx)
	    if cont>1000:
		print "Flash Detectado"

	    #Mascara
	    im = Image.open('escala%03d.jpg' % shot_idx)
   	    pix = im.load()
	    dirX = ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])
	    dirY = ([1, 2, 1], [0, 0, 0], [-1, -2, -1])
	    for x in range(0,ancho-3): #Ancho
	    	for y in range(0,largo-3): #Largo
			sumDirX=0
			sumDirY=0

			for i in range(0,3):
				for j in range(0,3):	
					sumDirX +=(pix[x+i,y+j][0]*dirX[i][j])
					sumDirY +=(pix[x+i,y+j][0]*dirY[i][j])

			potX = pow(sumDirX, 2)
			potY = pow(sumDirY, 2)
			res = int(math.sqrt(potX+potY))
			if res > 255:
				res = 255
			if res < 0:
				res = 0
			pix[x,y] = (res, res, res)
	    im.save('convolucion.jpg')

		#Se guarda posicion de pixeles blancos
 	    cola=[]
	    cola2=[]
	    im = Image.open('binarizada%03d.bmp' % shot_idx)
	    pix=im.load()
	    for i in range(ancho):
	        for j in range(largo):
        	    (r,g,b)=im.getpixel((i,j))
        	    if ((r,g,b)==(255,255,255)):
        	        cola.append((i,j))
        	    else:
        	        cola2.append((i,j))

		#Dilatacion e impresion de deteccion en imagen original
	    im = Image.open(fn)
	    pix=im.load()#Cargarimagen
    	    x=0
	    while x<len(cola):
        	(i,j)=cola[x]
        	(r,g,b)=im.getpixel((i,j))
        
        	try:
        	    if(pix[i+1,j]):#Vecinos derecho
        	        pix[i+1,j]=(0,255,0)
        	        
        	except:
        	    pass
        	try:
        	    if(pix[i-1,j]):#Vecino izq
        	        pix[i-1,j]=(0,255,0)
        	        
        	except:
        	    pass
        	try:
        	    if(pix[i,j+1]):#Vecino arriba
        	        pix[i,j+1]=(0,255,0)
        	except:
        	    pass
        	try:
        	    if(pix[i,j-1]):#Vecino abajo
        	        pix[i,j-1]=(0,255,0)
        	    
        	except:
        	    pass
        	try:
        	    if(pix[i+1,j+1]):#esq derecha
        	        pix[i+1,j+1]=(0,255,0)
        	        
        	except:
        	    pass
        	try:
        	    if(pix[i-1,j+1]):#esq izq
        	        pix[i-1,j+1]=(0,255,0)
        	        
        	except:
        	    pass
        	try:
        	    if(pix[i+1,j-1]):#esq der abajo
        	        pix[i+1,j-1]=(0,255,0)
        	except:
        	    pass
        	try:
        	    if(pix[i-1,j-1]):#esq izq abajo
        	        pix[i-1,j-1]=(0,255,0)
        	except:
        	    pass
	
        	x+=1 			
	    im.save('deteccion%03d.bmp' % shot_idx)
	
	
	    shot_idx += 1
    cv2.destroyAllWindows()
