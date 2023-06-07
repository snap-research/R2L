
"""
download and install:
-python-2.3.5.exe
-pygame-1.6.win32-py2.3.exe
-PyOpenGL-2.0.2.01.py2.3-numpy23.exe (the only one i know which have extension required)
-cgkit-1.1.0.win32-py2.3.exe (mat4)

double click on demoShadows.py
"""

#import openGL
import time

import imageio
import numpy as np
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *

#import pygame..
import pygame
import json
from pygame.locals import *

#
from shadowsDef import *
from math import sin, cos
from array import array
from struct import unpack
from objloader import OBJ

window_x=800.0
window_y=600.0
INCR = -0.25
INTERP_TIME = 0.25

def Set_screen():
    'init pygame-video'
    pygame.display.init()
    pygame.display.set_mode((int(window_x),int(window_y)), HWSURFACE|OPENGL|DOUBLEBUF,)
    pygame.display.set_caption('shadow DEMO','shadow DEMO')


def initGL():
    'init openGL'
    glClearColor(0.6,0.6,0.6,1.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)                                    

    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    
    glShadeModel(GL_SMOOTH)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

def cameraLoop(position):
    'camera at (22,28,-18) looking at (0,0,0)'
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()
    gluPerspective(65.0, (window_x/window_y), 0.1, 200.0)

    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()
    gluLookAt(  position[0], position[1], position[2],
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0 )

    glViewport(0, 0, int(window_x), int(window_y))

def lightLoop(position):
    'light which change position every frame'
    
    position = list(position)
    position.append(1.0)

    glPushMatrix()
    glDisable(GL_LIGHTING)
    glPointSize(5.0)
    glBegin(GL_POINTS)
    glColor4f(1.0,1.0,1.0,1.0)
    glVertex4fv(position)
    glEnd()
    glPopMatrix() 

    glLightfv(GL_LIGHT0, GL_DIFFUSE, ( 1.0,1.0,1.0,1.0 ))   # Setup The Diffuse Light 
    glLightfv(GL_LIGHT0, GL_SPECULAR, ( 0.6,0.6,0.6,1.0 ))  # Setup The Specular Light
    glLightfv(GL_LIGHT0, GL_AMBIENT, ( 0.1,0.1,0.1,1.0 ))   # Setup The Ambient Light 
    glLightfv(GL_LIGHT0, GL_POSITION, position)             # Position of The Light  

    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glEnable(GL_LIGHT0)
    
    glEnable(GL_LIGHTING)
    
def listFromFile(_file):
    'open _file and create a glList'
    all_vtx_data = array('f')
    mat_attr = array('f')

    f = open(_file, 'rb')
    all_vtx_data.fromfile(f, int(unpack('f',f.read(4))[0])*8 )
    mat_attr.fromfile(f, 11)
    f.close()

    trasparency = mat_attr[3]
    shine = mat_attr[4] 

    matAmb = [ 0.05, 0.05, 0.05, 1.0 ]
    matDiff = [mat_attr[0], mat_attr[1], mat_attr[2], trasparency ]            
    matSpec = [mat_attr[5], mat_attr[6], mat_attr[7], trasparency ]
    matEmis = [mat_attr[8], mat_attr[9], mat_attr[10], trasparency ]

    id_list = glGenLists(1)
    glNewList (id_list, GL_COMPILE)
    # check if opengl generated depth map for cube can be extracted and reinserted.
    glMaterialfv(GL_FRONT, GL_DIFFUSE, matDiff)
    glMaterialfv(GL_FRONT, GL_AMBIENT, matAmb)
    glMaterialfv(GL_FRONT, GL_SPECULAR, matSpec)
    glMaterialfv(GL_FRONT, GL_EMISSION, matEmis)
    glMaterialf(GL_FRONT, GL_SHININESS, shine)

    glBegin(GL_TRIANGLES)
    for i in range(0, len(all_vtx_data), 8):
        glNormal3fv((all_vtx_data[i+0], all_vtx_data[i+1], all_vtx_data[i+2]))
        glVertex3fv((all_vtx_data[i+5], all_vtx_data[i+6], all_vtx_data[i+7]))      
    glEnd()     

    glEndList()
    
    return id_list

def renderFloor(listID):
    'render floor'
    glPushMatrix()
    glScalef(2., 1., 2.)
    glTranslate(0.0, -5.0, 0.0)
    glCallList(listID)
    glPopMatrix()
    
def renderObj(listID, rot):
    'render (really easy) animated object'
    glPushMatrix()
    # glRotate(rot*0.5, 0.0, 1.0, 0.0)
    glTranslate(0.0, 0.0, 0.0)
    # glRotate(-rot, 1.0, -1.0, 1.0)
    glCallList(listID)
    glPopMatrix()

def read_shadow_map(filename):
    data = imageio.v2.imread(filename)
    data = (np.array(data) / 255.).astype(np.float32)
    data[data == 0] = 1 / 255.
    data = 1 / data
    # data = 255 * data
    data[data == data.max()] = 255.
    if data.shape != (400, 400):
        data = data[:, :, 0]
    data = np.flip(data, 0)
    return data


def load_blender_data(basedir, testskip=1):
    splits = ['test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)),
                  'r') as fp:
            metas[s] = json.load(fp)

    all_poses = []
    for s in splits:
        meta = metas[s]
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            poses.append(np.array(frame['transform_matrix']))
        poses = np.array(poses).astype(np.float32)
        all_poses.append(poses)

    poses = np.concatenate(all_poses, 0)
    return poses


#main
def main():
    'main loop'
    pause = False

    incrObjRotY = 0.0
    amply = 10.0

    Set_screen()
    initGL()

    # Load transforms
    poses = load_blender_data("shadows", 8)

    # lists
    floorID = listFromFile('./floor.msh')
    obj = OBJ('cube.obj')
    print("Object loading done")

    i = 0
    pose = poses[i]
    prev_epoch = time.time()
    cameraPosition = (0, 12, -18)
    # Load shadow map
    shadow_map_data = read_shadow_map(f"shadows/00{i}.png")
    # shadow Map
    textureMapID = CreateTextureShadow(shadow_map_data)
    lightPosition = [pose[0][3], pose[2][3], -pose[1][3]]

    while True:
        now = time.time()
        if now - prev_epoch > INTERP_TIME and not pause:
            prev_epoch = now
            i += 1
            if i > 24:
                i = 0
            pose = poses[i][:3, :4]
            # Load shadow map
            if i < 10:
                shadow_map_data = read_shadow_map(f"shadows/00{i}.png")
            else:
                shadow_map_data = read_shadow_map(f"shadows/0{i}.png")

            # shadow Map
            textureMapID = CreateTextureShadow(shadow_map_data)

            lightPosition = [pose[0][3], pose[2][3], -pose[1][3]]

        # just to animate things..
        cameraPosition = (sin(incrObjRotY*.01)*amply,20.0,cos(incrObjRotY*.01)*amply)
        # lightPosition = (1, 20, 1)
        if not pause:
            incrObjRotY += INCR

        # press 'q' or ESCAPE to quit..
        event = pygame.event.poll()
        if event.type == KEYUP and (event.key == K_ESCAPE or event.key == K_q):
            quit(0)
        # press 'p' to pause anims..
        if event.type == KEYUP and event.key == K_p:
            pause = not pause

        # render obj(s) casting shadows
        textureMatrix = CreateShadowBefore(position=lightPosition)
        renderObj(obj.gl_list, 0)
        CreateShadowAfter(textureMapID)

        # render camera
        cameraLoop(cameraPosition)
        # set glLight
        lightLoop( position=lightPosition )

        # render all-----------------------------------
        renderObj(obj.gl_list, 0)
        renderFloor(floorID)
        # render all-----------------------------------

        # render obj(s) where shadows cast
        RenderShadowCompareBefore(textureMapID, textureMatrix)
        #renderObj(objID,incrObjRotY)
        renderFloor(floorID)
        RenderShadowCompareAfter()

        # flip
        pygame.display.flip()
        

#starts demo..        
if __name__ == "__main__":
    main()