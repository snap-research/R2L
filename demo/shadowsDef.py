
#read demoShadows.py

#import openGL
import imageio
import numpy as np
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *

#import extension for shadows..
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL.ARB.depth_texture import *

from cgtypes import mat4

shadowMapSize = 400
spotFOV = 50.0

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def CreateTextureShadow(shadow_map_data):
    'before loop, crate a texture obj to store shadow map'
    id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, id)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowMapSize, shadowMapSize, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                 shadow_map_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP)

    return id
                
def CreateShadowBefore(position):
    """every frame render objs which casts shadows from light point of view
    position --> light position
    return --> projection matrix of the shadow map created"""
    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()
    gluPerspective(spotFOV, 1.0, 2., 6.)
    lightProjectionMatrix = mat4(glGetFloatv(GL_PROJECTION_MATRIX))

    glMatrixMode(GL_MODELVIEW)
    
    glLoadIdentity()
    gluLookAt(  position[0],
                position[1],
                position[2],
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0 )
    lightViewMatrix = mat4(glGetFloatv(GL_MODELVIEW_MATRIX))

    #Use viewport the same size as the shadow map
    glViewport(0, 0, shadowMapSize, shadowMapSize)

    #Draw back faces into the shadow map
    glCullFace(GL_FRONT)

    #Disable lighting, texture, use flat shading for speed
    glShadeModel(GL_FLAT)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)

    #Disable color writes
    glColorMask(0, 0, 0, 0)

    glPolygonOffset(0.5, 0.5)
    glEnable(GL_POLYGON_OFFSET_FILL) 

    glClear( GL_DEPTH_BUFFER_BIT )

    #eval projection matrix
    biasMatrix = mat4 ( 0.5, 0.0, 0.0, 0.5,
                        0.0, 0.5, 0.0, 0.5,
                        0.0, 0.0, 0.5, 0.5,
                        0.0, 0.0, 0.0, 1.0 )
    return biasMatrix*lightProjectionMatrix*lightViewMatrix

def CreateShadowAfter(shadowMapID):
    'write texture into texture obj and reset gl params'
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, shadowMapID)
    # glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, shadowMapSize, shadowMapSize)
    # image = glGetTexImage(target=GL_TEXTURE_2D, level=0, format=GL_DEPTH_COMPONENT, type=GL_FLOAT)
    # imageio.imwrite("shadow.png", to8b(image))

    glCullFace(GL_BACK)
    glShadeModel(GL_SMOOTH)
    glColorMask(1, 1, 1, 1)

    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_TEXTURE_2D)

def RenderShadowCompareBefore(shadowMapID, textureMatrix):
    'eval where draw shadows using ARB extension'
    glEnable(GL_TEXTURE_2D)          
    glBindTexture(GL_TEXTURE_2D, shadowMapID)

    glEnable(GL_TEXTURE_GEN_S)
    glEnable(GL_TEXTURE_GEN_T)
    glEnable(GL_TEXTURE_GEN_R)
    glEnable(GL_TEXTURE_GEN_Q)
    
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_S, GL_EYE_PLANE, np.asarray(textureMatrix.getRow(0)))

    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_T, GL_EYE_PLANE, np.asarray(textureMatrix.getRow(1)))

    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_R, GL_EYE_PLANE, np.asarray(textureMatrix.getRow(2)))

    glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
    glTexGenfv(GL_Q, GL_EYE_PLANE, np.asarray(textureMatrix.getRow(3)))

    #Enable shadow comparison
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB)

    #Shadow comparison should be true (in shadow) if r>texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_GREATER)

    #Shadow comparison should generate an INTENSITY result
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY)

    #Set alpha test to discard false comparisons
    glAlphaFunc(GL_EQUAL, 1.0)
    glEnable(GL_ALPHA_TEST)

    glLightfv(GL_LIGHT0, GL_DIFFUSE, ( 0.6,0.6,0.6,1.0 ))   # Diffuse Light for shadows

def RenderShadowCompareAfter():
    'reset gl params after comparison'
    glDisable(GL_TEXTURE_2D)

    glDisable(GL_TEXTURE_GEN_S)
    glDisable(GL_TEXTURE_GEN_T)
    glDisable(GL_TEXTURE_GEN_R)
    glDisable(GL_TEXTURE_GEN_Q)

    glDisable(GL_ALPHA_TEST) 


