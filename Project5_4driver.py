
# % Project 5_4 driver
# % EECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import cv2 as cv
import numpy as np
import vtkmodules.vtkRenderingCore
from DisplayVolume import *


def Project5_4driver(frame, verts, faces,
                     cam_position, cam_projection_direction, cam_view_up, cam_view_angle):
    # creating render window to perform augmented reality
    renwin = vtk.vtkRenderWindow()
    ren = vtk.vtkRenderer()
    renwin.AddRenderer(ren)
    # make size same as image
    renwin.SetOffScreenRendering(1)
    renwin.SetSize(np.shape(frame)[1],np.shape(frame)[0])

    cam = ren.GetActiveCamera()

    # Adding tumor surface to renderer
    pnts = vtk.vtkPoints()
    for j,p in enumerate(verts):
        pnts.InsertPoint(j,p)

    cells = vtk.vtkCellArray()
    for j in range(len(faces)):
        vil = vtk.vtkIdList()
        for k in range(3):
            vil.InsertNextId(faces[j,k])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetPolys(cells)

    poly.BuildCells()
    poly.BuildLinks()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0,1,1)
    actor.GetProperty().SetOpacity(1)
    ren.AddActor(actor)

    # Move virtual camera to real camera position and pose
    cam.SetPosition(cam_position[0],cam_position[1],cam_position[2])
    cam.SetFocalPoint(cam_position[0]+cam_projection_direction[0],
                      cam_position[1]+cam_projection_direction[1],
                      cam_position[2]+cam_projection_direction[2])
    cam.SetViewUp(-cam_view_up[0],-cam_view_up[1],-cam_view_up[2])

    cam.SetViewAngle(cam_view_angle)

    ren.ResetCameraClippingRange()
    renwin.Render()

    # capture render to an image matrix
    windowToImageFilter = vtkmodules.vtkRenderingCore.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renwin)
    windowToImageFilter.SetInputBufferTypeToRGBA()
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()
    out = windowToImageFilter.GetOutput()

    # vtk has y axis up, opencv has it down, so flipping y axis and blending rendering with image for AR
    rng = out.GetExtent()[3]
    for i in range(np.shape(frame)[1]):
        for j in range(np.shape(frame)[0]):
            r = out.GetScalarComponentAsFloat(i,rng - j,0,0)
            g = out.GetScalarComponentAsFloat(i,rng - j,0,1)
            b = out.GetScalarComponentAsFloat(i,rng - j,0,2)
            if g>0:
                frame[j,i,0] = frame[j,i,0] // 4 * 3 + b // 4
                frame[j,i,1] = frame[j,i,0] // 4 * 3 + g // 4
                frame[j,i,2] = frame[j,i,0] // 4 * + r // 4

    # display image and wait for key-press to exit
    cv.imshow("Augmented Reality (press any key to exit)",frame)
    cv.waitKey(0)



