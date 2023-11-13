# % Class to create interactive 3D image/mask viewer
# % EECE 8395: Engineering for Surgery
# % Fall 2023
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
#

# % Example usage:
# %   >> d = DisplayVolume()
# %   >> d.SetImage(mr, voxsz)
# %   >> d.Display()
# %       % Displays the 3D image in image np 3D array mr with voxel size voxsz
# %
# %   >> d.SetImage(mr,voxsz, contrast=1000, level=0)
# %   >> d.Display()
# %       % Displays the 3D image in image struct mr and adjusts the
# %       intensity contrast and window level
# %
# %   >> d.Update(direction=0,slc=20)
# %   >> d.Display()
# %       % For a currently displayed 3D image, changes the sagittal view (0) to
# %       slice 20
# %
# %   >> d.SetImage(mr, voxsz)
# %   >> d.AddMask(segmsk,color=[0,1,1],opacity=0.5, label = 'Mysegmentation')
# %   >> d.Display()
# %       % Displays the 3D image in  mr then overlays aqua
# %       colored contours of a segmentation mask in the segmsk struct on the
# %       3D views and displays a 3D isosurface of the mask with 0.5 opacity
# %       in the 3D viewer window


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from skimage import measure
import matplotlib
matplotlib.use('TkAgg')

usemayavi=1
if usemayavi:
    from mayavi import mlab
else:
    import vtk
class imagevolume:
    def __init__(self,data=0,voxsz=[1,1,1]):
        self.data = data
        self.voxsz = voxsz

class contourclass:
    def __init__(self, data):
        self.data = data

class surface:
    def __init__(self):
        self.verts = None
        self.faces = None
        self.normals = None

class object:
    def __init__(self, type, data, color=[1.0,0.0,0.0], opacity = 1.0):
        self.type = type
        self.data = data
        self.color = color
        self.opacity = opacity
        self.ms = None

class mask:
    def __init__(self, data, voxsz=[1.0,1.0,1.0], color=[1.0,0.0,0.0], label=''):
        self.data = data
        self.voxsz = voxsz
        self.color = color
        self.label = label
        self.cntrs = np.zeros([3,max(np.shape(data))],dtype=contourclass)
        self.verts = None
        self.faces = None

    def UpdateContours(self):
        dim = np.shape(self.data)
        X,Y = np.meshgrid(np.linspace(0,dim[0]-1,dim[0]), np.linspace(0,dim[1]-1,dim[1]))
        for i in range(np.shape(self.data)[2]):
            if np.min(self.data[:,:,i])<0.5 and np.max(self.data[:,:,i]) > 0.5:
                cntr = plt.contour(X,Y,np.transpose(self.data[:,:,i]),levels=[0.5])
                self.cntrs[0][i] = contourclass(cntr.allsegs[0])
        X, Y = np.meshgrid(np.linspace(0, dim[0] - 1, dim[0]), np.linspace(0, dim[2] - 1, dim[2]))
        for i in range(np.shape(self.data)[1]):
            if np.min(self.data[:, i, :]) < 0.5 and np.max(self.data[:, i, :]) > 0.5:
                cntr = plt.contour(X, Y, np.transpose(np.squeeze(self.data[:, i, :])), levels=[0.5])
                self.cntrs[1][i] = contourclass(cntr.allsegs[0])
        X, Y = np.meshgrid(np.linspace(0, dim[1] - 1, dim[1]), np.linspace(0, dim[2] - 1, dim[2]))
        for i in range(np.shape(self.data)[0]):
            if np.min(self.data[i, :, :]) < 0.5 and np.max(self.data[i, :, :]) > 0.5:
                cntr = plt.contour(X, Y, np.transpose(np.squeeze(self.data[i, :, :])), levels=[0.5])
                self.cntrs[2][i] = contourclass(cntr.allsegs[0])

        self.verts,self.faces,_,_ = measure.marching_cubes(self.data,0.5, spacing=self.voxsz)
        # self.verts[:,0] *= self.voxsz[0]
        # self.verts[:,1] *= self.voxsz[1]
        # self.verts[:,2] *= self.voxsz[2]

class DisplayVolume:
    def __init__(self):

        self.img = 0
        # self.msks = []
        self.objs = []

        self.slc = [0,0,0]
        self.contrast = 1
        self.level = 0
        plt.ion()
        self.fig = plt.figure()
        self.fig.suptitle('Display Volume (Press Esc to quit)', fontsize=16)
        self.ax = np.zeros([2,2], dtype=plt.Axes)
        self.ax[0,0] = self.fig.add_subplot(2,2,1)
        self.ax[0,1] = self.fig.add_subplot(2,2,2)
        self.ax[1,0] = self.fig.add_subplot(2,2,3)
        plt.axes(self.ax[0,0])

        self.quit = False
        binding_id2 = plt.connect('button_press_event', self.on_mouse_click)
        binding_id3 = plt.connect('key_press_event', self.on_key_press)
        if usemayavi:
            self.f3d = mlab.figure(bgcolor=(1,1,1))
        else:
            self.renwin = vtk.vtkRenderWindow()
            self.ren = vtk.vtkRenderer()
            self.renwin.AddRenderer(self.ren)
            self.inter = vtk.vtkRenderWindowInteractor()
            self.inter.SetRenderWindow(self.renwin)
            self.inter.Initialize()
            self.renwin.Render()
            self.inter.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())


    def on_mouse_click(self, event):
        if event.dblclick:
            if event.button is MouseButton.LEFT:
                direction = -1
                for i in range(0,3):
                    if event.inaxes == self.ax[i//2, i%2]:
                        direction = i
                if direction == 2:
                    pnt = [event.xdata, event.ydata, self.slc[2]]
                elif direction ==1:
                    pnt = [event.xdata, self.slc[1], event.ydata]
                elif direction == 0:
                    pnt = [self.slc[0], event.xdata, event.ydata]
                else:
                    return
                for i in range(0, 3):
                    if pnt[i]<0:
                        pnt[i]=0
                    elif pnt[i]>np.shape(self.img)[i]-1:
                        pnt[i] = np.shape(self.img)[i]-1
                    self.slc[i] = round(pnt[i])
                    self.Update(i)
                for i in range(0,3):
                    xlim = self.ax[i//2,i%2].get_xlim()
                    ylim = self.ax[i//2,i%2].get_ylim()
                    xrng = xlim[1] - xlim[0]
                    yrng = ylim[1] - ylim[0]
                    if i==0:
                        x = pnt[1]
                        y = pnt[2]
                    elif i==1:
                        x = pnt[0]
                        y = pnt[2]
                    else:
                        x = pnt[0]
                        y = pnt[1]
                    self.ax[i//2,i%2].set_xlim(x-xrng/2,x+xrng/2)
                    self.ax[i//2,i%2].set_ylim(y-yrng/2,y+yrng/2)
                    plt.axes(self.ax[i//2,i%2])
                    plt.plot([x, x], [y + 0.02 * yrng, y - 0.02 * yrng], 'r')
                    plt.plot([x + 0.02 * xrng, x - 0.02 * xrng], [y, y], 'r')


            if event.button is MouseButton.RIGHT:
                if event.inaxes == self.ax[1,0]:
                    self.ax[1,0].set_xlim(left=0, right=np.shape(self.img)[0] - 1)
                    self.ax[1,0].set_ylim(bottom=np.shape(self.img)[1] - 1,top=0)
                elif event.inaxes == self.ax[0,1]:
                    self.ax[0,1].set_xlim(left=0, right=np.shape(self.img)[0] - 1)
                    self.ax[0,1].set_ylim(top=np.shape(self.img)[2] - 1,bottom=0)
                elif event.inaxes == self.ax[0,0]:
                    self.ax[0,0].set_xlim(left=0, right=np.shape(self.img)[1] - 1)
                    self.ax[0,0].set_ylim(top=np.shape(self.img)[2] - 1,bottom=0)



    def on_key_press(self, event):
        if event.key == 'escape':
            self.quit = True
        for i in range(0,3):
            if event.inaxes == self.ax[i//2, i%2]:
                if (event.key == 'a' or event.key == 'up') and self.slc[i] < np.shape(self.img)[i]-1:
                    self.slc[i] += 1
                if (event.key == 'z' or event.key == 'down') and self.slc[i] > 0 :
                    self.slc[i] -= 1
                self.Update(i)
                break
        if event.key == 'g':
            self.level += .1*self.contrast
            self.Update()
        if event.key == 'v':
            self.level -= 0.1*self.contrast
            self.Update()
        if event.key == 'd':
            self.contrast *= 1.1
            self.Update()
        if event.key == 'c':
            self.contrast *= 0.9
            self.Update()


    def SetImage(self, img, voxsz, contrast=1000, level=0, interpolation='bilinear', autocontrast=True):
        self.img = img
        self.voxsz = voxsz
        self.slc = np.array(np.shape(img.data), dtype=int) //2
        self.contrast=contrast
        self.level = level
        self.interpolation = interpolation
        if autocontrast:
            self.AutoContrast()
        self.Update()

    def AutoContrast(self):
        mn = np.amin(self.img)
        mx = np.amax(self.img)
        bns = np.linspace(mn,mx,256)
        h,_ = np.histogram(self.img.ravel(), bins=bns)
        h = h.astype(np.float32)/np.sum(h)
        mini = 0
        tot = h[0]
        while tot<0.1:
            mini+=1
            tot += h[mini]

        maxi = mini
        while tot<0.99:
            maxi +=1
            tot += h[maxi]

        self.contrast = (maxi - mini) * (mx - mn) / 256.0
        self.level = 0.5 * (maxi + mini) * (mx - mn) / 256.0 + mn


    def AddMask(self, msk, color = [1.0,0.0,0.0], opacity = 1.0, label=''):
        mskobj = mask(msk, self.voxsz, color,label)
        mskobj.UpdateContours()
        # surf = surface()
        # surf.verts, surf.faces, surf.normals, values = measure.marching_cubes(msk,0.5)
        # surf.verts[:, 0] *= self.voxsz[0]
        # surf.verts[:, 1] *= self.voxsz[1]
        # surf.verts[:, 2] *= self.voxsz[2]
        # obj = object(0,surf, color=color, opacity = opacity)
        obj = object(1, mskobj, color=color, opacity = opacity)
        self.objs.append(obj)
        # self.msks.append(mskobj)

        self.Update()


    def Update(self,direction = -1,slc = -1,level = float("nan"),contrast = float("nan"),resize = -1):
        if (not np.isnan(level)):
            self.level = level
        if (not np.isnan(contrast)):
            self.contrast = contrast
        if (resize != -1):
            self.resize = resize
        if direction == -1:
            for i in range(4):
                self.Update(direction = i)
        else:
            if slc >= 0:
                self.slc[direction] = slc
            if direction <3:
                plt.figure(self.fig)
                plt.axes(self.ax[direction // 2, direction % 2])
                xlim = self.ax[direction//2, direction%2].get_xlim()
                ylim = self.ax[direction//2, direction%2].get_ylim()
                plt.cla()
                if direction == 2:
                    self.ax[direction//2, direction%2].imshow(np.transpose(self.img[:, :, self.slc[direction]]),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2,
                                                                            vmax=self.level + self.contrast/2)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[1]/self.voxsz[0])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0,
                                                                                         right=np.shape(self.img)[0] - 1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=np.shape(self.img)[1] - 1,
                                                                                         top=0)
                    vstr = 'Axial view'
                    z = 'z'
                    for i in range(len(self.objs)):
                        if self.objs[i].type==1 and self.objs[i].data.cntrs[0][self.slc[2]]:
                            for j in range(len(self.objs[i].data.cntrs[0][self.slc[2]].data)):
                                plt.plot(self.objs[i].data.cntrs[0][self.slc[2]].data[j][0],
                                         self.objs[i].data.cntrs[0][self.slc[2]].data[j][1], color=self.objs[i].color)
                elif direction == 1:
                    self.ax[direction//2, direction%2].imshow(np.transpose(np.squeeze(self.img[:, self.slc[direction], :])),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2, vmax=self.level + self.contrast/2)
                    plt.xlabel('x')
                    plt.ylabel('z')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[2] / self.voxsz[0])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0,
                                                                                         right=np.shape(self.img)[0] - 1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=0,
                                                                                         top=np.shape(self.img)[2] - 1)
                    vstr = 'Coronal view'
                    z = 'y'
                    for i in range(len(self.objs)):
                        if self.objs[i].type==1 and self.objs[i].data.cntrs[1][self.slc[1]]:
                            for j in range(len(self.objs[i].data.cntrs[1][self.slc[1]].data)):
                                plt.plot(self.objs[i].data.cntrs[1][self.slc[1]].data[j][0],
                                         self.objs[i].data.cntrs[1][self.slc[1]].data[j][1], color=self.objs[i].color)
                elif direction == 0:
                    self.ax[direction//2, direction%2].imshow(np.transpose(np.squeeze(self.img[self.slc[direction], :, :])),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2, vmax=self.level + self.contrast/2)
                    plt.xlabel('y')
                    plt.ylabel('z')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[2] / self.voxsz[1])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0, right=np.shape(self.img)[1]-1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=0, top=np.shape(self.img)[2]-1)
                    vstr = f'Sagittal view Contrast = {self.contrast:.1f} Level = {self.level:.1f}'
                    z = 'x'
                    for i in range(0,np.size(self.objs)):
                        if self.objs[i].type == 1:
                            max_index = len(self.objs[i].data.cntrs[2]) - 1
                            if self.slc[0] > max_index:
                                print(f"Warning: Attempted to access index {self.slc[0]} which is out of bounds. Resetting to {max_index}.")
                                self.slc[0] = max_index
                                for j in range(len(self.objs[i].data.cntrs[2][self.slc[0]].data)):
                                    plt.plot(self.objs[i].data.cntrs[2][self.slc[0]].data[j][0],
                                            self.objs[i].data.cntrs[2][self.slc[0]].data[j][1], color=self.objs[i].color)


                plt.title(f'{vstr}: Slice {z} = {self.slc[direction]}')
            elif direction == 3:
                if usemayavi:
                    mlab.figure(self.f3d)
                else:
                    self.renwin.MakeCurrent()
                    self.ren.RemoveAllViewProps()
                mns = 1e8*np.ones(3)
                mxs = -1e8*np.ones(3)
                for i in range(0, np.size(self.objs)):
                    if self.objs[i].type==0 or self.objs[i].type==1:
                        for j in range(0,3):
                            mn = np.min(self.objs[i].data.verts[:,j])
                            if mn < mns[j]:
                                mns[j] = mn
                            mx = np.max(self.objs[i].data.verts[:, j])
                            if mx > mxs[j]:
                                mxs[j] = mx
                        if usemayavi:
                            if self.objs[i].ms == None:
                                self.objs[i].ms = mlab.triangular_mesh(self.objs[i].data.verts[:,0],
                                                 self.objs[i].data.verts[:,1],
                                                 self.objs[i].data.verts[:,2],
                                                 self.objs[i].data.faces,
                                                 color=(self.objs[i].color[0], self.objs[i].color[1] , self.objs[i].color[2]),
                                                 opacity=self.objs[i].opacity)
                            else:
                                self.objs[i].ms.mlab_source.set(x=self.objs[i].data.verts[:,0],
                                         y=self.objs[i].data.verts[:,1],
                                         z=self.objs[i].data.verts[:,2])
                        else:
                            # if self.objs[i].ms == None:
                            pnts = vtk.vtkPoints()
                            for j,p in enumerate(self.objs[i].data.verts):
                                pnts.InsertPoint(j,p)

                            cells = vtk.vtkCellArray()
                            for j in range(len(self.objs[i].data.faces)):
                                vil = vtk.vtkIdList()
                                for k in range(3):
                                    vil.InsertNextId(self.objs[i].data.faces[j,k])
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
                            actor.GetProperty().SetColor(self.objs[i].color[0] ,self.objs[i].color[1] ,self.objs[i].color[2] )
                            actor.GetProperty().SetOpacity(self.objs[i].opacity)
                            self.ren.AddActor(actor)
                            mn = actor.GetCenter()
                            self.ren.GetActiveCamera().SetFocalPoint(mn[0], mn[1], mn[2])
                if usemayavi:
                    self.f3d.scene.render()

    def Display(self, blocking=True):
        self.Update()
        if blocking:
            while (self.quit == False):
                if usemayavi:
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.start_event_loop(0.3)
                else:
                    self.ren.ResetCameraClippingRange()
                    self.renwin.Render()
                    self.inter.ProcessEvents()

    def Close(self):
        plt.close(self.fig)
        if usemayavi:
            mlab.close(self.f3d)
        else:
            del self.renwin, self.ren, self.inter