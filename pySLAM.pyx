cimport slam_system as sls
cimport numpy as np
import numpy as np
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
import time
import cython
cdef extern from "Python.h":
    void PyEval_InitThreads()

def init_threads():
    PyEval_InitThreads()

cdef class SLAM_Frame:
    cdef double[::1] cam_to_world
    cdef float fx,fy,cx,cy,scale,timestamp
    cdef int frame_id,w,h

    def __cinit__(self,
                 int frame_id,
                 int w,
                 int h,
                 float fx,
                 float fy,
                 float cx,
                 float cy,
                 float scale,
                 float timestamp,
                 np.ndarray[np.double_t] cam_to_world ):
        pass

    def __init__(self,
                 int frame_id,
                 int w,
                 int h,
                 float fx,
                 float fy,
                 float cx,
                 float cy,
                 float scale,
                 float timestamp,
                 np.ndarray[np.double_t] cam_to_world ):
        self.frame_id =  frame_id
        self.w =  w
        self.h =  h
        self.fx =  fx
        self.fy =  fy
        self.cx =  cx
        self.cy =  cy
        self.scale =  scale
        self.timestamp =  timestamp
        self.cam_to_world = cam_to_world

    def __str__(self):
        return "SLAM Frame id:%s,ts:%s"%(self.frame_id,self.timestamp)

cdef class SLAM_K_Frame(SLAM_Frame):
    cdef  float[:,::1] point_cloud

    def __str__(self):
        cdef int pc_count 
        if self.point_cloud is None:
            pc_count = 0
        else:
            pc_count = len(self.point_cloud)
        return "SLAM Key Frame id:%s,ts:%s, Point Count:%s"%(self.frame_id,self.timestamp,pc_count)

cdef cppclass cyOutput3DWrapper(sls.Output3DWrapper):
    PyObject* callback
    __init__(object callback):  # constructor. "this" argument is implicit.
        Py_INCREF(callback)
        this.callback = <PyObject*>callback

    __dealloc__():  # destructor
        Py_DECREF(<object>this.callback)

    @cython.boundscheck(False)
    void publishKeyframe(sls.Frame* f) with gil:
        cdef int plvl = 0
        cdef int x,y,w,h
        cdef float fx,fy,cx,cy,fxi,fyi,cxi,cyi
        cdef sls.shared_lock[sls.shared_mutex] lock = f.getActiveLock()
        
        w = f.width(plvl)
        h = f.height(plvl)
        fx = f.fx(plvl)
        fy = f.fy(plvl)
        cx = f.cx(plvl)
        cy = f.cy(plvl)

        fxi = 1/fx
        fyi = 1/fy
        cxi = -cx / fx
        cyi = -cy / fy

        #pose
        cdef sls.Sim3 pose = f.getScaledCamToWorld(0)
        cdef const double * t_mat = pose.matrix().data()
        cdef np.ndarray[np.double_t,ndim=1] trans_mat = np.empty(4*4,np.double)
        cdef float scale = pose.scale()
        for x in range(16):
            trans_mat[x] = t_mat[x]

        cdef SLAM_K_Frame frame = SLAM_K_Frame(  f.id(),
                                w,h,
                                fx,fy,cx,cy,
                                scale,
                                f.timestamp(),
                                trans_mat)


        #depth map: 
        cdef const float * idepth = f.idepth(plvl)
        cdef const float * idepthVar = f.idepthVar(plvl)
        cdef const float * img = f.image(plvl)
        cdef float depth
        cdef float depth4 
        cdef float scaledTH = 10**-3.0 #log10 of threshold on point's variance, in the respective keyframe's scale. min: -10.0, default: -3.0, max: 1.0
        cdef float absTH = 10**-1.0 #log10 of threshold on point's variance, in absolute scale. min: -10.0, default: -1.0, max: 1.0
        cdef np.ndarray[np.float32_t,ndim=2] points = np.empty((w*h,3+4),np.float32)

        cdef int no_points = 0

        for y in range(1,h-1):
            for x in range(1,w-1):
                if(idepth[x+y*w] <= 0):
                    continue
                depth = 1. / idepth[x+y*w]
                depth4 = depth*depth 
                depth4*= depth4
                if idepthVar[x+y*w] * depth4 > scaledTH:
                    continue
                if idepthVar[x+y*w] * depth4 * scale*scale > absTH:
                    continue

                no_points +=1
                #xyz
                points[no_points,0]= (x*fxi + cxi) * depth
                points[no_points,1]= (y*fyi + cyi) * depth
                points[no_points,2]= depth
                #rgba
                points[no_points,3]= img[x+y*w]
                points[no_points,4]= img[x+y*w]
                points[no_points,5]= img[x+y*w]
                points[no_points,6]= 100
        if no_points:
            points.resize((no_points,3+4),refcheck=False)
            frame.point_cloud = points
        else:
            frame.point_cloud = None

        # bool paramsStillGood = my_scaledTH == scaledDepthVarTH &&
        #             my_absTH == absDepthVarTH &&
        #             my_scale*1.2 > camToWorld.scale() &&
        #             my_scale < camToWorld.scale()*1.2 &&
        #             my_minNearSupport == minNearSupport &&
        #             my_sparsifyFactor == sparsifyFactor;


        #             if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;

        #             if(my_minNearSupport > 1)
        #             {
        #                 int nearSupport = 0;
        #                 for(int dx=-1;dx<2;dx++)
        #                     for(int dy=-1;dy<2;dy++)
        #                     {
        #                         int idx = x+dx+(y+dy)*width;
        #                         if(originalInput[idx].idepth > 0)
        #                         {
        #                             float diff = originalInput[idx].idepth - 1.0f / depth;
        #                             if(diff*diff < 2*originalInput[x+y*width].idepth_var)
        #                                 nearSupport++;
        #                         }
        #                     }

        #                 if(nearSupport < my_minNearSupport)


        #     // create new ones, static
        #     vertexBufferId=0;
        #     glGenBuffers(1, &vertexBufferId);
        #     glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);         // for vertex coordinates
        #     glBufferData(GL_ARRAY_BUFFER, sizeof(MyVertex) * vertexBufferNumPoints, tmpBuffer, GL_STATIC_DRAW);
        #     vertexBufferIdValid = true;

        (<object>this.callback)(frame)

    void publishTrackedFrame(sls.Frame* f) with gil:
        cdef int plvl = 0
        cdef int x,y,w,h
        cdef float fx,fy,cx,cy,fxi,fyi,cxi,cyi
        cdef sls.shared_lock[sls.shared_mutex] lock = f.getActiveLock()
        
        w = f.width(plvl)
        h = f.height(plvl)
        fx = f.fx(plvl)
        fy = f.fy(plvl)
        cx = f.cx(plvl)
        cy = f.cy(plvl)

        fxi = 1/fx
        fyi = 1/fy
        cxi = -cx / fx
        cyi = -cy / fy

        #pose
        cdef sls.Sim3 pose = f.getScaledCamToWorld(0)
        cdef const double * t_mat = pose.matrix().data()
        cdef np.ndarray[np.double_t,ndim=1] trans_mat = np.empty(4*4,np.double)
        cdef float scale = pose.scale()
        for x in range(16):
            trans_mat[x] = t_mat[x]

        cdef SLAM_Frame frame = SLAM_Frame(  f.id(),
                                w,h,
                                fx,fy,cx,cy,
                                scale,
                                f.timestamp(),
                                trans_mat)

        (<object>this.callback)(frame)


    void publishKeyframeGraph(sls.KeyFrameGraph* graph) with gil:
        pass
        # print 'Graph'

    void publishDebugInfo(sls.Matrix201f data) with gil:
        print 'DebugInfo'


cdef class Slam_Context:
    cdef sls.SlamSystem *thisptr
    cdef cyOutput3DWrapper * output_wrapper
    def __cinit__(self, int w, int h,float[::1] K, enableSLAM=True):
        cdef sls.Matrix3f _K
        cdef float * K_d = _K.data()
        self.set_settings()

        for x in range(9):
            K_d[x] = K[x]
        self.thisptr = new sls.SlamSystem(w, h, _K, enableSLAM)


        
    def __init__(self, int w, int h, enableSLAM=True):
        pass
             
    def __dealloc__(self):
        del self.thisptr
        
    def setVisualization(self,callback):
        cdef cyOutput3DWrapper* output_wrapper = new cyOutput3DWrapper(callback)
        self.output_wrapper = output_wrapper
        self.thisptr.setVisualization(output_wrapper)

    def unsetVisualization(self):
        self.thisptr.setVisualization(NULL)

    def init(self, unsigned char[:,::1] image,int id, double ts):
        self.thisptr.randomInit(&image[0,0], ts, id)
        
    def track_frame(self,unsigned char[:,::1] image, int id, double ts, bint blockedUntilMapped):
        with nogil:
            self.thisptr.trackFrame(&image[0,0], id, blockedUntilMapped, ts)
        
    def finalize(self):
        self.thisptr.finalize()

    def optimize_graph(self):
        self.thisptr.optimizeGraph()
 
    def set_settings(self):
        sls.minUseGrad = 5       #1, 50
        sls.cameraPixelNoise2 = 16  #1, 50
        
        sls.KFUsageWeight = 4.0   #0.0, 20
        sls.KFDistWeight = 3.0    #0.0, 20
        
        sls.doSlam = True
        sls.doKFReActivation = True
        sls.doMapping = True
        sls.useFabMap = True

        sls.allowNegativeIdepths = True
        sls.useSubpixelStereo = True
        sls.useAffineLightningEstimation = False
        sls.multiThreading = True
        
        sls.maxLoopClosureCandidates = 10 #0, 50
        sls.loopclosureStrictness = 1.5 #0.0, 100
        sls.relocalizationTH = 0.7 #0, 1
        
        sls.depthSmoothingFactor = 1 # 0, 10

        sls.displayDepthMap = True
        sls.onSceenInfoDisplay = True
        # todo. automate this. Load dependecies with package.
        sls.packagePath = '/home/pupil/rosbuild_ws/package_dir/lsd_slam/lsd_slam_core/'

cdef class Slam_Undistorter:
    cdef sls.Undistorter *thisptr
    
    def __cinit__(self,camera_config_path):
        pass
    def __init__(self,camera_config_path):
        self.thisptr = sls.Undistorter.getUndistorterForFile(camera_config_path)
    
    def get_output_size(self):
        height,width = self.thisptr.getOutputWidth(), self.thisptr.getOutputHeight()
        return width,height
    
    def undistort(self, unsigned char[:,::1] raw_image):
        cdef np.ndarray[np.uint8_t,ndim=2] undistorted_image = np.zeros(self.get_output_size(),dtype=np.uint8)
        self.thisptr.undistort2(&raw_image[0,0], &undistorted_image[0,0])
        return undistorted_image