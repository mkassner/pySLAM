cimport slam_system as sls
cimport numpy as np
import numpy as np
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
import time

cdef extern from "Python.h":
    void PyEval_InitThreads()

def init_threads():
    PyEval_InitThreads()


cdef cppclass cyOutput3DWrapper(sls.Output3DWrapper):
    PyObject* callback

    __init__(object callback):  # constructor. "this" argument is implicit.
        Py_INCREF(callback)
        this.callback = <PyObject*>callback


    __dealloc__():  # destructor
        Py_DECREF(<object>this.callback)

    void publishKeyframe(sls.Frame* f) with gil:

        cdef sls.shared_lock[sls.shared_mutex] lock = f.getActiveLock()
        print f.id(),f.width(),f.height()

        cdef sls.Sim3 pose = f.getScaledCamToWorld()
        for x in range(7):
            print pose.data()[x]
        pose.data() # assuming float data and skipping explicit sim3 cast

        # boost::shared_lock<boost::shared_mutex> lock = f->getActiveLock();

        # fMsg.id = f->id();
        # fMsg.time = f->timestamp();
        # fMsg.isKeyframe = true;

        # int w = f->width(publishLvl);
        # int h = f->height(publishLvl);

        # memcpy(fMsg.camToWorld.data(),f->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
        # fMsg.fx = f->fx(publishLvl);
        # fMsg.fy = f->fy(publishLvl);
        # fMsg.cx = f->cx(publishLvl);
        # fMsg.cy = f->cy(publishLvl);
        # fMsg.width = w;
        # fMsg.height = h;


        # fMsg.pointcloud.resize(w*h*sizeof(InputPointDense));

        # InputPointDense* pc = (InputPointDense*)fMsg.pointcloud.data();

        # const float* idepth = f->idepth(publishLvl);
        # const float* idepthVar = f->idepthVar(publishLvl);
        # const float* color = f->image(publishLvl);

        # for(int idx=0;idx < w*h; idx++)
        # {
        #     pc[idx].idepth = idepth[idx];
        #     pc[idx].idepth_var = idepthVar[idx];
        #     pc[idx].color[0] = color[idx];
        #     pc[idx].color[1] = color[idx];
        #     pc[idx].color[2] = color[idx];
        #     pc[idx].color[3] = color[idx];
        # }


        print "Key Frame"
        # (<object>this.callback)()

    void publishTrackedFrame(sls.Frame* f) with gil:
        pass
        print "Tracked Frame"
        # (<object>this.callback)()

    void publishKeyframeGraph(sls.KeyFrameGraph* graph) with gil:
        print 'Graph'

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
        
    def setVisualization(self):
        def hi():
            print "hi"

        cdef cyOutput3DWrapper* output_wrapper = new cyOutput3DWrapper(hi)
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