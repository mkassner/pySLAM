from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

cdef extern from "<Eigen/Eigen>" namespace "Eigen":
    cdef cppclass Matrix3f:
        Matrix3f() except + 
        Matrix3f(int rows, int cols) except + 
        float * data()

    cdef cppclass aligned_allocator[T]: 
        pass

    cdef cppclass Matrix31f "Eigen::Matrix<float, 3,1>": # eigen defaults to column major layout
        Matrix() except + 
        float * data()

    cdef cppclass Matrix201f "Eigen::Matrix<float, 20,1>": # eigen defaults to column major layout
        Matrix() except + 
        float * data()

    cdef cppclass Vector4f:
        pass


cdef extern from "<util/SophusUtil.h>" :
    cdef cppclass SE3:
        pass
    cdef cppclass Sim3:
        # Sim3[T] cast[T]()
        float * data()
    cdef cppclass SO3:
        pass
    

cdef extern from "<boost/thread/shared_mutex.hpp>" namespace "boost":
    cdef cppclass shared_mutex:
        pass

cdef extern from "<boost/thread/locks.hpp>" namespace "boost":
    cdef cppclass shared_lock[T]:
        pass


cdef extern from "<IOWrapper/Output3DWrapper.h>" namespace "lsd_slam":
    cdef cppclass Output3DWrapper:
        _Output3DWrapper() except +


        void publishKeyframeGraph(KeyFrameGraph* graph) with gil
        # publishes a keyframe. if that frame already existis, it is overwritten, otherwise it is added.
        void publishKeyframe(Frame* kf) with gil

        # published a tracked frame that did not become a keyframe (yet i.e. has no depth data)
        void publishTrackedFrame(Frame* kf) with gil

        #publishes graph and all constraints, as well as updated KF poses.
        # void publishTrajectory(vector[Matrix31f] pt, string identifier) with gil
        # void publishTrajectoryIncrement(Matrix31f pt, string identifier) with gil

        void publishDebugInfo(Matrix201f data) with gil


cdef extern from "<util/settings.h>" namespace "lsd_slam":
    # keystrokes
    extern bint autoRun
    extern bint autoRunWithinFrame
    extern int debugDisplay
    extern bint displayDepthMap
    extern bint onSceenInfoDisplay
    extern bint dumpMap
    extern bint doFullReConstraintTrack
    
    # dyn
    extern bint printPropagationStatistics
    extern bint printFillHolesStatistics
    extern bint printObserveStatistics
    extern bint printObservePurgeStatistics
    extern bint printRegularizeStatistics
    extern bint printLineStereoStatistics
    extern bint printLineStereoFails
    
    extern bint printTrackingIterationInfo
    extern bint printThreadingInfo
    
    extern bint printKeyframeSelectionInfo
    extern bint printConstraintSearchInfo
    extern bint printOptimizationInfo
    extern bint printRelocalizationInfo
    
    extern bint printFrameBuildDebugInfo
    extern bint printMemoryDebugInfo
    
    extern bint printMappingTiming
    extern bint printOverallTiming
    extern bint plotTrackingIterationInfo
    extern bint plotSim3TrackingIterationInfo
    extern bint plotStereoImages
    extern bint plotTracking
    
    
    extern bint allowNegativeIdepths
    extern bint useMotionModel
    extern bint useSubpixelStereo
    extern bint multiThreading
    extern bint useAffineLightningEstimation
    
    extern float freeDebugParam1
    extern float freeDebugParam2
    extern float freeDebugParam3
    extern float freeDebugParam4
    extern float freeDebugParam5
    
    
    extern float KFDistWeight
    extern float KFUsageWeight
    extern int maxLoopClosureCandidates
    extern int propagateKeyFrameDepthCount
    extern float loopclosureStrictness
    extern float relocalizationTH
    
    
    extern float minUseGrad
    extern float cameraPixelNoise2
    extern float depthSmoothingFactor
    
    extern bint useFabMap
    extern bint doSlam
    extern bint doKFReActivation
    extern bint doMapping
    
    extern bint saveKeyframes
    extern bint saveAllTracked
    extern bint saveLoopClosureImages
    extern bint saveAllTrackingStages
    extern bint saveAllTrackingStagesInternal
    
    extern bint continuousPCOutput
    extern string packagePath

cdef extern from "<GlobalMapping/KeyFrameGraph.h>" namespace "lsd_slam":
    cdef cppclass KeyFrameGraph:
        KeyFrameGraph() except +


cdef extern from "<DataStructures/FramePoseStruct.h>" namespace "lsd_slam":
    cdef cppclass FramePoseStruct:
        FramePoseStruct() except +

# ctypedef FramePoseStruct* FramePoseStruct_p


cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass pose_vector "std::vector<FramePoseStruct*, Eigen::aligned_allocator<lsd_slam::FramePoseStruct*>":
        pass

cdef extern from "<DataStructures/Frame.h>" namespace "lsd_slam":
    cdef cppclass Frame:
        Frame(int id, int width, int height,Matrix3f& K, double timestamp, unsigned char* image) except +

        int id()
        int width()
        int height()
        const Matrix3f& K(int level = 0) const
        const Matrix3f& KInv(int level = 0) const
        # /** Returns K(0, 0). */
        inline float fx(int level = 0) const
        # /** Returns K(1, 1). */
        inline float fy(int level = 0) const
        # /** Returns K(0, 2). */
        inline float cx(int level = 0) const
        # /** Returns K(1, 2). */
        inline float cy(int level = 0) const
        # /** Returns KInv(0, 0). */
        inline float fxInv(int level = 0) const
        # /** Returns KInv(1, 1). */
        inline float fyInv(int level = 0) const
        # /** Returns KInv(0, 2). */
        inline float cxInv(int level = 0) const
        # /** Returns KInv(1, 2). */
        inline float cyInv(int level = 0) const
        
        # /** Returns the frame's recording timestamp. */
        inline double timestamp() const

        inline float* image(int level = 0)
        inline const Vector4f* gradients(int level = 0)
        inline const float* maxGradients(int level = 0)
        inline bint hasIDepthBeenSet() const
        inline const float* idepth(int level = 0)
        inline const float* idepthVar(int level = 0)
        inline const unsigned char* validity_reAct()
        inline const float* idepth_reAct()
        inline const float* idepthVar_reAct()

        inline bint* refPixelWasGood()
        inline bint* refPixelWasGoodNoCreate()
        inline void clear_refPixelWasGood()
        FramePoseStruct* pose

        Sim3 getScaledCamToWorld(int num=0)
        bint hasTrackingParent()
        Frame* getTrackingParent() 
        shared_lock[shared_mutex] getActiveLock()


cdef extern from "SlamSystem.h" namespace "lsd_slam":
    cdef cppclass SlamSystem:
        SlamSystem(int w, int h, Matrix3f K, bint enableSLAM) except +
        
        void randomInit(unsigned char* image, double timeStamp, int id) nogil
        void trackFrame(unsigned char* image, unsigned int frameID, bint blockUntilMapped, double timestamp) nogil
        
        void finalize()
        void optimizeGraph()
        void setVisualization(Output3DWrapper* outputWrapper) except +
        
        Frame* getCurrentKeyframe()
        pose_vector getAllPoses()


cdef extern from "<util/Undistorter.h>" namespace "lsd_slam":
    cdef cppclass Undistorter:
        Undistorter() except +

        void undistort2(unsigned char*, unsigned char*)
        
        int getOutputWidth()
        int getOutputHeight()
    
        @staticmethod
        Undistorter* getUndistorterForFile(const char* configFilename)

    
