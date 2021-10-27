import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        # Init
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        # Use Levenberg algorithm (can be Guass-Newton instead like in paper?)
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        g2o.

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        # 3D pose graph vertex (x,y,z,q.x,q.y,q.z) as q.w is given from sqrt(1-||q.x,q.y,q.z||)
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


def main():
    # Main
    print("Hello world!")
    # Extract frames from video (or images in the beginning)
    # Estimate wind turbine model (2D image points) from internal wind turbine model through the current estimated pose of the camera
    # Compare this estimated model with the output from the CNN
    # Add these to the optimizer and run optimization

if __name__ == "__main__":
    main()