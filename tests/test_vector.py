import unittest

import numpy as np

import examples
import vector
import analysis
import assignment
import gradient


class HolisticVectorTest(unittest.TestCase):
    def test_system_priority_matrix(self):
        system = examples.get_palencia_system()
        pm = vector.system_priority_matrix(system)

        expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.assertTrue((pm == expected).all())

    def test_palencia(self):
        """
        Test that it correctly creates a priority matrix
        """
        system = examples.get_palencia_system()

        extractor = gradient.MappingPriorityExtractor()
        vector_analysis = vector.VectorHolisticFPAnalysis()
        pd = assignment.PDAssignment()
        matrix_extractor = vector.MappingPrioritiesMatrix()

        x = extractor.extract(system)
        input = [x]
        pm1 = matrix_extractor.apply(system, input)

        _, _, _, _, mappings, priorities = vector.get_vectors(system)
        pm2 = vector.priority_matrix(priorities) * (mappings == mappings.T)

        self.assertTrue((pm1 == pm2).all())

    def test_gradient(self):
        """
        Test that we can correctly force a priority matrix into the GDPA algorithm
        """

        sched_test = analysis.HolisticFPAnalysis(limit_factor=10, reset=False)
        extractor = gradient.PriorityExtractor()
        cost_function = gradient.InvslackCost(extractor=extractor, analysis=sched_test)
        stop_function = gradient.FixedIterationsStop(iterations=5)
        delta_function = gradient.AvgSeparationDelta(factor=1.5)
        batch_cost_function = gradient.VectorHolisticFPBatchCosts(vector.PrioritiesMatrix())
        gradient_function = gradient.StandardGradient(delta_function=delta_function,
                                                      batch_cost_function=batch_cost_function)
        update_function = gradient.NoisyAdam()
        optimizer = gradient.StandardGradientDescent(extractor=extractor,
                                                     cost_function=cost_function,
                                                     stop_function=stop_function,
                                                     gradient_function=gradient_function,
                                                     update_function=update_function,
                                                     verbose=False)

        pd = assignment.PDAssignment(normalize=True)

        system = examples.get_palencia_system()
        pd.apply(system)
        optimizer.apply(system)
        sched_test.apply(system)

        self.assertTrue(system.is_schedulable())
