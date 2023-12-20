import random
import unittest

import torch
import examples
import vector_mlx as vector
import analysis
import assignment
import gradient
import analysis


class HolisticVectorTest(unittest.TestCase):
    def test_palencia(self):
        system = examples.get_palencia_system()
        initial_assignment = assignment.extract_assignment(system)
        pm1 = vector.system_priority_matrix(system)

        holistic_lineal = analysis.HolisticFPAnalysis()
        analysis.reset_wcrt(system)
        holistic_lineal.apply(system)
        r1 = [t.wcrt for t in system.tasks]

        holistic_vector = vector.VectorHolisticFPAnalysis()
        holistic_vector.clear_results()
        analysis.reset_wcrt(system)
        holistic_vector.apply(system)
        r2 = [t.wcrt for t in system.tasks]

        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a, b)

        # PD

        pd = assignment.PDAssignment()
        pd.apply(system)
        pm2 = vector.system_priority_matrix(system)

        ## Lineal
        analysis.reset_wcrt(system)
        holistic_lineal.apply(system)
        r3 = [t.wcrt for t in system.tasks]

        ## Vector (two priorites at the same time)
        assignment.insert_assignment(system, initial_assignment)
        holistic_vector.clear_results()
        analysis.reset_wcrt(system)
        holistic_vector.apply(system, scenarios=pm2.reshape(1, 6, 6))
        r = holistic_vector.full_response_times

        # r[:0,] should be equal to r1
        # r[:1,] should be equal to r3

        self.assertListEqual(r1, r[:, 0].tolist())
        self.assertListEqual(r3, r[:, 1].tolist())

    def test_cached_analysis_big(self):
        rnd = random.Random(42)
        system = examples.get_big_system(random=rnd, utilization=0.8, balanced=True)
        pd = assignment.PDAssignment()
        pd.apply(system)

        holistic1 = vector.VectorHolisticFPAnalysis()
        analysis.reset_wcrt(system)
        holistic1.apply(system)
        results1 = [t.wcrt for t in system.tasks]

        holistic2 = vector.VectorHolisticFPAnalysis()
        analysis.reset_wcrt(system)
        holistic2.apply(system)
        results2 = [t.wcrt for t in system.tasks]

        self.assertListEqual(results1, results2)

    def test_cached_analysis_scenarios_small(self):
        rnd = random.Random(1)
        system = examples.get_small_system(random=rnd, utilization=0.7, balanced=True)
        extractor = gradient.PriorityExtractor()
        mapper = vector.PrioritiesMatrix()
        pd = assignment.PDAssignment()

        v1 = extractor.extract(system)
        pd.apply(system)
        v2 = extractor.extract(system)
        pm = mapper.apply(system, [v1, v2])

        holistic1 = vector.VectorHolisticFPAnalysis()
        holistic2 = vector.VectorHolisticFPAnalysis()

        analysis.reset_wcrt(system)
        holistic1.apply(system, scenarios=pm)
        r1 = holistic1.full_response_times

        analysis.reset_wcrt(system)
        holistic2.apply(system, scenarios=pm)
        r2 = holistic2.full_response_times
        self.assertTrue(torch.all(r1 == r2))