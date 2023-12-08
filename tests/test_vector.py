import unittest

import numpy as np
import examples
import vector, vector2
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

        self.assertTrue(np.all(np.equal(r1, r[:,0])))
        self.assertTrue(np.all(np.equal(r3, r[:,1])))

    def test_system_priority_matrix(self):
        system = examples.get_palencia_system()
        pm = vector.system_priority_matrix(system)

        expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.assertTrue((pm == expected).all())

    def test_priority_matrix(self):
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

    def test_finished_scenarios(self):
        scenarios = 100
        tasks = 10
        mask = np.ones((scenarios, tasks, 1))

        # no scenario finished
        finished = []
        self.assertTrue(set(vector.finished_scenarios(mask)) == set(finished))

        # first finished
        finished = [0]
        for s in finished:
            mask[s,::] = 0
        self.assertTrue(set(vector.finished_scenarios(mask)) == set(finished))

        # + last finished
        finished.append(99)
        for s in finished:
            mask[s,::] = 0
        self.assertTrue(set(vector.finished_scenarios(mask)) == set(finished))

        # random finished in the middle
        finished.extend([1, 3, 10, 11, 20, 21, 22, 23, 50, 75, 80, 98])
        for s in finished:
            mask[s,::] = 0
        self.assertTrue(set(vector.finished_scenarios(mask)) == set(finished))

        # all finished
        finished = range(scenarios)
        for s in finished:
            mask[s, ::] = 0
        self.assertTrue(set(vector.finished_scenarios(mask)) == set(finished))

    def test_extract_scenario_data(self):
        matrix = np.array(range(100)).reshape(10, 10, -1)

        for i in range(10):
            data = np.array(range(i*10, i*10+10)).reshape(-1, 1)
            extracted = vector.extract_scenario_data(matrix, i)
            self.assertTrue(np.all(np.equal(data, extracted)))

        matrix = np.array(range(100)).reshape(-1, 2, 2)
        for i in range(25):
            data = np.array(range(i*4, i*4+4)).reshape(2, 2)
            extracted = vector.extract_scenario_data(matrix, i)
            self.assertTrue(np.all(np.equal(data, extracted)))

    def test_scenarios_over_limit(self):
        s,t = 6, 10

        r = np.array([range(s*t)]).reshape(s, t, 1)

        # no scenario should exceed limit
        r_limit = np.array([100]*t).reshape(t, 1)
        res = vector2.scenarios_over_limit(r, r_limit)
        self.assertTrue(~np.all(res))

        # only last scenario should exceed limit
        limits = [100] * (t-1) + [58]
        r_limit = np.array(limits).reshape(t, 1)
        res = vector2.scenarios_over_limit(r, r_limit)
        self.assertTrue(np.sum(res) == 1)
        self.assertTrue(np.all(res[-1]))
        self.assertTrue(~np.all(res[0:-1]))

        # every scenario should exceed limit
        r_limit = np.array([0] * t).reshape(t, 1)
        res = vector2.scenarios_over_limit(r, r_limit)
        self.assertTrue(np.all(res))

    def test_cache_scenario_results(self):
        t = 6
        s = 4
        pm = np.random.choice(a=[False, True], size=(s, t, t))
        r = np.random.randint(100, size=(s, t, 1))
        cache = vector2.ResultsCache()

        pm0 = vector2.extract_scenario_data(pm, 0)
        pm1 = vector2.extract_scenario_data(pm, 1)
        pm2 = vector2.extract_scenario_data(pm, 2)
        pm3 = vector2.extract_scenario_data(pm, 3)

        r0 = vector2.extract_scenario_data(r, 0)
        r1 = vector2.extract_scenario_data(r, 1)
        r2 = vector2.extract_scenario_data(r, 2)
        r3 = vector2.extract_scenario_data(r, 3)

        # cahe 0 scenarios
        mask = np.array([False]*s)
        vector2.cache_scenario_results(r, pm, mask, cache)
        self.assertEqual(len(cache), 0)

        # cahe scenario 0
        mask = np.array([True, False, False, False])
        vector2.cache_scenario_results(r, pm, mask, cache)
        self.assertEqual(len(cache), 1)
        self.assertTrue(np.array_equal(cache.get(pm0), r0))
        self.assertTrue(np.array_equal(cache.get(pm1), None))
        self.assertTrue(np.array_equal(cache.get(pm2), None))
        self.assertTrue(np.array_equal(cache.get(pm3), None))

        # add scenario 3
        mask = np.array([False, False, False, True])
        vector2.cache_scenario_results(r, pm, mask, cache)
        self.assertEqual(len(cache), 2)
        self.assertTrue(np.array_equal(cache.get(pm0), r0))
        self.assertTrue(np.array_equal(cache.get(pm1), None))
        self.assertTrue(np.array_equal(cache.get(pm2), None))
        self.assertTrue(np.array_equal(cache.get(pm3), r3))

        # add scenario 1 and 2
        mask = np.array([False, True, True, False])
        vector2.cache_scenario_results(r, pm, mask, cache)
        self.assertEqual(len(cache), 4)
        self.assertTrue(np.array_equal(cache.get(pm0), r0))
        self.assertTrue(np.array_equal(cache.get(pm1), r1))
        self.assertTrue(np.array_equal(cache.get(pm2), r2))
        self.assertTrue(np.array_equal(cache.get(pm3), r3))

    def test_remove_scenarios(self):
        t = 4
        s = 4
        m1 = np.array(range(32)).reshape(s, t, -1)
        m2 = m1+100;
        m3 = m1*3.5

        # remove 0 scenarios
        mask = np.array([False, False, False, False])
        mo1, mo2, mo3 = vector2.remove_scenarios(mask, m1, m2, m3)
        self.assertTrue(np.array_equal(m1, mo1))
        self.assertTrue(np.array_equal(m2, mo2))
        self.assertTrue(np.array_equal(m3, mo3))

        # remove scenario 0
        mask = np.array([True, False, False, False])
        mo1, mo2, mo3 = vector2.remove_scenarios(mask, m1, m2, m3)
        self.assertEqual(m1.shape[0], mo1.shape[0] + 1)
        self.assertEqual(m2.shape[0], mo2.shape[0] + 1)
        self.assertEqual(m3.shape[0], mo3.shape[0] + 1)

        # remove every scenario
        mask = np.array([True, True, True, True])
        mo1, mo2, mo3 = vector2.remove_scenarios(mask, m1, m2, m3)
        self.assertEqual(m1.shape[0], mo1.shape[0] + 4)
        self.assertEqual(m2.shape[0], mo2.shape[0] + 4)
        self.assertEqual(m3.shape[0], mo3.shape[0] + 4)

    def test_build_results_from_cache(self):
        t = 5
        s = 4
        pm = np.random.choice(a=[False, True], size=(s, t, t))
        r = np.random.randint(100, size=(s, t, 1))

        pm0 = vector2.extract_scenario_data(pm, 0)
        pm1 = vector2.extract_scenario_data(pm, 1)
        pm2 = vector2.extract_scenario_data(pm, 2)
        pm3 = vector2.extract_scenario_data(pm, 3)

        r0 = vector2.extract_scenario_data(r, 0)
        r1 = vector2.extract_scenario_data(r, 1)
        r2 = vector2.extract_scenario_data(r, 2)
        r3 = vector2.extract_scenario_data(r, 3)

        cache = vector2.ResultsCache()
        cache.insert(pm0, r0)
        cache.insert(pm1, r1)
        cache.insert(pm2, r2)
        cache.insert(pm3, r3)

        expected = r.ravel(order="F").reshape(t, s)
        res = vector2.build_results_from_cache(pm, cache)
        self.assertTrue(np.array_equal(expected, res))

