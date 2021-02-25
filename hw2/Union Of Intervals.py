
#################################
# Your name: eilon_storzi
#################################

import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import ndarray
import matplotlib.patches as mpatches
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        objarray = np.zeros((m, 2))
        x_vec = np.random.uniform(0, 1, m)
        x_vec.sort()
        y_vec = []
        for x in x_vec:
            if ((x >= 0 and x <= 0.2) or (x >= 0.4 and x < 0.6) or x >= 0.8 and x <= 1):
                y = np.random.choice([0, 1], p=[0.2, 0.8])
            elif ((x >= 0.2 and x <= 0.4) or (x >= 0.6 and x < 0.8)):
                y = np.random.choice([0, 1], p=[0.9, 0.1])
            y_vec.append(y)

        for i in range(m):
            objarray[i] = x_vec[i], y_vec[i]
        return objarray


    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        pairs = self.sample_from_D(m)
        plt.plot([pairs[i][0] for i in range(len(pairs))], [pairs[i][1] for i in range(len(pairs))], "go")
        plt.ylabel("labels")
        plt.xlabel("objects")
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        xlines = [0.2, 0.4, 0.6, 0.8, 1]
        for j in xlines:
            plt.axvline(x=j, linestyle="--")
        b_intervals, error = intervals.find_best_interval([pairs[i][0] for i in range(len(pairs))],
                                                          [pairs[i][1] for i in range(len(pairs))], k)
        for interval in b_intervals:
            plt.hlines(y=-0.1, xmin=interval[0], xmax=interval[1], colors='r', lw=5, )
        red_patch = mpatches.Patch(color='red', label='best intervals')
        plt.legend(handles=[red_patch], bbox_to_anchor=(1.1, 1.1))
        #plt.show()
        return



    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        erm_array = [0 for i in range(m_first, m_last + 1, step)]
        true_error_array = [0 for i in range(m_first, m_last + 1, step)]
        for j in range(T):
            for m in range(m_first, m_last + 1, step):
                pairs_array = self.sample_from_D(m)
                b_intervals, es_error = intervals.find_best_interval(
                    [pairs_array[i][0] for i in range(len(pairs_array))],
                    [pairs_array[i][1] for i in range(len(pairs_array))], k)
                ep_error = self.true_error(b_intervals)
                erm_array[(m // step) - (m_first // step)] += (es_error / m)
                true_error_array[(m // step) - (m_first // step)] += (ep_error)
        for i in range(len(erm_array)):
            erm_array[i] = erm_array[i] / T
        for j in range(len(true_error_array)):
            true_error_array[j] = true_error_array[j] / T
        green_patch = mpatches.Patch(color='green', label='True Error')
        blue_patch = mpatches.Patch(color='blue', label='Empirical Error')
        plt.legend(handles=[green_patch, blue_patch])
        plt.plot([m for m in range(m_first, m_last + 1, step)], erm_array, 'bo')
        plt.plot([m for m in range(m_first, m_last + 1, step)], true_error_array, 'go')
        #plt.show()
        avg_es_ep = np.ndarray(shape=(len(true_error_array), 2))
        for i in range(len(true_error_array)):
            avg_es_ep[i][0] = erm_array[i]
            avg_es_ep[i][1] = true_error_array[i]
        return avg_es_ep

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        erm_array = [0 for i in range(k_first, k_last + 1, step)]
        true_error_array = [0 for i in range(k_first, k_last + 1, step)]
        pairs_array = pairs_array = self.sample_from_D(m)
        best_k = k_first
        best_emp = 1
        for k in range(k_first, k_last + 1, step):
            b_intervals, es_error = intervals.find_best_interval([pairs_array[i][0] for i in range(len(pairs_array))],
                                                                 [pairs_array[i][1] for i in range(len(pairs_array))],
                                                                 k)
            ep_error = self.true_error(b_intervals)
            erm_array[(k // step) - (k_first // step)] += (es_error / m)
            if (es_error / m < best_emp):
                best_k = k
                best_emp = es_error / m
            true_error_array[(k // step) - (k_first // step)] += (ep_error)
        green_patch = mpatches.Patch(color='green', label='True Error')
        blue_patch = mpatches.Patch(color='blue', label='Empirical Error')
        plt.legend(handles=[green_patch, blue_patch])
        plt.plot([k for k in range(k_first, k_last + 1, step)], erm_array, 'bo')
        plt.plot([k for k in range(k_first, k_last + 1, step)], true_error_array, 'go')
        #plt.show()
        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        delta = 0.1
        erm_array = [0 for i in range(k_first, k_last + 1, step)]
        true_error_array = [0 for i in range(k_first, k_last + 1, step)]
        penalty_and_erm_array = [0 for i in range(k_first, k_last + 1, step)]
        penalty_array = []
        pairs_array = pairs_array = self.sample_from_D(m)
        best_k = k_first
        best_emp_with_penalty = 1
        for k in range(k_first, k_last + 1, step):
            d = 2 * k
            penalty = np.sqrt((8 / m) * (d * (np.log((2 * np.e * m) / 2)) + (np.log((4 / delta)))))
            penalty_array.append(penalty)
            b_intervals, es_error = intervals.find_best_interval([pairs_array[i][0] for i in range(len(pairs_array))],[pairs_array[i][1] for i in range(len(pairs_array))],k)
            ep_error = self.true_error(b_intervals)
            erm_array[(k // step) - (k_first // step)] += (es_error / m)
            penalty_and_erm_array[(k // step) - (k_first // step)] += (es_error / m) + (penalty)
            if ((es_error / m) + penalty < best_emp_with_penalty):
                best_k = k
                best_emp_with_penalty = (es_error / m) + penalty
            true_error_array[(k // step) - (k_first // step)] += (ep_error)
        green_patch = mpatches.Patch(color='green', label='True Error')
        blue_patch = mpatches.Patch(color='blue', label='Empirical Error')
        pink_patch = mpatches.Patch(color='pink', label='Empirical Error+Penalty')
        red_patch = mpatches.Patch(color='red', label='Penalty')
        plt.legend(handles=[green_patch, blue_patch, red_patch, pink_patch])

        plt.plot([k for k in range(k_first, k_last + 1, step)], erm_array, 'bo')
        plt.plot([k for k in range(k_first, k_last + 1, step)], true_error_array, 'go')
        plt.plot([k for k in range(k_first, k_last + 1, step)], penalty_and_erm_array, 'mo')
        plt.plot([k for k in range(k_first, k_last + 1, step)], penalty_array, 'ro')
        #plt.show()
        return best_k

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        best_k_list = []
        pairs_array = pairs_array = self.sample_from_D(m)
        hold_out_size = m // 5
        for i in range(T):
            best_k = 1
            best_hold_out_error = 1
            np.random.shuffle(pairs_array)
            hold_out_array = [pairs_array[i] for i in range(hold_out_size)]
            train_array = [pairs_array[i] for i in range(hold_out_size, m)]
            train_array.sort(key=lambda x: x[0])
            for k in range(1, 11):
                b_intervals, es_error = intervals.find_best_interval(
                    [train_array[i][0] for i in range(len(train_array))],
                    [train_array[i][1] for i in range(len(train_array))], k)
                error = (self.hold_out_error(hold_out_array, b_intervals)) / hold_out_size
                if (error < best_hold_out_error):
                    best_k = k
                    best_hold_out_error = error
            best_k_list.append(best_k)
        experiment_list = [0] * 10
        for k in best_k_list:
            experiment_list[k - 1] += 1
        max_k = 0
        best_hypothesis = 0
        for i in range(len(experiment_list)):
            if (experiment_list[i] > max_k):
                max_k = experiment_list[i]
                best_hypothesis = i + 1
        return best_hypothesis

    #################################
    # Place for additional methods
    def intersection_length_of_intervals(self,interval1, interval2):
        start = 0
        end = 0
        if (interval1[1] <= interval2[0] or interval2[1] <= interval1[0]):
            return 0
        if (interval1[0] >= interval2[0]):
            start = interval1[0]
            end = min(interval1[1], interval2[1])
        else:
            start = interval2[0]
            end = min(interval1[1], interval2[1])
        return (end - start)

    def true_error(self,intervals):
        error = 0
        one_intervals = [(0.0, 0.2), (0.4, 0.6), (0.8, 1)]
        zero_intervals = [(0.2, 0.4), (0.6, 0.8)]
        matching_1_len = 0
        matching_0_len = 0
        opposite_intervals = [(0, intervals[0][0])]
        intervals_sum = 0
        opposite_intervals_sum = 0
        not_matching_1_len = 0
        not_matching_0_len = 0
        for i in range(len(intervals)):
            intervals_sum += (intervals[i][1] - intervals[i][0])
        for j in range(len(intervals) - 1):
            opposite_intervals.append((intervals[j][1], intervals[j + 1][0]))
        opposite_intervals.append((intervals[len(intervals) - 1][1], 1))
        for i in range(len(opposite_intervals)):
            opposite_intervals_sum += (opposite_intervals[i][1] - opposite_intervals[i][0])
        for interval in intervals:
            for i in range(len(one_intervals)):
                matching_1_len += self.intersection_length_of_intervals(interval, one_intervals[i])
        for interval in opposite_intervals:
            for j in range(len(zero_intervals)):
                matching_0_len += self.intersection_length_of_intervals(interval, zero_intervals[j])
        not_matching_1_len = intervals_sum - matching_1_len
        not_matching_0_len = opposite_intervals_sum - matching_0_len
        error += 0.2 * matching_1_len + 0.9 * not_matching_1_len + 0.8 * not_matching_0_len + 0.1 * matching_0_len
        return error

    def hold_out_error(self, hold_out_array, intervals):
        count = 0
        for elem in hold_out_array:
            flag_inside_intervals = False
            for interval in intervals:
                if ((interval[0] <= elem[0]) and (elem[0] <= interval[1])):
                    flag_inside_intervals = True
            if (flag_inside_intervals):
                if (elem[1] == 0):
                    count += 1
            else:
                if (elem[1] == 1):
                    count += 1
        return count
    #################################

if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
