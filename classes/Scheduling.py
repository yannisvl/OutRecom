from classes.Scheduling.Algorithms.AllocationScaledGreedy import AllocationScaledGreedy
from classes.Scheduling.Algorithms.SimpleScaledGreedy import SimpleScaledGreedy
from classes.Scheduling.Algorithms.ScaledGreedy import ScaledGreedy
from classes.Scheduling.Algorithms.OptScheduling2 import OptScheduling2
from classes.Scheduling.Datasets.Yahoo import Yahoo
from utils.SchedulingUtils import makespan

import numpy as np
import random

def from_df_to_numpy(df):
    pivot_df = df.pivot(index='id', columns='keyphrase', values='bid').fillna(10**6)
    array_2d = pivot_df.to_numpy()
    return array_2d


class Scheduling():
    def __init__(self):
        self.optAlg = OptScheduling2(50)
        self.predAlg = AllocationScaledGreedy(1)
        self.compAlg1 = SimpleScaledGreedy()
        self.compAlg2 = ScaledGreedy(1)
        self.dataset = Yahoo("datasets/Scheduling/ydata-ysm-keyphrase-bid-imp-click-v1_0")
        self.num_preds = 100

    def create_prediction(self, times):
        
        minVal = np.min(self.dataset.min_values)
        maxVal = np.max(self.dataset.max_values)
        
        normal_distributions = np.empty_like(times, dtype=float)
        for j in range(times.shape[1]):
            std_dev = self.dataset.max_values[j] / 3.0
            normal_distributions[:, j] = np.random.normal(loc=0, scale=std_dev, size=times.shape[0])

        total = np.clip((times+normal_distributions)*self.dataset.non_zero_vals, minVal, maxVal) + times*self.dataset.zero_vals
        return total
    
    def create_prediction1(self, times):
        random_array_normal = np.random.normal(loc=0, scale=200, size=(100, 100))
        noisy = times + random_array_normal
        return np.clip(noisy, 1, 1500)
    
    def run_experiment1(self):
        instance = self.dataset.instance
        np.savetxt("instances/instance0.txt", instance)

        opt = self.optAlg.solve(instance)
        opt_makespan = makespan(instance, opt)
        print("opt found! with makespan =", opt_makespan)

        #opt, ssg, sg, asg, err, eta
        data = np.zeros((self.num_preds, 6))
        for i in range(self.num_preds):
            print("iteration prediction", i)
            pred_times = self.create_prediction(instance)
            np.savetxt("instances/instance"+str(i+1)+".txt", pred_times)

            pred_assign = self.optAlg.solve(pred_times)
            pred_assign_makespan = makespan(instance, pred_assign)

            err = pred_assign_makespan / opt_makespan
            eta = max(np.max(pred_times / instance), np.max(instance / pred_times))

            simpleScaledGreedy_output = self.compAlg1.solve(instance, pred_times, pred_assign)
            scaledGreedy_output = self.compAlg2.solve(instance, pred_times, pred_assign, pred_assign_makespan)
            allocationScaledGreedy_output = self.predAlg.solve(instance, pred_assign)

            SSG_makespan = makespan(instance, simpleScaledGreedy_output)
            SG_makespan = makespan(instance, scaledGreedy_output)
            ASG_makespan = makespan(instance, allocationScaledGreedy_output)

            print(f"opt {opt_makespan}, SimpleSG {SSG_makespan}, ScaledG {SG_makespan}, AllocationSG {ASG_makespan}, eta {eta}, err {err}")
            data[i,:] = (opt_makespan, SSG_makespan, SG_makespan, ASG_makespan, err, eta)
        
        np.savez("experiments/sched/preds"+str(self.num_preds)+'.npz', array1=data)

    def run_experiment(self):
        self.experiment_many_instances_smart_pred()

    def experiment_many_instances_smart_pred(self):
        max_jobs = 100
        rounds = 100
        keyphrases = self.dataset.read_keyphrases()
        
        #opt, ssg, sg, asg, err, eta
        data = np.zeros((rounds, 6))
        for i in range(rounds):
            number_of_jobs = np.random.randint(2,max_jobs)
            jobs = random.sample(keyphrases, number_of_jobs)
            day = self.dataset.random_day(jobs)
            
            #create instance
            ps = self.dataset.df[self.dataset.df['keyphrase'].isin(jobs) & self.dataset.df['day']==day]
            average_bid = ps.groupby(['keyphrase', 'id'])['bid'].mean().to_frame()
            instance = from_df_to_numpy(average_bid)

            #create prediction
            pred = self.dataset.df[self.dataset.df['keyphrase'].isin(jobs) & self.dataset.df['day']<day]
            average_past = pred.groupby(['keyphrase', 'id', 'day'])['bid'].mean()
            avg_of_avg_past = average_past.groupby(['keyphrase', 'id'])['bid'].mean().to_frame()
            pred_times = from_df_to_numpy(avg_of_avg_past)

            #run
            opt = self.optAlg.solve(instance)
            opt_makespan = makespan(instance, opt)

            pred_assign = self.optAlg.solve(pred) # maybe use ranks
            pred_assign_makespan = makespan(instance, pred_assign)

            err = pred_assign_makespan / opt_makespan
            eta = max(np.max(pred_times / instance), np.max(instance / pred_times))

            simpleScaledGreedy_output = self.compAlg1.solve(instance, pred_times, pred_assign)
            scaledGreedy_output = self.compAlg2.solve(instance, pred_times, pred_assign, pred_assign_makespan)
            allocationScaledGreedy_output = self.predAlg.solve(instance, pred_assign)
            SSG_makespan = makespan(instance, simpleScaledGreedy_output)
            SG_makespan = makespan(instance, scaledGreedy_output)
            ASG_makespan = makespan(instance, allocationScaledGreedy_output)

            print(f"opt {opt_makespan}, SimpleSG {SSG_makespan}, ScaledG {SG_makespan}, AllocationSG {ASG_makespan}, eta {eta}, err {err}")
            data[i,:] = (opt_makespan, SSG_makespan, SG_makespan, ASG_makespan, err, eta)

        np.savez("experiments/sched/preds.npz", array1=data)