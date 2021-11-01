
#参数的默认值 parameters
N = 3              #卖家数 Number of providers
M = 9              #买家数  Number of IoT devices
# c_max = 20         #卖家的成本系数的上限值 Upper bound of c_j (Unit cost for computing service of j)
# V_max = 500        #买家的任务完成奖励的上限值  Upper bound of V_i (task completion utility of i)
a_max = 2          #买家的完成任务所需的CPU工作时的上限值 upper bound of a_i (required CPU occupied time of i)
y_min =  0.0225     #动作空间的下限值 Min Auxiliary price profile for all providers(Pmax = 40)
y_max =  0.0625     #动作空间的上限值 Max Auxiliary price profile for all providers(Pmin = 16)
actionNumber = 8   #动作空间的大小
times = 50000      #迭代次数 iteration times

results_dir = 'results'

#生成买家参数、卖家参数
# generate provider's unit cost, task completion utility of device j, and required resource occupied time of i
c = [20,20,20]
V = [45,48,52,40,60,76,80,82,84]
a = [1.8]*M
# a = np.random.uniform(a_max - 0.5, a_max, size = M)
max_resources_per_seller = [12,40,80]
consumer_penalty_coeff = 0.05
producer_penalty_coeff = 0.05



################################ Mid term report coefficients ################################