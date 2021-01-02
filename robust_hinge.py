import sys
import random
import math

##########################Input############
datafile = sys.argv[1]
labelfile = sys.argv[2]

#############read data###################
f = open(datafile)
line = f.readline()
data = []
while line != '':
    a = line.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    l2.append(float(1))
    data.append(l2)
    line = f.readline()
rows = len(data)
cols = len(data[0])
f.close()
##########read label and assign label 0 as -1 ######
f = open(labelfile)
trainlabels = {}
l = f.readline()
while l != '':
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    if (trainlabels[int(a[1])]) == 0:
        trainlabels[int(a[1])] = -1
    l = f.readline()
f.close()


#######calculating varaiance of feature vector##############

def variance(data):
    ro = len(data)
    n = ro
    col = len(data[0])
    summ = []
    mean = []
    v = []
    var = []
    for l in range(0, col, 1):
        summ.append(0)
        mean.append(0)
        v.append(0)
        var.append(0)

    for j in range(0, col, 1):
        for k in range(0, ro, 1):
            summ[j] += data[k][j]
        mean[j] = summ[j] / n
    for t in range(0, col, 1):
        for r in range(0, ro, 1):
            v[t] += (data[r][t] - mean[t]) ** 2
        var[t] = math.sqrt(v[t] / n)
    return var


##########initialise W########################

w = []
for i in range(0, cols, 1):
    w.append(random.uniform(-0.01, 0.01))


###############calculating dot product#################
def dotproduct(u, v):
    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0, len(u), 1):
        dp += u[i] * v[i]
    return dp


variance_data = []
variance_data = variance(data)  ##calculating variance of features in dataset
############adding gaussian noise ---feature+random(variance,-variance)##########
gaussian = []
noise = [[0 for i in range(cols)] for j in range(rows)]
for i in range(0, cols, 1):
    for j in range(0, rows, 1):
        # noise[j][i] += random.uniform(0,3.6*variance_data[i])
        noise[j][i] = 4.2 * variance_data[i]  ##--2.929 3--59 3.4 54 3.5 40  --------3.6 30---

for i in range(0, cols, 1):
    for j in range(0, rows, 1):
        data[j][i] = data[j][i] + noise[j][i]
# print('adding noise',data[0])
###########hingeloss###########

eta = 0.01
dellf = []
for j in range(0, cols, 1):
    dellf.append(0)
######Objective iteration#####

prevobj = 1000000
obj = 10

while (abs(prevobj - obj) > .01):
    # Compute differential (dell_f)
    prevobj = obj
    dell_f = []
    for j in range(0, cols, 1):
        dell_f.append(0)
    #####delf update
    for i in range(0, rows, 1):
        if trainlabels.get(i) != None:
            dp = dotproduct(w, data[i])
            sub_grad = trainlabels.get(i) * dp
            if sub_grad < 1:
                for j in range(0, cols, 1):
                    dell_f[j] += trainlabels.get(i) * data[i][j]
    #######update w
    for j in range(cols):
        w[j] += eta * dell_f[j]
    error = 0

    ####error calculation#######

    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            dp = dotproduct(w, data[i])
            hinge_loss = 1 - (trainlabels.get(i) * dp)
            if (hinge_loss > 0):
                error += hinge_loss
            else:
                error += 0
    obj = error
    #print("objective is", obj)

#print(w)
#print("len w", len(w), cols)
w_out = [str(element) for element in w]
w_out = ' '.join(w_out)
OUT = open("w_vector", 'w')
OUT.write(w_out)
OUT.write('\n')
OUT.write(str(w[len(w)-1])+'\n')
OUT.close()


########predictions
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        y = dotproduct(w, data[i])
        if (y > 0):
            print("1 ", i)
        else:
            print("0 ", i)
