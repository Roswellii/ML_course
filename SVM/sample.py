import sys

data_x = [{1:0.697, 2:0.46}, {1:0.774, 2:0.376}, {1:0.634, 2:0.264}, {1:0.608, 2:0.318},
          {1:0.556, 2:0.215}, {1:0.403, 2:0.237}, {1:0.481, 2:0.149}, {1:0.437, 2:0.211},
          {1:0.666, 2:0.091}, {1:0.243, 2:0.267}, {1:0.245, 2:0.057}, {1:0.343, 2:0.099},
          {1:0.639, 2:0.161}, {1:0.657, 2:0.198}, {1:0.36, 2:0.37}, {1:0.593, 2:0.042},
          {1:0.719, 2:0.103},]
data_y = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


from svm import svm
from svm import svmutil

print('线性核：')

# 线性核
c_acc = 0
for c_param in range(1, 10000, 100):
    prob = svm.svm_problem(data_y, data_x, isKernel=True)
    param = svm.svm_parameter('-t 0 -c %d -q' % c_param)
    model = svmutil.svm_train(prob, param, '-q')
    p_label, p_acc, p_val = svmutil.svm_predict(data_y, data_x, model, '-q')
    if p_acc[0] >= 70 and p_acc[0] > c_acc:
        c_acc = p_acc[0]
        print(c_acc)

print('\n高斯核：')

# 高斯核
c_acc = 10000
for c_param in range(1, 10000, 100):
    prob = svm.svm_problem(data_y, data_x, isKernel=True)
    param = svm.svm_parameter('-t 2 -c %d -q' % c_param)
    model = svmutil.svm_train(prob, param, '-q')
    p_label, p_acc, p_val = svmutil.svm_predict(data_y, data_x, model, '-q')
    if p_acc[0] >= 70 and p_acc[0] > c_acc:
        c_acc = p_acc[0]
        print(c_acc)
