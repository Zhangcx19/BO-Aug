# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from decimal import Decimal


OPERATION = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Solarize", "Posterize", "Contrast", "Color", "Brightness", "Sharpness", "AutoContrast", "Invert", "Equalize"]
exp0s = list()

def construct_good_policies(policies):
  """AutoAugment policies found on Cifar."""
  global exp0s
  length = len(policies)
  for i in range(length//5):
      a = []
      sub_policy = policies[i*5:(i+1)*5]
      operations = sub_policy[0] * 1
      for j in range(2):
          if j == 0:
              operation_index = operations // 14
          else:
              operation_index = operations % 14
          operation_name = OPERATION[int(operation_index)]
          operation_pro = sub_policy[2 * j + 1]
          operation_mag = sub_policy[2 * j + 2]
          b = (operation_name, operation_pro, operation_mag)

          # operation_pro = str(sub_policy[2*j+1])
          # operation_mag = str(sub_policy[2*j+2])
          # operation_pro_t = float(Decimal(operation_pro).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP"))
          # operation_mag_t = float(Decimal(operation_mag).quantize(Decimal("0."), rounding = "ROUND_HALF_UP"))
          # b = (operation_name, operation_pro_t, operation_mag_t)
          a.append(b)
      exp0s.append(a)

def good_policies():
    return exp0s

def delete_exp0s():
    global exp0s
    exp0s = []

if __name__ == '__main__':
    policies = [109.20106953, 0.68744719, 3.5974516, 0.34782689, 6.69234296,
              137.19266635, 0.28751239, 7.89317224, 0.96544629, 4.98142085,
              74.98771347, 0.99302917, 3.03588378, 0.63758538, 3.09574188,
              9.40884823e+01, 1.35866024e-01, 2.23605996e+00, 8.54421859e-01, 3.39767442e+00,
              5.10162967e+01, 5.51995875e-01, 4.15257666e+00, 1.43381437e-02, 3.70342795e+00,
              2.37403561e+00, 9.46327006e-01, 8.98322018e+00, 5.74013210e-01, 1.93171616e+00,
              1.84314328e+02, 3.05114167e-01, 6.19262659e+00, 6.21558722e-01, 8.88480359e-03,
              5.37311256e+01, 4.52710724e-01, 4.32227023e+00, 9.64577141e-01, 4.67866033e+00,
              1.24243716e+01, 8.46156673e-01, 2.66230654e+00, 5.05679308e-01, 2.53413119e+00,
              1.55676984e+02, 5.95669582e-01, 8.16849530e+00, 7.34205692e-01, 6.58333647e+00,
              1.25925553e+02, 8.95386018e-02, 7.03711050e-01, 5.10269755e-01, 2.39371388e+00,
              1.73103566e+02, 1.96604196e-01, 7.38150520e+00, 3.62209250e-01, 2.98331999e+00,
              16.13123165, 0.53446554, 5.04162728, 0.99780608, 7.44855232,
              174.41834826, 0.18125575, 8.12656322, 0.99633167, 8.75760733,
              93.08033121, 0.59277896, 8.1990227, 0.96245571, 5.89129844,
              1.35014605e+02, 5.03736991e-01, 3.24525604e+00, 9.85316144e-01, 4.19673992e+00,
              4.57709886e+01, 7.71666943e-02, 4.50353200e+00, 8.80749965e-01, 2.26248302e-01,
              2.86799252e+01, 2.53072454e-01, 4.37616276e+00, 3.43747964e-01, 1.70618167e+00,
              165.40196028, 0.71150485, 4.38419689, 0.18726134, 7.09883013,
              187.1155305, 0.77083213, 8.40102888, 0.236383, 1.35294306,
              81.05222221, 0.7149217, 5.33828718, 0.78634629, 6.47043838,
              1.57754846e+02, 2.12182984e-01, 6.92002239e+00, 8.09418927e-03, 6.52557457e+00,
              1.84751368e+02, 1.36123941e-01, 6.05405157e+00, 9.48045261e-01, 3.63816330e+00,
              3.72569965e+01, 1.69496072e-01, 5.57341552e+00, 5.92098224e-01, 1.20581358e+00]
    p = construct_good_policies(policies)
    q = good_policies()
    print(q)