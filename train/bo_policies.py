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
exp0s = list()#get_aa_good_policies()

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
          operation_pro = sub_policy[2*j+1]
          operation_mag = sub_policy[2*j+2]
          b = (operation_name, operation_pro, operation_mag)
          a.append(b)
      exp0s.append(a)


def construct_good_policies_discrete(policies):
  """AutoAugment policies found on Cifar."""
  global exp0s
  length = len(policies)
  for i in range(length // 5):
      a = []
      sub_policy = policies[i * 5:(i + 1) * 5]
      operations = sub_policy[0] * 1
      for j in range(2):
          if j == 0:
              operation_index = operations // 14
          else:
              operation_index = operations % 14
          operation_name = OPERATION[int(operation_index)]
          operation_pro = str(sub_policy[2 * j + 1])
          operation_mag = str(sub_policy[2 * j + 2])
          operation_pro_t = float(Decimal(operation_pro).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"))
          operation_mag_t = float(Decimal(operation_mag).quantize(Decimal("0."), rounding="ROUND_HALF_UP"))
          b = (operation_name, operation_pro_t, operation_mag_t)
          a.append(b)
      exp0s.append(a)


def good_policies():
    return exp0s

if __name__ == '__main__':
    policies = [74.59492671, 0.36237805, 4.22216973, 0.6892949, 1.11414227,
              184.47725265, 0.95289816, 1.12166779, 0.97034174, 5.90216261,
              166.6672381, 0.97743375, 4.7498256, 0.70251432, 2.70503843,
              170.61345706,   0.37366146,   3.18541466,   0.36306211,   4.61747465,
              105.00565615,   0.47818459,   3.70850263,   0.96498194,   5.7581574,
              87.43685832,   0.87823868,   4.84292846,   0.90731033,   4.50219931,
              147.51363180435672, 0.984372338768757, 5.5285824519706654, 0.6449963388926072, 1.2622244538115026,
              161.48011302433662, 0.9935170037142828, 1.1801603463348727, 0.4658036181615953, 5.3732496764506035,
              105.1724909279528, 0.002275186043534568, 6.481846204028834, 0.6869268226528746, 5.1583644407761415,
              1.52352028e+02, 8.77769322e-01, 2.23426611e+00, 9.65695578e-01, 4.49885070e+00,
              1.61413411e+02, 2.07856827e-01, 3.60585464e+00, 9.31157273e-01, 1.34199690e+00,
              1.71488264e+01, 1.68465268e-01, 9.41388233e-02, 7.04164605e-01, 2.55797013e+00,
              132.410747, 0.0850113539, 4.38492219, 0.662534357, 0.52068686,
              191.437234, 0.45616425, 2.3181548, 0.536892042, 0.745798575,
              170.64691, 0.182257642, 0.51721665, 0.684560009, 6.00586061,
              5.52869348e+01, 1.35400838e-01, 7.75693058e+00, 8.44078671e-01, 2.24554202e+00,
              1.84410906e+02, 8.35614195e-01, 3.53845264e+00, 3.61538332e-01, 3.71963333e+00,
              2.87502624e+01, 8.25932956e-01, 2.30504150e+00, 8.82225343e-02, 2.04503324e+00,
              85.27519470607159, 0.34705151936395495, 6.068544898846506, 0.8997885493449134, 5.056779166575173,
              169.17232208874125, 0.7118675405513857, 6.901967243076479, 0.4892178196162571, 3.8570475099774617,
              37.1324248182375, 0.2053655273130128, 5.61039340388344, 0.3963110846341643, 4.22155715781329,
              39.86747642,   0.37736407,   0.66926481,   0.52113547,   8.93671391,
              173.70769851,   0.3329326,    8.85341563,   0.23311253,   5.0473408,
              47.53902196,   0.95851189,   5.54664908,   0.4071068,    8.6902641]
    p = construct_good_policies(policies)
    q = good_policies()
    print(q)