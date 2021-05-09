
import sys
import random
import numpy as np

def uniform_index(index, size):
  return      uniform_index(index + size, size) if index <  0     \
         else uniform_index(index - size, size) if size  <= index \
         else index

def gen_square_lattice(height, width, init = None):
  return [[init for x in range(width)] for y in range(height)]

def calc_contraction(tensor00, tensor01, tensor10, tensor11):
  tensor0 = np.einsum("abcD  , efDh   -> aebfch  ", tensor00, tensor01)
  tensor1 = np.einsum("abcD  , efDh   -> aebfch  ", tensor10, tensor11)
  tensor  = np.einsum("abCDef, CDijkl -> abijekfl", tensor0 , tensor1 )
  return tensor

def tensor2matrix(tensor, indexes0, indexes1):
  tensor_shape = np.array(tensor.shape)
  degree0      = np.prod(tensor_shape[indexes0])
  degree1      = np.prod(tensor_shape[indexes1])
  matrix       = tensor.transpose(indexes0 + indexes1) \
                       .reshape([degree0, degree1])
  return matrix

def calc_svd(matrix, principal_degree):
  unitary_matrix, singular_values, adjoint_matrix = np.linalg.svd(matrix)
  principal_unitary_matrix       = unitary_matrix[:, :principal_degree]
  principal_singular_values      = singular_values[:principal_degree]
  principal_adjoint_matrix       = adjoint_matrix[:principal_degree, :]
  principal_sqrt_singular_values = np.sqrt(principal_singular_values)
  principal_matrix0 =   principal_unitary_matrix \
                      @ np.diag(principal_sqrt_singular_values)
  principal_matrix1 =   np.diag(principal_sqrt_singular_values) \
                      @ principal_adjoint_matrix
  return principal_matrix0, principal_matrix1

def calc_contraction_and_vertical_svd(tensor00, tensor01, \
                                      tensor10, tensor11, \
                                      principal_degree):
  tensor           = calc_contraction(tensor00, tensor01, tensor10, tensor11)
  tensor_shape     = np.array(tensor.shape)
  up_indexes       = [0, 1, 4, 6]
  bottom_indexes   = [2, 3, 5, 7]
  matrix           = tensor2matrix(tensor, up_indexes, bottom_indexes)
  matrix0, matrix1 = calc_svd(matrix, principal_degree)
  up_tensor     = matrix0.reshape([tensor_shape[0] * tensor_shape[1], \
                                   tensor_shape[4], \
                                   tensor_shape[6], \
                                   principal_degree]) \
                         .transpose([0, 3, 1, 2])
  bottom_tensor = matrix1.reshape([principal_degree, \
                                   tensor_shape[2] * tensor_shape[3], \
                                   tensor_shape[5], \
                                   tensor_shape[7]]) \
                         .transpose([0, 1, 2, 3])
  return up_tensor, bottom_tensor

def calc_contraction_and_horizontal_svd(tensor00, tensor01, \
                                        tensor10, tensor11, \
                                        principal_degree):
  tensor           = calc_contraction(tensor00, tensor01, tensor10, tensor11)
  tensor_shape     = np.array(tensor.shape)
  left_indexes     = [0, 2, 4, 5]
  right_indexes    = [1, 3, 6, 7]
  matrix           = tensor2matrix(tensor, left_indexes, right_indexes)
  matrix0, matrix1 = calc_svd(matrix, principal_degree)
  left_tensor  = matrix0.reshape([tensor_shape[0], \
                                  tensor_shape[2], \
                                  tensor_shape[4] * tensor_shape[5], \
                                  principal_degree]) \
                        .transpose([0, 1, 2, 3])
  right_tensor = matrix1.reshape([principal_degree,
                                  tensor_shape[1], \
                                  tensor_shape[3], \
                                  tensor_shape[6] * tensor_shape[7]]) \
                        .transpose([1, 2, 0, 3])
  return left_tensor, right_tensor

length             =   int(sys.argv[1])
inv_temperature    = float(sys.argv[2])
interaction_values =  eval(sys.argv[3])
principal_degree   =   int(sys.argv[4])
random_seed        =   int(sys.argv[5])
random.seed(random_seed)

interactions = np.zeros([length, length, length, length])

for y in range(length):
  for x in range(length):
    bottom_y = uniform_index(y + 1, length)
    bottom_x = uniform_index(x    , length)
    right_y  = uniform_index(y    , length)
    right_x  = uniform_index(x + 1, length)
    bottom_interaction = random.choice(interaction_values)
    right_interaction  = random.choice(interaction_values)
    interactions[y][x][bottom_y][bottom_x] = bottom_interaction
    interactions[bottom_y][bottom_x][y][x] = bottom_interaction
    interactions[y][x][right_y][right_x]   = right_interaction
    interactions[right_y][right_x][y][x]   = right_interaction

delta = np.zeros([2, 2, 2, 2])
delta[0, 0, 0, 0] = 1.0
delta[1, 1, 1, 1] = 1.0

curr_tn_height = length
curr_tn_width  = length
curr_tn        = gen_square_lattice(curr_tn_height, curr_tn_width)

for y in range(length):
  for x in range(length):
    bottom_y = uniform_index(y + 1, curr_tn_height)
    bottom_x = uniform_index(x    , curr_tn_width )
    right_y  = uniform_index(y    , curr_tn_height)
    right_x  = uniform_index(x + 1, curr_tn_width )
    bottom_interaction = interactions[y][x][bottom_y][bottom_x]
    right_interaction  = interactions[y][x][right_y][right_x]
    bottom_same = np.exp(-inv_temperature * bottom_interaction * (+1.0))
    bottom_diff = np.exp(-inv_temperature * bottom_interaction * (-1.0))
    right_same  = np.exp(-inv_temperature * right_interaction  * (+1.0))
    right_diff  = np.exp(-inv_temperature * right_interaction  * (-1.0))
    bottom_matrix = np.array([[bottom_same, bottom_diff], \
                              [bottom_diff, bottom_same]])
    right_matrix  = np.array([[right_same , right_diff ], \
                              [right_diff , right_same ]])
    tensor = np.einsum("aBcD, Bf, Dh -> afch", \
                       delta, bottom_matrix, right_matrix)
    curr_tn[y][x] = tensor

while not (curr_tn_height == 1 or curr_tn_width == 1):
  if curr_tn_height <= curr_tn_width:
    next_tn_height = curr_tn_height
    next_tn_width  = curr_tn_width // 2
    next_tn        = gen_square_lattice(next_tn_height, next_tn_width)
    for curr_y in range(0, curr_tn_height, 2):
      for next_x, curr_x in enumerate(range(1, curr_tn_width, 2)):
        up_y     = uniform_index(curr_y    , curr_tn_height)
        bottom_y = uniform_index(curr_y + 1, curr_tn_height)
        left_x   = uniform_index(curr_x    , curr_tn_width )
        right_x  = uniform_index(curr_x + 1, curr_tn_width )
        up_tensor, bottom_tensor \
          = calc_contraction_and_vertical_svd(curr_tn[up_y    ][left_x ],
                                              curr_tn[up_y    ][right_x],
                                              curr_tn[bottom_y][left_x ],
                                              curr_tn[bottom_y][right_x],
                                              principal_degree)
        next_tn[curr_y    ][next_x] = up_tensor
        next_tn[curr_y + 1][next_x] = bottom_tensor
    curr_tn_height = next_tn_height
    curr_tn_width  = next_tn_width
    curr_tn        = next_tn
  else:
    next_tn_height = curr_tn_height // 2
    next_tn_width  = curr_tn_width
    next_tn        = gen_square_lattice(next_tn_height, next_tn_width)
    for next_y, curr_y in enumerate(range(1, curr_tn_height, 2)):
      for curr_x in range(0, curr_tn_width, 2):
        up_y     = uniform_index(curr_y    , curr_tn_height)
        bottom_y = uniform_index(curr_y + 1, curr_tn_height)
        left_x   = uniform_index(curr_x    , curr_tn_width )
        right_x  = uniform_index(curr_x + 1, curr_tn_width )
        left_tensor, right_tensor \
          = calc_contraction_and_horizontal_svd(curr_tn[up_y    ][left_x ],
                                                curr_tn[up_y    ][right_x],
                                                curr_tn[bottom_y][left_x ],
                                                curr_tn[bottom_y][right_x],
                                                principal_degree)
        next_tn[next_y][curr_x    ] = left_tensor
        next_tn[next_y][curr_x + 1] = right_tensor
    curr_tn_height = next_tn_height
    curr_tn_width  = next_tn_width
    curr_tn        = next_tn

if curr_tn_height == 2 and curr_tn_width == 1:
  result = np.einsum("ABCC, BAGG -> ", curr_tn[0][0], curr_tn[1][0])
  print(result)
else:
  result = np.einsum("AACD, EEDC -> ", curr_tn[0][0], curr_tn[0][1])
  print(result)
