import sys
sys.path.append('./scripts')

from utils_win_cube_copy import CascadasFast
cascada = CascadasFast(cube_shape_x=1500, projection="y", 
                           win_shape=(62, 62, 128))