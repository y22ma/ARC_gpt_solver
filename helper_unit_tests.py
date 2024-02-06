from helper_functions import *

assert get_objects([['a','a','a'],['a','.','a'],['a','a','a']],more_info=False)==[{'tl':(0, 0),'grid':[['a','a','a'],['a','.','a'],['a','a','a']]},{'tl':(1,1),'grid':[['$']]}]
assert get_pixel_coords([['a','a'],['d','f']])=={'a':[(0, 0),(0, 1)],'d':[(1, 0)],'f':[(1, 1)]}
assert empty_grid(3, 2)==[['.','.'], ['.','.'], ['.','.']]
assert crop_grid([['a','a','b'],['.','a','b']],(0, 0),(1, 1))==[['a','a'],['.','a']]
assert tight_fit([['.','.','.'],['.','a','.'],['.','.','.']])==[['a']]
assert combine_object({'tl':(0, 0),'grid':[['a','a'],['a','.']]},{'tl': (1, 1),'grid':[['f']]})=={'tl':(0, 0),'grid':[['a','a'],['a','f']]}
assert rotate_clockwise([['a','b'],['d','e']],90)==[['d','a'],['e','b']]
assert rotate_clockwise([['a','b'],['d','e']],270)==[['b','e'],['a','d']]
assert horizontal_flip([['a','b','c'],['d','e','f']])==[['c','b','a'], ['f','e','d']]
assert vertical_flip([['a','b','c'],['d','e','f']])==[['d','e','f'],['a','b','c']]
assert replace([['a','.'],['a','a']],[['a','a']],[['c','c']])==[['a','.'],['c','c']]
assert change_object_color({'tl':(0,0),'grid':[['a','.']]},'b')=={'tl':(0,0),'grid':[['b','.']]}
assert get_object_color({'tl':(0,0),'grid':[['a','.']]})=='a'
assert fill_object([['.','.'],['.','.']],{'tl':(0, 1),'grid':[['c'],['c']]})==[['.','c'],['.','c']]
assert fill_value([['.','a'],['.','a']],(1,1),'b')==[['.','a'],['.','b']]
assert fill_row([['a','a'],['c','a']],0,'b')==[['b','b'],['c','a']]
assert fill_col([['a','a'],['c','a']],0,'b')==[['b','a'],['b','a']]
assert fill_rect([['a','a'],['c','a']],(0,0),(1,1),'b')==[['b','b'],['b','b']]
assert fill_between_coords([['.','.']],(0,0),(0,1),'a')==[['a','a']]

assert object_contains_color({'tl':(0,0),'grid':[['a']]},'a')==True
assert on_same_line((1,1),(1,2),'row')==True
assert on_same_line((1,1),(2,1),'col')==True
assert on_same_line((1,1),(2,2),'diag')==True
